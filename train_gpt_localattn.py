"""
Experimental trainer for a larger local-attention architecture.

This file keeps the existing training/eval/quantization stack from train_gpt.py,
but swaps in a plain 8-layer transformer with local+global causal attention and
SwiGLU MLPs. Defaults are tuned for quick experiments on the 7-shard subset.
"""

from __future__ import annotations

import copy
import io
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

import train_gpt as base

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:
    create_block_mask = None
    flex_attention = None


class Hyperparameters:
    # Data / tokenizer.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024_7shards")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    max_train_shards = int(os.environ.get("MAX_TRAIN_SHARDS", 7))
    run_id = os.environ.get("RUN_ID", f"localattn_{uuid.uuid4()}")
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 196_608))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 1000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 100))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 5))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 24_576))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 768))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))

    # Attention layout.
    local_attn_window = int(os.environ.get("LOCAL_ATTN_WINDOW", 512))
    global_tokens = int(os.environ.get("GLOBAL_TOKENS", 16))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    use_flex_attention = bool(int(os.environ.get("USE_FLEX_ATTENTION", "1")))
    flex_block_size = int(os.environ.get("FLEX_BLOCK_SIZE", 128))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 8))
    num_heads = int(os.environ.get("NUM_HEADS", 16))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 2048))
    mlp_hidden_dim = int(os.environ.get("MLP_HIDDEN_DIM", 5120))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters. These are slightly softer than the baseline because
    # this model is much larger and is intended for short, local experiments.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))


def local_global_mask(
    local_window: int,
    global_tokens: int,
    q_idx: Tensor,
    kv_idx: Tensor,
) -> Tensor:
    causal = kv_idx <= q_idx
    local_start = torch.clamp(q_idx - local_window + 1, min=0)
    local = kv_idx >= local_start
    global_mask = kv_idx < global_tokens
    allowed = causal & (local | global_mask)
    return allowed


def build_local_global_attn_bias(
    seq_len: int,
    local_window: int,
    global_tokens: int,
    device: torch.device,
) -> Tensor:
    q_idx = torch.arange(seq_len, device=device)
    k_idx = torch.arange(seq_len, device=device)
    allowed = local_global_mask(local_window, global_tokens, q_idx[:, None], k_idx[None, :])
    bias = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
    bias.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    return bias[None, None, :, :]


class LocalGlobalCausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        local_window: int,
        global_tokens: int,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.local_window = local_window
        self.global_tokens = global_tokens
        self.use_flex_attention = create_block_mask is not None and flex_attention is not None
        self.flex_block_size = 128
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = base.CastedLinear(dim, dim, bias=False)
        self.c_k = base.CastedLinear(dim, kv_dim, bias=False)
        self.c_v = base.CastedLinear(dim, kv_dim, bias=False)
        self.proj = base.CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = base.Rotary(self.head_dim, base=rope_base)
        self._block_mask_cache: dict[tuple[int, torch.device], object] = {}
        self._attn_bias_cache: dict[tuple[int, torch.device], Tensor] = {}

    def configure_backend(self, *, use_flex_attention: bool, flex_block_size: int) -> None:
        self.use_flex_attention = use_flex_attention and create_block_mask is not None and flex_attention is not None
        self.flex_block_size = flex_block_size

    def _get_block_mask(self, seq_len: int, device: torch.device):
        key = (seq_len, device)
        if key not in self._block_mask_cache:
            if create_block_mask is None:
                raise RuntimeError("flex_attention BlockMask utilities are unavailable")
            local_window = self.local_window
            global_tokens = self.global_tokens

            def mask_mod(batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
                del batch, head
                return local_global_mask(local_window, global_tokens, q_idx, kv_idx)

            self._block_mask_cache[key] = create_block_mask(
                mask_mod,
                1,
                self.num_heads,
                seq_len,
                seq_len,
                device=device,
                BLOCK_SIZE=self.flex_block_size,
            )
        return self._block_mask_cache[key]

    def _get_math_bias(self, seq_len: int, device: torch.device) -> Tensor:
        key = (seq_len, device)
        if key not in self._attn_bias_cache:
            self._attn_bias_cache[key] = build_local_global_attn_bias(
                seq_len,
                local_window=self.local_window,
                global_tokens=self.global_tokens,
                device=device,
            )
        return self._attn_bias_cache[key]

    def prepare_attention(self, seq_len: int, device: torch.device) -> None:
        if self.use_flex_attention:
            self._get_block_mask(seq_len, device)
        else:
            self._get_math_bias(seq_len, device)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = base.apply_rotary_emb(q, cos, sin)
        k = base.apply_rotary_emb(k, cos, sin)
        if self.use_flex_attention:
            block_mask = self._get_block_mask(seqlen, x.device)
            y = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        else:
            attn_bias = self._get_math_bias(seqlen, x.device)
            with sdpa_kernel(SDPBackend.MATH):
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_bias,
                    is_causal=False,
                    enable_gqa=(self.num_kv_heads != self.num_heads),
                )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = base.CastedLinear(dim, hidden_dim, bias=False)
        self.up = base.CastedLinear(dim, hidden_dim, bias=False)
        self.proj = base.CastedLinear(hidden_dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        local_window: int,
        global_tokens: int,
        mlp_hidden_dim: int,
    ):
        super().__init__()
        self.attn_norm = base.RMSNorm()
        self.mlp_norm = base.RMSNorm()
        self.attn = LocalGlobalCausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_base=rope_base,
            local_window=local_window,
            global_tokens=global_tokens,
        )
        self.mlp = SwiGLUMLP(dim, mlp_hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LocalAttentionGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_hidden_dim: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        local_window: int,
        global_tokens: int,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=model_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    rope_base=rope_base,
                    local_window=local_window,
                    global_tokens=global_tokens,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = base.RMSNorm()
        self.lm_head = None if tie_embeddings else base.CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def configure_attention_backend(self, *, use_flex_attention: bool, flex_block_size: int) -> None:
        for block in self.blocks:
            block.attn.configure_backend(
                use_flex_attention=use_flex_attention,
                flex_block_size=flex_block_size,
            )

    def prepare_attention(self, seq_len: int, device: torch.device) -> None:
        for block in self.blocks:
            block.attn.prepare_attention(seq_len, device)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.train_seq_len <= args.local_attn_window:
        raise ValueError(
            f"TRAIN_SEQ_LEN={args.train_seq_len} must be larger than LOCAL_ATTN_WINDOW={args.local_attn_window} "
            "so local attention is not equivalent to full causal attention"
        )
    if args.global_tokens <= 0:
        raise ValueError(f"GLOBAL_TOKENS must be positive, got {args.global_tokens}")
    if args.global_tokens >= args.train_seq_len:
        raise ValueError(
            f"GLOBAL_TOKENS={args.global_tokens} must be smaller than TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    base.zeropower_via_newtonschulz5 = torch.compile(base.zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and world_size > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if args.grad_accum_steps <= 0:
        raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {args.grad_accum_steps}")
    grad_accum_steps = args.grad_accum_steps if not distributed or "GRAD_ACCUM_STEPS" in os.environ else max(1, 8 // world_size)
    grad_scale = 1.0 / grad_accum_steps
    local_train_tokens = args.train_batch_tokens // (world_size * grad_accum_steps)
    if local_train_tokens < args.train_seq_len:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must provide at least one sequence per rank and micro-step; "
            f"got TRAIN_BATCH_TOKENS={args.train_batch_tokens}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    active_train_files = min(actual_train_files, args.max_train_shards) if args.max_train_shards > 0 else actual_train_files
    val_tokens = base.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(
        f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files} "
        f"active_train_shards:{active_train_files}"
    )
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------
    base_model = LocalAttentionGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        local_window=args.local_attn_window,
        global_tokens=args.global_tokens,
    ).to(device).bfloat16()
    base_model.configure_attention_backend(
        use_flex_attention=args.use_flex_attention,
        flex_block_size=args.flex_block_size,
    )
    base_model.prepare_attention(args.train_seq_len, device)
    for module in base_model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
    base.restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok]
    optimizer_head: torch.optim.Optimizer | None = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_head)
    optimizer_muon: base.Muon | None = None
    if matrix_params:
        optimizer_muon = base.Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        optimizers.append(optimizer_muon)
    optimizer_scalar: torch.optim.Optimizer | None = None
    if scalar_params:
        optimizer_scalar = torch.optim.Adam(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_scalar)

    n_params = sum(p.numel() for p in base_model.parameters())
    local_batch_seqs = local_train_tokens // args.train_seq_len
    log0(f"model_params:{n_params}")
    log0(
        f"world_size:{world_size} distributed:{distributed} grad_accum_steps:{grad_accum_steps} "
        f"local_batch_seqs:{local_batch_seqs}"
    )
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:local_global num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"local_window:{args.local_attn_window} global_tokens:{args.global_tokens} "
        f"backend:{'flex_attention' if args.use_flex_attention and create_block_mask is not None and flex_attention is not None else 'sdpa_math'}"
    )
    log0(
        f"mlp:swi_glu hidden_dim:{args.mlp_hidden_dim} tie_embeddings:{args.tie_embeddings} "
        f"seq_len:{args.train_seq_len}"
    )
    log0(
        f"embed_lr:{token_lr} head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------
    train_loader = base.DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        max_files=args.max_train_shards,
    )

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = base.DistributedTokenLoader(
            args.train_files,
            rank,
            world_size,
            device,
            max_files=args.max_train_shards,
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = base.eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        if optimizer_muon is not None:
            frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
            muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = base.quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(base.dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = base.eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
