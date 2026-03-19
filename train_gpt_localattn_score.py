"""
Score-oriented local-attention trainer.

This wrapper keeps the local+global masked-attention implementation from
`train_gpt_localattn.py`, but swaps in wider, shallower defaults aimed at
using more of a 40 GB pod and more of the 16 MB artifact budget.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import train_gpt_localattn as localattn

local_global_mask = localattn.local_global_mask
build_local_global_attn_bias = localattn.build_local_global_attn_bias
LocalGlobalCausalSelfAttention = localattn.LocalGlobalCausalSelfAttention
SwiGLUMLP = localattn.SwiGLUMLP
Block = localattn.Block
LocalAttentionGPT = localattn.LocalAttentionGPT


def estimate_model_params(
    *,
    vocab_size: int,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_hidden_dim: int,
    tie_embeddings: bool,
) -> int:
    if model_dim % num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    kv_dim = model_dim * num_kv_heads // num_heads
    attention_params = 2 * model_dim * model_dim + 2 * model_dim * kv_dim
    mlp_params = 3 * model_dim * mlp_hidden_dim
    embedding_params = vocab_size * model_dim
    lm_head_params = 0 if tie_embeddings else vocab_size * model_dim
    return embedding_params + num_layers * (attention_params + mlp_params) + lm_head_params


class Hyperparameters(localattn.Hyperparameters):
    # Score runs should default to the full dataset path and all available shards.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    max_train_shards = int(os.environ.get("MAX_TRAIN_SHARDS", 0))
    run_id = os.environ.get("RUN_ID", f"localattn_score_{uuid.uuid4()}")

    # Spend the wallclock on learning, not on frequent evals.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1_200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 131_072))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Keep local+global attention as the main recipe, but with a wider window.
    local_attn_window = int(os.environ.get("LOCAL_ATTN_WINDOW", 768))
    global_tokens = int(os.environ.get("GLOBAL_TOKENS", 32))
    use_flex_attention = bool(int(os.environ.get("USE_FLEX_ATTENTION", "1")))
    flex_block_size = int(os.environ.get("FLEX_BLOCK_SIZE", 128))

    # Wider, shallower profile to reduce per-token layer hopping while staying
    # closer to the proven ~16 MB artifact regime from the repo baselines.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 10))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 5))
    model_dim = int(os.environ.get("MODEL_DIM", 640))
    mlp_hidden_dim = int(os.environ.get("MLP_HIDDEN_DIM", 1536))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))

    # Reuse the baseline-style optimizer defaults for score-oriented runs.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


def _build_artifact_source() -> str:
    wrapper_code = Path(__file__).read_text(encoding="utf-8")
    shared_code = Path(localattn.__file__).read_text(encoding="utf-8")
    return (
        wrapper_code
        + "\n\n# --- shared dependency: train_gpt_localattn.py ---\n"
        + shared_code
    )


def main() -> None:
    artifact_path: str | None = None
    original_hparams = localattn.Hyperparameters
    original_file = localattn.__file__

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_localattn_score_artifact.py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(_build_artifact_source())
            artifact_path = tmp.name

        localattn.Hyperparameters = Hyperparameters
        localattn.__file__ = artifact_path
        localattn.main()
    finally:
        localattn.Hyperparameters = original_hparams
        localattn.__file__ = original_file
        if artifact_path is not None:
            Path(artifact_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
