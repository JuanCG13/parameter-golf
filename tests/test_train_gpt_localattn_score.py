import unittest

import torch

import train_gpt_localattn_score as score


class TrainGptLocalAttnScoreTests(unittest.TestCase):
    def test_aggressive_score_defaults(self) -> None:
        self.assertEqual(score.Hyperparameters.train_seq_len, 1024)
        self.assertEqual(score.Hyperparameters.local_attn_window, 768)
        self.assertEqual(score.Hyperparameters.global_tokens, 32)
        self.assertEqual(score.Hyperparameters.num_layers, 4)
        self.assertEqual(score.Hyperparameters.model_dim, 640)
        self.assertEqual(score.Hyperparameters.num_heads, 10)
        self.assertEqual(score.Hyperparameters.num_kv_heads, 5)
        self.assertEqual(score.Hyperparameters.mlp_hidden_dim, 1536)
        self.assertEqual(score.Hyperparameters.train_batch_tokens, 131_072)

    def test_local_global_mask_keeps_global_prefix_and_local_window(self) -> None:
        q_idx = torch.arange(6)[:, None]
        kv_idx = torch.arange(6)[None, :]
        allowed = score.local_global_mask(local_window=3, global_tokens=2, q_idx=q_idx, kv_idx=kv_idx)
        expected = torch.tensor(
            [
                [True, False, False, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
                [True, True, True, True, False, False],
                [True, True, True, True, True, False],
                [True, True, False, True, True, True],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(allowed, expected))

    def test_estimated_param_count_matches_model(self) -> None:
        config = dict(
            vocab_size=1024,
            num_layers=4,
            model_dim=640,
            num_heads=10,
            num_kv_heads=5,
            mlp_hidden_dim=1536,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            local_window=768,
            global_tokens=32,
        )
        model = score.LocalAttentionGPT(**config)
        expected = score.estimate_model_params(
            vocab_size=config["vocab_size"],
            num_layers=config["num_layers"],
            model_dim=config["model_dim"],
            num_heads=config["num_heads"],
            num_kv_heads=config["num_kv_heads"],
            mlp_hidden_dim=config["mlp_hidden_dim"],
            tie_embeddings=config["tie_embeddings"],
        )
        actual = sum(param.numel() for param in model.parameters())
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
