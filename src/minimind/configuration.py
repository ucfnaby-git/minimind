import math
from typing import Any

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        # Model architecture
        vocab_size: int = 6400,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int | None = 4,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        hidden_act: str = "silu",
        # Normalization and regularization
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        # Positional embeddings
        max_position_embeddings: int = 32768,
        rope_theta: float = 1e6,
        inference_rope_scaling: bool = False,
        rope_scaling: dict[str, Any] | None = None,
        # Attention implementation
        flash_attn: bool = True,
        # Mixture of Experts
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        # Hugging Face standard parameters
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Model architecture
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        if head_dim is None:
            self.head_dim = hidden_size // num_attention_heads
        else:
            self.head_dim = head_dim

        if intermediate_size is None:
            self.intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
        else:
            self.intermediate_size = intermediate_size

        self.hidden_act = hidden_act

        # Normalization and regularization
        self.rms_norm_eps = rms_norm_eps
        self.dropout = dropout

        # Positional embeddings
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling

        default_rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn",
        }

        self.rope_scaling = (
            rope_scaling
            if rope_scaling is not None
            else (default_rope_scaling if inference_rope_scaling else None)
        )

        # Attention implementation
        self.flash_attn = flash_attn

        # Mixture of Experts
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        if moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
        else:
            self.moe_intermediate_size = moe_intermediate_size

        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        self._validate()

    def _validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")

        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")

        kv_heads = (
            self.num_key_value_heads
            if self.num_key_value_heads is not None
            else self.num_attention_heads
        )

        if kv_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")

        if self.num_attention_heads % kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        if self.moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be positive")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        if self.use_moe and not (1 <= self.num_experts_per_tok <= self.num_experts):
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
