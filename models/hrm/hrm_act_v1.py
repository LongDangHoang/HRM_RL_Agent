from typing import Tuple, List
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    SwiGLU,
    Attention,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: TensorDictBase  # Current data is the non-halted elements in the batch, used for loss computation and logging

    def to(self, *args, **kwargs):
        self.inner_carry.z_H = self.inner_carry.z_H.to(*args, **kwargs)
        self.inner_carry.z_L = self.inner_carry.z_L.to(*args, **kwargs)
        self.steps = self.steps.to(*args, **kwargs)
        self.halted = self.halted.to(*args, **kwargs)
        self.current_data = self.current_data.to(*args, **kwargs)
        return self


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    max_memory_size: int = 1
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    train_halt_head: bool = True
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"

    @property
    def effective_seq_len(self):
        return self.seq_len * self.max_memory_size


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states
            + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps
        )
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(
            self.config.puzzle_emb_ndim // -self.config.hidden_size
        )  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.effective_seq_len
                + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.effective_seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.H_layers)
            ]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.L_layers)
            ]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(
                        -1, self.puzzle_emb_len, self.config.hidden_size
                    ),
                    embedding,
                ),
                dim=-2,
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        # Scale
        return self.embed_scale * embedding

    @property
    def device(self):
        return self.H_init.device

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.effective_seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=self.H_init.device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.effective_seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=self.L_init.device,
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        h_init: torch.Tensor,
        l_init: torch.Tensor,
    ):
        if len(h_init.shape) != 1:
            assert (
                h_init.shape[0] == reset_flag.shape[0]
                and h_init.shape[1] == self.config.hidden_size
            )
            assert (
                l_init.shape[0] == reset_flag.shape[0]
                and l_init.shape[1] == self.config.hidden_size
            )
            h_init = h_init.unsqueeze(1)
            l_init = l_init.unsqueeze(1)

        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), h_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), l_init, carry.z_L),
        )

    def forward(
        self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: TensorDictBase
    ) -> Tuple[
        HierarchicalReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not (
                        (_H_step == self.config.H_cycles - 1)
                        and (_L_step == self.config.L_cycles - 1)
                    ):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    @property
    def device(self):
        return self.inner.device

    def initial_carry(self, batch: TensorDictBase):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data=TensorDict(
                {k: torch.empty_like(v) for k, v in batch.items()},
                batch_size=batch.batch_size,
            ),
        )

    def forward(
        self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: TensorDictBase
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, TensorDictBase]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(
            carry.halted,
            carry.inner_carry,
            self.inner.H_init if ("seed_h_init" not in batch) else batch["seed_h_init"],
            self.inner.L_init if ("seed_l_init" not in batch) else batch["seed_l_init"],
        )

        new_steps = torch.where(
            carry.halted, 0, carry.steps
        )  # Sequence halted due to reaching max computation steps are reset to 0 to allow training to continue

        new_current_data = TensorDict(
            {
                k: (
                    torch.where(
                        carry.halted.view(
                            (-1,) + (1,) * (batch[k].ndim - 1)
                        ),  # view shape = (-1, 1, ..., 1)
                        batch[k],  # If halted, start again from the data of the batch
                        v,  # If not halted, use the data of the carry (which contains data from previous batches)
                    )
                )
                for k, v in carry.current_data.items()
            },
            batch_size=batch.batch_size,
            device=carry.current_data.device,
        )

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = TensorDict(
            {
                "logits": logits,
                "q_halt_logits": q_halt_logits,
                "q_continue_logits": q_continue_logits,
            },
            batch_size=batch.batch_size,
            device=logits.device,
        )

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if (
                self.training
                and (self.config.halt_max_steps > 1)
                and self.config.train_halt_head
            ):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(
                    new_inner_carry, new_current_data
                )[-1]

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return HierarchicalReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
