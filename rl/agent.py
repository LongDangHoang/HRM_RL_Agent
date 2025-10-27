import torch

from jaxtyping import Float

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torch import Tensor

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Carry,
)
from rl.interfaces import HRMQNetTrainingConfig


class HRMQValueNet(TensorDictModuleBase):
    """
    A network that computes Q-values for a given state and action using the Hierarchical Reasoning Model (HRM).
    """

    def __init__(self, config: HRMQNetTrainingConfig):
        super().__init__()
        self.in_keys = ["inputs"]
        self.out_keys = ["action_value", "puzzle_identifiers"]
        self.config = config
        self.model = HierarchicalReasoningModel_ACTV1(config.arch_config_dict)
        self.carry: HierarchicalReasoningModel_ACTV1Carry | None = None

        if self.config.use_last_hidden_state_to_seed_next_environment_step:
            self.out_keys.extend(
                ["seed_h_init", "seed_l_init", "prev_seed_h_init", "prev_seed_l_init"]
            )

    def _seed_puzzle_identifiers(self, batch_data: TensorDict) -> TensorDict:
        if "puzzle_identifiers" not in batch_data:
            batch_data["puzzle_identifiers"] = torch.zeros(
                batch_data.batch_size, dtype=torch.long, device=batch_data.device
            )
        return batch_data

    def _initial_carry(
        self, batch_data: TensorDict
    ) -> HierarchicalReasoningModel_ACTV1Carry:
        batch_data = self._seed_puzzle_identifiers(batch_data)
        return self.model.initial_carry(batch_data)

    def _q_values_with_carry(
        self,
        batch_data: TensorDict,
        carry: HierarchicalReasoningModel_ACTV1Carry | None,
    ) -> tuple[
        HierarchicalReasoningModel_ACTV1Carry, Float[Tensor, "batch action_space_size"]
    ]:
        if carry is None:
            carry = self._initial_carry(batch_data).to(self.model.device)
        batch_data = self._seed_puzzle_identifiers(batch_data)
        carry, outputs = self.model.forward(carry=carry, batch=batch_data)
        logits = outputs[
            "logits"
        ]  # called logits because the formulation is a linear transform of the hidden dimensions back to the vocabulary size, shape (batch_size, num_observations * sequence_length, vocab_size)
        q_values = logits[
            :, -1, : self.config.dataset.action_space_size
        ]  # take the first action_space_size dimensions, last token as the Q-values. Note this linear layer has no bias and wastes a lot of parameters, but simplifies code a bit for now
        return carry, q_values

    def q_values_on_current_carry(
        self, batch_data: TensorDict
    ) -> tuple[
        HierarchicalReasoningModel_ACTV1Carry, Float[Tensor, "batch action_space_size"]
    ]:
        """
        Compute Q-values for all actions given a state.

        Note this depends on the current "carry" of the model, which contains data the model is operating on recurrently. The model will ignore the states and actions if the
        current carry is not halted. The carry is also updated after the operation
        """
        self.carry, q_values = self._q_values_with_carry(batch_data, self.carry)
        return self.carry, q_values

    def q_values_on_new_carry(
        self, batch_data: TensorDict
    ) -> tuple[
        HierarchicalReasoningModel_ACTV1Carry, Float[Tensor, "batch action_space_size"]
    ]:
        """
        Compute Q-values for a given state and action.

        This runs from an initial carry until the end of the sequence, which is useful for prediction. This has no gradient flow by default as backpropagation through time is untested in the HRM model.
        """
        q_values = None
        current_carry = self._initial_carry(batch_data).to(self.model.device)
        is_first_carry = True  # first carry is all halted by design so the data is simply copied from batch data
        while not current_carry.halted.all() or is_first_carry:
            current_carry, q_values = self._q_values_with_carry(
                carry=current_carry, batch_data=batch_data
            )
            is_first_carry = False

        if q_values is None:
            raise ValueError("No q-values computed as initial carry were all halted")

        return current_carry, q_values

    def forward(self, batch_data: TensorDict) -> TensorDict:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Forward starts from initial carry by default, used mostly during inference"""
        if self.config.use_last_hidden_state_to_seed_next_environment_step:
            initial_seed_h_init = (
                (
                    batch_data["seed_h_init"]
                    if "seed_h_init" in batch_data
                    else self.model.inner.H_init.unsqueeze(0).expand(
                        (batch_data.batch_size[0], -1)
                    )
                )
                .detach()
                .clone()
            )
            initial_seed_l_init = (
                (
                    batch_data["seed_l_init"]
                    if "seed_l_init" in batch_data
                    else self.model.inner.L_init.unsqueeze(0).expand(
                        (batch_data.batch_size[0], -1)
                    )
                )
                .detach()
                .clone()
            )

        if self.model.training:
            carry, action_values = self.q_values_on_current_carry(batch_data)
        else:
            carry, action_values = self.q_values_on_new_carry(batch_data)

        out_tensor = TensorDict(
            {
                **{k: v for k, v in batch_data.items() if k not in ("action_value",)},
                "action_value": action_values,
            },
            batch_data.batch_size,
            device=batch_data.device,
        )

        if self.config.use_last_hidden_state_to_seed_next_environment_step:
            out_tensor["seed_h_init"] = carry.inner_carry.z_H[:, -1, :].detach().clone()
            out_tensor["seed_l_init"] = carry.inner_carry.z_L[:, -1, :].detach().clone()
            out_tensor["prev_seed_h_init"] = initial_seed_h_init
            out_tensor["prev_seed_l_init"] = initial_seed_l_init

        return out_tensor
