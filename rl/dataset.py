from tensordict import TensorDictBase
from torch.utils.data import IterableDataset
from typing import Callable
from rl.interfaces import GymRLDatasetConfig, evaluating_net_context
from rl.environment import REGISTER_ALL_IMPORT
from utils.minihack_rect import MiniHackMapRect


import torch
from tensordict.nn import TensorDictModuleBase
from torchrl.data import (
    Composite,
    LazyTensorStorage,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.envs import (
    Compose,
    EnvBase,
    FlattenObservation,
    GymEnv,
    ObservationTransform,
    RenameTransform,
    SerialEnv,
    TransformedEnv,
    InitTracker,
    StepCounter,
)
from torchrl.modules import EGreedyModule, QValueActor


import random


assert REGISTER_ALL_IMPORT  # to avoid linting error

class SimpleCollector:
    """
    Simple collector that handles rolling out env according to policy and storing it in a buffer

    TorchRL collector seems powerful but it may be a bit too complicated, so we will just handroll it somewhat simply
    Will likely be super inefficient comparec to torchRL but should work... hopefully
    """

    def __init__(
        self,
        dataset: "GymRLDataset",
        buffer: TensorDictReplayBuffer,
        policy: TensorDictModuleBase,
        exploration_module: EGreedyModule,
    ):
        self.buffer = buffer
        self.policy = policy
        self.exploration_module = exploration_module
        self.dataset = dataset

        self.num_batches_per_update = (
            self.dataset.config.frames_per_update
            // self.dataset.config.data_collection_batch_size
        )
        assert (
            self.dataset.config.frames_per_update
            % self.dataset.config.data_collection_batch_size
            == 0
        )

    def __iter__(self):
        # set up batch_size envs so that our policy which runs on cuda can be efficient
        envs = SerialEnv(
            self.dataset.config.data_collection_batch_size,
            self.dataset.create_env,
        )
        inner_current_state = envs.reset()

        while True:
            for _ in range(self.num_batches_per_update):
                if self.dataset.config.do_not_skip_running_model_if_random_action:
                    with torch.no_grad(), evaluating_net_context(self.policy):
                        policy_decision = self.policy(
                            inner_current_state.to(self.policy.device)
                        )
                    policy_decision = self.exploration_module(policy_decision)
                elif random.random() < self.exploration_module.eps:
                    policy_decision = envs.rand_action(inner_current_state)
                else:
                    with torch.no_grad(), evaluating_net_context(self.policy):
                        policy_decision = self.policy(
                            inner_current_state.to(self.policy.device)
                        )

                # step the environmet
                transitions, inner_current_state = envs.step_and_maybe_reset(
                    policy_decision.cpu()
                )

                # reset all seed_h_init to the current policy h_init if the env resets
                if "seed_h_init" in inner_current_state:
                    env_resetted = inner_current_state["is_init"][:, 0] == True
                    inner_current_state["seed_h_init"][env_resetted, :] = self.policy[
                        0
                    ].model.inner.H_init.to(
                        inner_current_state.device
                    )  # TODO: this is not very generalised but hard to handle otherwise
                    inner_current_state["seed_l_init"][env_resetted, :] = self.policy[
                        0
                    ].model.inner.L_init.to(inner_current_state.device)

                # chug that in the buffer
                self.buffer.extend(transitions)

            # yield to dataset
            yield


class GymRLDataset(IterableDataset):
    def __init__(self, config: GymRLDatasetConfig):
        self.config = config
        self.create_env = self.make_create_env_func()
        self.base_env = self.create_env()
        self._is_training = True

        # Replay buffer
        self.buffer = self.create_replay_buffer()

    def convert_to_canonical(self, env_step_return: TensorDictBase):
        return env_step_return

    def make_create_env_func(self) -> Callable[[], EnvBase]:
        """Must generate a picklable-method for use for SerialEnv"""
        args = {
            "env_name": self.config.env_name,
            "device": self.config.env_device,
            **self.config.env_kwargs,
        }

        def __make_env():
            return GymEnv(**args)

        return __make_env

    def create_replay_buffer(self):
        replay_buffer = TensorDictReplayBuffer(
            batch_size=self.config.training_batch_size,
            storage=LazyTensorStorage(self.config.buffer_capacity),
            prefetch=self.config.num_workers,
            transform=lambda td: td.to(self.config.storing_device),
        )
        return replay_buffer

    def create_collector(
        self,
        buffer: TensorDictReplayBuffer,
        policy: TensorDictModuleBase,
        exploration_module: EGreedyModule,
    ) -> SimpleCollector:
        # while torch rl has a collector interface, its internals seem a bit hard to wrap your head around
        # so for simplicity we will make something that rollout env for some number of steps using policy
        # and yielding observations
        return SimpleCollector(self, buffer, policy, exploration_module)

    def initialise_policy_and_collector(
        self, actor: QValueActor, exploration_module: EGreedyModule
    ):
        self.actor = actor
        self.exploration_module = exploration_module
        self.collector = self.create_collector(
            self.buffer, self.actor, self.exploration_module
        )

    def __iter__(self):
        yield from self._iter_train()

    def _iter_train(self):
        """
        Collect experiences into a buffer then sample for training
        """

        assert self.collector is not None, "Data collector is not initialised"

        # step through the collector to gather data
        collector_iterator = iter(self.collector)
        next(collector_iterator)

        batches_since_last_data_collected = 0
        while True:
            if batches_since_last_data_collected == self.config.updates_per_data:
                next(collector_iterator)
                batches_since_last_data_collected = 0

            batch = self.buffer.sample(self.config.training_batch_size)
            batches_since_last_data_collected += 1
            yield batch

    def validation_rollout(self):
        """
        Roll out a batch-size number of environments to its conclusion under a policy
        """
        assert self.actor is not None, "Actor is not initialised"
        envs = SerialEnv(
            self.config.data_collection_batch_size,
            self.create_env,
            device=self.config.env_device,
        )

        with torch.no_grad(), evaluating_net_context(self.actor):
            return envs.rollout(
                max_steps=self.config.max_episode_steps,
                policy=self.actor,
                auto_cast_to_device=True,
                break_when_any_done=False,
                break_when_all_done=True,
            )


class MiniHackFilterInputTransform(ObservationTransform):
    def __init__(
        self,
        first_row: int = 0,
        last_row: int = 21,
        first_col: int = 0,
        last_col: int = 79,
    ):
        super().__init__(in_keys=["chars"], out_keys=["chars"])
        self.rect = MiniHackMapRect(
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        observation = observation[
            ...,
            self.rect.first_row : self.rect.last_row,
            self.rect.first_col : self.rect.last_col,
        ]
        return observation

    forward = ObservationTransform._call

    def transform_observation_spec(
        self, observation_spec: TensorSpec | Composite
    ) -> TensorSpec | Composite:
        def __spec_transform(observation_spec: TensorSpec):
            observation_spec = self._apply_transform(observation_spec)  # pyright: ignore[reportAssignmentType, reportArgumentType]
            return observation_spec

        if isinstance(observation_spec, Composite):
            _specs = observation_spec._specs
            _specs["chars"] = __spec_transform(observation_spec["chars"].clone())
            return Composite(
                _specs,
                shape=observation_spec.shape,
                device=observation_spec.device,  # pyright: ignore[reportArgumentType]
            )
        else:
            return __spec_transform(observation_spec)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # with _set_missing_tolerance(self, True):
        return self._call(tensordict_reset)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f'first_row={self.rect.first_row}, last_row={self.rect.last_row}, first_col={self.rect.first_col}, last_col={self.rect.last_col}, in_keys=["chars"], out_keys=["chars"]'
            ")"
        )


class MiniHackFullObservationSimpleEnvironmentDataset(GymRLDataset):
    VOCAB_CHARS = list(range(32, 127))  # ASCII characters
    DUNGEON_SHAPE = (21, 79)

    def __init__(self, config: GymRLDatasetConfig):
        super().__init__(config)
        self.vocab_size = 127 + self.config.action_space_size

        # check aligning with config
        assert self.vocab_size == self.config.vocab_size, (
            f"Got mismatch vocab size. Expected from config: {self.config.vocab_size}; got {self.vocab_size}"
        )

    def make_create_env_func(self) -> Callable[[], EnvBase]:
        gym_env_args = dict(
            env_name=self.config.env_name,
            device=self.config.env_device,
            max_episode_steps=self.config.max_episode_steps,
            observation_keys=["chars"],
        )
        gym_env_args.update(
            **self.config.env_kwargs,
        )

        if "5x5" in self.config.env_name:
            first_row, last_row, first_col, last_col = 9, 14, 36, 41
        elif "9x9" in self.config.env_name:
            first_row, last_row, first_col, last_col = 7, 16, 34, 43
        elif (
            "Corridor-Maze" in self.config.env_name
        ):  # functionally 11x13 with some offset
            first_row, last_row, first_col, last_col = 5, 16, 38, 51
        elif "4-Rooms" in self.config.env_name:  # functionally 11x11
            first_row, last_row, first_col, last_col = 6, 17, 39, 50
        elif "15x15" in self.config.env_name:
            first_row, last_row, first_col, last_col = 3, 18, 32, 47
        else:
            raise ValueError("No appropriate filter found for minihack environment")

        def _create_env():
            return TransformedEnv(
                GymEnv(**gym_env_args),
                Compose(
                    MiniHackFilterInputTransform(
                        first_row=first_row,
                        last_row=last_row,
                        first_col=first_col,
                        last_col=last_col,
                    ),
                    FlattenObservation(first_dim=-2, last_dim=-1, in_keys=["chars"]),
                    RenameTransform(in_keys=["chars"], out_keys=["inputs"]),
                    StepCounter(max_steps=self.config.max_episode_steps),
                    InitTracker(),
                ),
            )

        return _create_env
