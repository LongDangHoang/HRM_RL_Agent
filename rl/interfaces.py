from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import torch.nn as nn


@dataclass
class GymRLDatasetConfig:
    # details on the environment interface
    vocab_size: int
    seq_len: int
    action_space_size: int
    num_puzzle_identifiers: int = 1
    env_name: str = "CartPole-v1"
    env_kwargs: dict = field(default_factory=dict)
    max_episode_steps: int = 200
    do_not_skip_running_model_if_random_action: bool = False  # by default, to save time, if eps-greedy takes a random action, then the policy won't be run

    buffer_capacity: int = 1_000  # number of frames the buffer can hold
    frames_per_update: int = (
        320  # how many observations to return every time the collector is iterated over
    )
    data_collection_batch_size: int = (
        64  # how many observations to sample from buffer for training
    )
    training_batch_size: int = 64
    updates_per_data: int = 64  # how many batches sampled from buffer before sampling a new batch from collector
    num_workers: int = 4  # how many workers to fetch from buffer

    # device for putting data collection
    env_device: str = "cpu"
    policy_device: str = "cuda"
    storing_device: str = "cpu"


@dataclass
class HRMArchExcludeDataConfig:
    # HRM layers config
    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"
    train_halt_head: bool = False

    # hyperparams
    max_memory_size: int = 1  # determines how many observations to remember, multiply with dataset seq_len to get max value for positional embedding. Ideally should be in arch but avoiding modifying the arc
    puzzle_emb_ndim: int = 0  # number of dimensions to embed the puzzle onto, 0 by default which means the model is unaware of the puzzle and so there must only be one type of environment during training and testing
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


@dataclass
class HRMQNetTrainingConfig:
    """
    Configuration for the HRM agent training.
    """

    # HRM optimization parameters
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    max_training_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # architectural parameters
    arch_exclude_data_dependent: HRMArchExcludeDataConfig

    # dataset (environment) parameters
    dataset: GymRLDatasetConfig

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float = 1e-4

    # Q-Learning parameters
    discount_factor: float = 0.99
    start_eps: float = 1
    end_eps: float = 0.1
    eps_decay_steps: int = 1000000  # number of steps to linearly decay epsilon over
    use_target_network: bool = False
    target_network_decay_factor: float = 0.99

    # whether to only reset z_H and z_L on episode reset
    use_last_hidden_state_to_seed_next_environment_step: bool = True

    # instrumentation - logging
    project_name: str = "HRM Agent Deep Q Network"
    resume_from_run: str | None = (
        None  # if set, will resume training weights. Note that dataset is not resumable.
    )
    log_wandb: bool = True

    # evaluation - timing
    model_checkpoint_every_minutes: float = 45
    evaluate_every_minutes: float = 45

    @property
    def arch_config_dict(self):
        return {
            **asdict(self.arch_exclude_data_dependent),
            "batch_size": self.dataset.training_batch_size,
            "seq_len": self.dataset.seq_len,
            "vocab_size": self.dataset.vocab_size,
            "num_puzzle_identifiers": self.dataset.num_puzzle_identifiers,
        }


# utilities for managing eval mode correctly
@contextmanager
def evaluating_net_context(net: nn.Module):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


@contextmanager
def training_net_context(net: nn.Module):
    """Temporarily switch to training mode"""
    iseval = not net.training
    try:
        net.train()
        yield net
    finally:
        if iseval:
            net.eval()
