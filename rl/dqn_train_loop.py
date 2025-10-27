import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import re
import logging
import os
import boto3
import copy
import dotenv
import math
import hydra
import torch
import time
import tempfile
import tqdm
import wandb

from adam_atan2 import AdamATan2
from collections import OrderedDict
from dataclasses import asdict, dataclass
from jaxtyping import Int, Float
from omegaconf import OmegaConf, DictConfig, SCMode

from tensordict import TensorDictBase
from torchrl.modules import QValueActor, EGreedyModule
from torch import nn
import torch.distributed as dist

from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from rl.agent import HRMQValueNet
from rl.interfaces import (
    HRMQNetTrainingConfig,
    training_net_context,
    evaluating_net_context,
)
from rl.dataset import GymRLDataset, MiniHackFullObservationSimpleEnvironmentDataset


logger = logging.getLogger(__name__)


def exponential_smoothing_update(
    to_update_net: nn.Module,
    update_with_net: nn.Module,
    decay_factor: float = 0.99,
):
    """Does not hanlde thing like batch norm"""
    update_with_params = OrderedDict(update_with_net.named_parameters())
    to_update_params = OrderedDict(to_update_net.named_parameters())

    # check if both model contains the same set of keys
    assert update_with_params.keys() == to_update_params.keys()

    with torch.no_grad():
        for name, param in update_with_params.items():
            # equivalent to to_update_params[name] = to_update_params * decay_factor + (1 - decay_factor) * params
            to_update_params[name].sub_(
                (1.0 - decay_factor) * (to_update_params[name] - param)
            )

    update_with_buffers = OrderedDict(update_with_net.named_buffers())
    to_update_buffers = OrderedDict(to_update_net.named_buffers())

    # check if both model contains the same set of keys
    assert update_with_buffers.keys() == to_update_buffers.keys()

    for name, buffer in update_with_buffers.items():
        # buffers are copied
        to_update_buffers[name].copy_(buffer)


@dataclass
class HRMAgentTrainingState:
    global_step: int = 0
    evaluation_metrics: list[float] | None = None
    total_seconds_training: float = 0
    last_checkpoint_training_seconds_elapsed: float | None = None
    last_evaluation_training_seconds_elapsed: float | None = None


class HRMAgentTrainingModule:
    def __init__(
        self,
        config: HRMQNetTrainingConfig,
        dataset: GymRLDataset,
    ):
        self.config = config
        self.qvalue_net = HRMQValueNet(self.config)
        self.dataset = dataset
        self.actor = QValueActor(self.qvalue_net, spec=dataset.base_env.action_spec)
        self.egreedy_module = EGreedyModule(
            spec=self.actor.spec,
            annealing_num_steps=self.config.eps_decay_steps,
            eps_init=self.config.start_eps,
            eps_end=self.config.end_eps,
        )

        if self.config.use_target_network:
            self.target_network = HRMQValueNet(config)
            for param in self.target_network.parameters():
                param.requires_grad = False
            for buffer in self.target_network.buffers():
                buffer.requires_grad = False
            self.target_network.eval()

        # Estimator
        self.mse = nn.MSELoss()

        # validate some config
        if self.config.use_last_hidden_state_to_seed_next_environment_step:
            assert (
                self.config.dataset.training_batch_size
                == self.config.dataset.data_collection_batch_size
            )

    def pre_training_setup(
        self, checkpoint_dir: str | None = None, run_name: str | None = None
    ):
        """
        Initialise all the necessary training states, like optimisers, checkpointing, wandb, etc.
        """
        self.state = HRMAgentTrainingState()
        self.optimisers = self.configure_optimizers()

        if self.config.log_wandb:
            wandb.login()
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                id=self.config.resume_from_run if self.config.resume_from_run else None,
                resume="must" if self.config.resume_from_run else None,
                config={**asdict(self.config), "mode": "online"},
            )

            # watch model -- very expensive though, so off by default
            # wandb.watch(self.qvalue_net, log="gradients")
        else:
            self.wandb_run = None

        if self.wandb_run is not None:
            self.run_name = self.wandb_run.id
            if run_name is not None:
                logger.warning(
                    "Passed in argument run_name is ignored in favour of wandb run name"
                )
        else:
            self.run_name = run_name
            if run_name is None:
                logger.warning("Run has no name, checkpointing will not be performed")

        self.checkpoint_dir = checkpoint_dir

        if self.config.resume_from_run:
            logger.warning("Loading from checkpoint does not resume dataset state")
            self.load_from_checkpoint(run_name=self.config.resume_from_run)

        # log number of parameters
        self.log(
            "num_parameters",
            sum(param.numel() for param in self.qvalue_net.parameters()),
            pbar=False,
        )

    def post_training_step_callbacks(self):
        """
        Run all the necessary callbacks after a training step
        """
        self.validation_callback()
        self.checkpointing_callback()

    def validation_callback(self, force: bool = False):
        if (
            force
            or (
                (self.state.last_evaluation_training_seconds_elapsed is None)
                and (
                    (self.state.total_seconds_training // 60)
                    > self.config.evaluate_every_minutes
                )
            )
            or (
                (self.state.last_evaluation_training_seconds_elapsed is not None)
                and (
                    (
                        self.state.total_seconds_training
                        - self.state.last_evaluation_training_seconds_elapsed
                    )
                    // 60
                )
                > self.config.evaluate_every_minutes
            )
        ):
            metric = self.validation()
            self.state.last_evaluation_training_seconds_elapsed = (
                self.state.total_seconds_training
            )
            if self.state.evaluation_metrics is not None:
                self.state.evaluation_metrics.append(metric)
            else:
                self.state.evaluation_metrics = [metric]
            tqdm.tqdm.write(
                f"Evaluation metric at time step {self.state.global_step}: {metric}"
            )

    def checkpointing_callback(self, skip_timing_test: bool = False):
        # checkpointing
        if (
            self.checkpoint_dir is not None
            and self.run_name
            and (
                skip_timing_test
                or (
                    (
                        (self.state.last_checkpoint_training_seconds_elapsed is None)
                        and (
                            (self.state.total_seconds_training // 60)
                            > self.config.model_checkpoint_every_minutes
                        )
                    )
                    or (
                        (
                            self.state.last_checkpoint_training_seconds_elapsed
                            is not None
                        )
                        and (
                            (
                                self.state.total_seconds_training
                                - self.state.last_checkpoint_training_seconds_elapsed
                            )
                            // 60
                        )
                        > self.config.model_checkpoint_every_minutes
                    )
                )
            )
        ):
            self.save_to_checkpoint(self.run_name)
            self.state.last_checkpoint_training_seconds_elapsed = (
                self.state.total_seconds_training
            )

    def load_from_checkpoint(
        self,
        run_name: str,
        ckpt_path_name: str = "last.ckpt",
        restore_config: bool = True,
    ):
        """
        Load ckpt from local or on S3
        """
        if self.checkpoint_dir is None:
            raise ValueError(
                "Cannot load from checkpoint if no checkpointing directory is provided during init!"
            )

        def _load_from_ckpt_path(ckpt_path: str):
            state_dict = torch.load(ckpt_path)
            self.qvalue_net.load_state_dict(dict(state_dict["qvalue_net"]))
            self.egreedy_module.load_state_dict(dict(state_dict["exploration_module"]))
            if self.config.use_target_network and "target_network" in state_dict:
                self.target_network.load_state_dict(dict(state_dict["target_network"]))
            self.optimisers = self.configure_optimizers()

            # restore configuration
            if restore_config:
                current_config = copy.deepcopy(self.config)
                checkpoint_config = OmegaConf.to_container(
                    OmegaConf.merge(
                        OmegaConf.structured(HRMQNetTrainingConfig),
                        state_dict["config"],
                    ),
                    structured_config_mode=SCMode.INSTANTIATE,
                )  # pyright: ignore[reportAssignmentType]
                checkpoint_config: HRMQNetTrainingConfig

                # use current config for some values that are not supposed to be changed when resuming from checkpoint
                self.config = checkpoint_config
                self.config.model_checkpoint_every_minutes = (
                    current_config.model_checkpoint_every_minutes
                )
                self.config.evaluate_every_minutes = (
                    current_config.evaluate_every_minutes
                )
                self.config.log_wandb = current_config.log_wandb
                self.config.resume_from_run = current_config.resume_from_run

            # restore optimisers
            for optim, state in zip(self.optimisers, state_dict["optimizers"]):
                optim.load_state_dict(state)

            # restore internal training state
            self.state = HRMAgentTrainingState(**state_dict["state"])

        if self.checkpoint_dir.startswith("s3"):
            with tempfile.TemporaryDirectory() as ckpt_dir:
                s3_client = boto3.client("s3")
                s3_client.download_file(
                    os.getenv("AWS_S3_BUCKET"),
                    f"{self.config.project_name}/{run_name}/{ckpt_path_name}",
                    Path(ckpt_dir) / ckpt_path_name,
                )
                ckpt_path = str((Path(ckpt_dir) / ckpt_path_name).resolve())
                _load_from_ckpt_path(ckpt_path)
        else:
            _load_from_ckpt_path(
                str((Path(self.checkpoint_dir) / run_name / ckpt_path_name).resolve())
            )

    def save_to_checkpoint(self, run_name: str):
        if self.checkpoint_dir is None:
            return

        state_dict = {
            "qvalue_net": self.qvalue_net.state_dict(),
            "exploration_module": self.egreedy_module.state_dict(),
            "optimizers": [optim.state_dict() for optim in self.optimisers],
            "config": asdict(self.config),
            "state": asdict(self.state),
        }

        if self.config.use_target_network:
            state_dict["target_network"] = self.target_network.state_dict()

        def _save_to_dir(save_dir: str):
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            last_ckpt_path = str((Path(save_dir) / "last.ckpt").resolve())
            torch.save(state_dict, last_ckpt_path)

            # save top-k = 1
            top_k_path = None
            if self.state.evaluation_metrics is not None and (
                self.state.evaluation_metrics[-1] == max(self.state.evaluation_metrics)
            ):
                top_k_path = str(
                    (
                        Path(save_dir)
                        / f"optim_step={self.state.global_step}_eval_metric={self.state.evaluation_metrics[-1]:.3f}.ckpt"
                    ).resolve()
                )
                torch.save(state_dict, top_k_path)

            return last_ckpt_path, top_k_path

        if self.checkpoint_dir.startswith("s3"):
            with tempfile.TemporaryDirectory() as ckpt_dir:
                last_ckpt_path, top_k_path = _save_to_dir(ckpt_dir)
                s3_client = boto3.client("s3")
                s3_client.upload_file(
                    last_ckpt_path,
                    os.getenv("AWS_S3_BUCKET"),
                    f"{self.config.project_name}/{run_name}/last.ckpt",
                )
                if top_k_path:
                    s3_client.upload_file(
                        top_k_path,
                        os.getenv("AWS_S3_BUCKET"),
                        f"{self.config.project_name}/{run_name}/{Path(top_k_path).name}",
                    )
        else:
            _save_to_dir(str(Path(self.checkpoint_dir) / run_name))

    def config_logger(self, pbar: tqdm.std.tqdm | None):
        self.pbar = pbar

    def log(self, metric_name: str, value: torch.Tensor | float, pbar: bool = True):
        """
        Utility method for clean logging
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()

        if self.config.log_wandb:
            wandb.log({metric_name: value}, step=self.state.global_step)

        if (
            hasattr(self, "pbar") and self.pbar is not None and pbar
        ):  # log to local pbar
            current_postfix_str = self.pbar.postfix if self.pbar.postfix else ""

            if metric_name in current_postfix_str:
                new_postfix_str = re.sub(
                    rf"{metric_name}=([\d\.-]+)",
                    f"{metric_name}={value:.3f}",
                    current_postfix_str,
                )
            else:
                new_postfix_str = f"{current_postfix_str + ' ; ' if current_postfix_str else ''}{metric_name}={value:.3f}"
            self.pbar.set_postfix_str(new_postfix_str, refresh=False)

    def training_step(self, batch: TensorDictBase):
        """
        Takes in a batch of data and computes the loss for training.

        The batch is based off TorchRL's interface, and contains the target Q-values for the next states (i.e. max discounted Q-values of actions over action space taken from the next state)

        It should have the keys observation, action, reward, terminated, done, truncated, puzzle_identifiers
        It would be good to be able to type hint these keys, but too much work for now.

        The model HRMAgent performs backpropagation against this target. Note, importantly, the model is using its internal carry, so the new incoming batch may be discarded depending on the carry state.
        """
        start_time = time.time()
        with training_net_context(self.qvalue_net):
            if self.qvalue_net.carry is None or self.qvalue_net.carry.halted.any():
                if (
                    self.config.use_last_hidden_state_to_seed_next_environment_step
                    and "prev_seed_h_init" in batch
                ):
                    batch = batch.clone()  # avoid a
                    batch["next"]["seed_h_init"] = batch[
                        "seed_h_init"
                    ].clone()  # next env state seed is the converged value of the current state
                    batch["next"]["seed_l_init"] = batch["seed_l_init"].clone()
                    batch["seed_h_init"] = batch.pop(
                        "prev_seed_h_init"
                    )  # current state seed is the prev state converged value
                    batch["seed_l_init"] = batch.pop("prev_seed_l_init")

                with torch.no_grad():
                    if not self.config.use_target_network:
                        with evaluating_net_context(self.qvalue_net):
                            new_state_max_q_values = (
                                self.qvalue_net(batch["next"])["action_value"]
                                .max(dim=-1)
                                .values
                            )
                    else:
                        with evaluating_net_context(self.target_network):
                            new_state_max_q_values = (
                                self.target_network(batch["next"])["action_value"]
                                .max(dim=-1)
                                .values
                            )
                    new_targets = torch.where(
                        batch["next"]["terminated"].squeeze(),
                        batch["next"]["reward"].squeeze(),
                        batch["next"]["reward"].squeeze()
                        + new_state_max_q_values * self.config.discount_factor,
                    )
                batch["targets"] = new_targets
                self.log(
                    "train_batch_avg_reward", batch["next"]["reward"].mean(), pbar=False
                )
            else:
                batch = self.qvalue_net.carry.current_data

            qvalues = self.qvalue_net(batch)["action_value"]

            assert self.qvalue_net.carry is not None
            current_data = self.qvalue_net.carry.current_data
            qvalues_selected_action = torch.gather(
                qvalues,
                dim=1,
                index=current_data["action"].argmax(dim=-1, keepdim=True),
            ).squeeze()
            targets = current_data["targets"].to(qvalues_selected_action.dtype)
            loss_value: torch.Tensor = self.mse(qvalues_selected_action, targets)

        loss_value.backward()
        self.log("train_loss", loss_value)

        # Apply optimizer
        lr_this_step = None

        for optim, base_lr in zip(
            self.optimisers, (self.config.puzzle_emb_lr, self.config.lr)
        ):  # pyright: ignore[reportArgumentType]
            lr_this_step = cosine_schedule_with_warmup_lr_lambda(
                current_step=self.state.global_step,
                base_lr=base_lr,
                num_warmup_steps=round(self.config.lr_warmup_steps),
                num_training_steps=self.config.max_training_steps,
                min_ratio=self.config.lr_min_ratio,
            )

            if base_lr == self.config.lr:
                self.log("model_lr", lr_this_step, pbar=False)

            for param_group in optim.param_groups:
                param_group["lr"] = lr_this_step

            optim.step()
            optim.zero_grad()

        # update e-greedy epsilon
        self.egreedy_module.step()
        self.log("exploration_eps", self.egreedy_module.eps.data.item(), pbar=False)  # type: ignore

        # update target network
        if self.config.use_target_network:
            exponential_smoothing_update(
                self.target_network,
                self.qvalue_net,
                decay_factor=self.config.target_network_decay_factor,
            )

        self.state.global_step += 1
        self.state.total_seconds_training += time.time() - start_time
        return loss_value

    def validation(self) -> float:
        """
        Steps through a batch size of environment and log the evaluation metrics as well as report the main metric for checkpointing top-k
        """
        trajectories = self.dataset.validation_rollout()

        is_next_not_done: Int[torch.Tensor, "batch_size number_of_steps 1"] = (
            torch.logical_not(trajectories["next"]["terminated"])
        )
        is_done_at_last: Int[torch.Tensor, "batch_size"] = torch.logical_not(
            is_next_not_done[:, -1, 0]
        )  # whether the last step in the trajectory is done or not
        episode_length: Int[torch.Tensor, "batch_size"] = (
            is_next_not_done.sum(dim=1)[:, 0] + is_done_at_last
        )  # +1 to account for the transition to done if it exists

        within_episode_mask = is_next_not_done.detach().clone()
        within_episode_mask[
            torch.arange(within_episode_mask.shape[0]), episode_length - 1, 0
        ] = True  # keep the last transition of the episode as it transitions to done

        # since some trajectories end earlier than others, it will be truncated/terminated earlier, with the final reward/transition repeated
        # in such case, we want to keep up to the episode length only
        total_reward: Float[torch.Tensor, "batch_size"] = (
            trajectories["next"]["reward"] * within_episode_mask
        ).sum(dim=1)[:, 0]
        last_reward: Float[torch.Tensor, "batch_size"] = trajectories["next"]["reward"][
            :, -1, 0
        ]
        mean_total_reward = total_reward.mean().detach().cpu().item()

        self.log("val_avg_reward", mean_total_reward)
        self.log(
            "val_avg_episode_length",
            episode_length.detach().cpu().to(torch.float).mean(),
            pbar=False,
        )
        self.log(
            "val_frac_envs_terminated_reward_1",
            torch.logical_and((last_reward > 0.95), is_done_at_last)
            .detach()
            .cpu()
            .to(torch.float)
            .mean(),
            pbar=False,
        )
        return mean_total_reward

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                self.qvalue_net.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=self.config.puzzle_emb_weight_decay,
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
            ),
            AdamATan2(
                self.qvalue_net.model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=self.config.weight_decay,
            ),
        ]
        return optimizers


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(
            0.0,
            (1 - min_ratio)
            * 0.5
            * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
    )


@hydra.main(config_path="config", config_name="cfg_dqn.yaml", version_base=None)
def launcher(cfg: DictConfig):
    dotenv.load_dotenv()
    # load config and initialise QValue net as well as iterable dataset
    # requires a base data
    typed_cfg: HRMQNetTrainingConfig = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.structured(HRMQNetTrainingConfig), cfg),
        structured_config_mode=SCMode.INSTANTIATE,
    )  # pyright: ignore[reportAssignmentType]
    dataset = MiniHackFullObservationSimpleEnvironmentDataset(config=typed_cfg.dataset)
    with torch.device(
        "cuda"
    ):  # make sure that the buffers used in HRM are initialised on CUDA for backprop
        hrm_agent_training_module = HRMAgentTrainingModule(typed_cfg, dataset)
    dataset.initialise_policy_and_collector(
        hrm_agent_training_module.actor,
        hrm_agent_training_module.egreedy_module,  # pyright: ignore[reportArgumentType]
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("medium")

    hrm_agent_training_module.pre_training_setup(
        checkpoint_dir="s3",  # this is for local, for s3 run, change to "s3"
        run_name=None,  # this is for local, for s3 run with wandb, leave as None
    )
    pbar = tqdm.tqdm(
        total=typed_cfg.max_training_steps,
        desc="Steps",
        initial=hrm_agent_training_module.state.global_step,
    )
    for current_step, training_batch in enumerate(
        dataset, start=hrm_agent_training_module.state.global_step
    ):
        training_batch = training_batch.to(torch.device("cuda"))
        hrm_agent_training_module.config_logger(pbar=pbar)
        hrm_agent_training_module.training_step(training_batch)
        hrm_agent_training_module.post_training_step_callbacks()
        pbar.update(1)

        if current_step + 1 == typed_cfg.max_training_steps:
            break

    # validate and checkpoint at the end
    hrm_agent_training_module.validation_callback(force=True)
    hrm_agent_training_module.checkpointing_callback(skip_timing_test=True)

    pbar.close()


if __name__ == "__main__":
    launcher()
