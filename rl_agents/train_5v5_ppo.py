#!/usr/bin/env python3
"""Distributed PPO trainer for 5v5 in Google Research Football."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import gym
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import gfootball.env as football_env
from rl_agents.networks import FootballPolicy


def init_distributed() -> Tuple[int, int, torch.device]:
    """Initializes torch.distributed if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_shape, action_shape):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, *action_shape), dtype=np.int64)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.pos = 0

    def add(self, obs, actions, logprobs, rewards, dones, values):
        self.obs[self.pos] = obs
        self.actions[self.pos] = actions
        self.logprobs[self.pos] = logprobs
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values
        self.pos += 1

    def compute_returns(self, last_value, last_done, gamma, gae_lambda):
        gae = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int):
        total_steps = self.num_envs * self.num_steps
        indices = np.arange(total_steps)
        np.random.shuffle(indices)
        for batch_indices in BatchSampler(SubsetRandomSampler(indices), batch_size, drop_last=True):
            batch_obs = self.obs.reshape(total_steps, *self.obs.shape[2:])[batch_indices]
            batch_actions = self.actions.reshape(total_steps, *self.actions.shape[2:])[batch_indices]
            batch_logprobs = self.logprobs.reshape(total_steps)[batch_indices]
            batch_advantages = self.advantages.reshape(total_steps)[batch_indices]
            batch_returns = self.returns.reshape(total_steps)[batch_indices]
            batch_values = self.values.reshape(total_steps)[batch_indices]
            yield (
                batch_obs,
                batch_actions,
                batch_logprobs,
                batch_advantages,
                batch_returns,
                batch_values,
            )


def make_env_factory(
    env_name: str,
    representation: str,
    rewards: str,
    action_set: str,
    stacked: bool,
    seed: int,
    players_left: int,
    players_right: int,
    logdir: Optional[str],
    difficulty: float,
    render: bool = False,
) -> Callable[[], gym.Env]:
    def _init():
        env = football_env.create_environment(
            env_name=env_name,
            stacked=stacked,
            representation=representation,
            rewards=rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=render,
            dump_frequency=0,
            logdir=logdir or "",
            number_of_left_players_agent_controls=players_left,
            number_of_right_players_agent_controls=players_right,
            other_config_options={
                "right_team_difficulty": difficulty,
                "left_team_difficulty": difficulty,
                "action_set": action_set,
            },
        )
        env.seed(seed)
        return env

    return _init


def evaluate(policy: FootballPolicy, device: torch.device, args, global_step: int) -> float:
    env = football_env.create_environment(
        env_name=args.env_name,
        stacked=args.stacked,
        representation=args.representation,
        rewards=args.rewards,
        render=False,
        dump_frequency=0,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        number_of_left_players_agent_controls=args.controlled_left,
        number_of_right_players_agent_controls=args.controlled_right,
        other_config_options={
            "right_team_difficulty": args.eval_difficulty,
            "left_team_difficulty": args.eval_difficulty,
            "action_set": args.action_set,
        },
    )
    episodic_returns = []
    obs = env.reset()
    done = False
    with torch.no_grad():
        while len(episodic_returns) < args.eval_episodes:
            obs_tensor = torch.as_tensor(obs, device=device)
            action, _, _, _ = policy.act(obs_tensor)
            if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                act = action.cpu().numpy().astype(np.int64)
            else:
                act = action.cpu().numpy()
            obs, reward, done, _info = env.step(act)
            if done:
                episodic_returns.append(float(np.array(reward).sum()))
                obs = env.reset()
                done = False
    env.close()
    mean_return = float(np.mean(episodic_returns))
    print(f"[eval] step={global_step} episodes={len(episodic_returns)} mean_return={mean_return:.8f}")
    return mean_return


def save_checkpoint(policy, optimizer, step: int, path: Path, extra: Optional[Dict] = None):
    payload = {
        "step": step,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "extra": extra or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class EpisodeMetricsCollector:
    """Collects per-episode metrics emitted by the environment."""

    def __init__(self):
        self._episodes = []

    def _maybe_add(self, info: Optional[dict]):
        if not info or "episode_metrics" not in info:
            return
        self._episodes.append(info["episode_metrics"])

    def add_info(self, info):
        if info is None:
            return
        if isinstance(info, dict):
            if "final_info" in info:
                final_infos = info.get("final_info", [])
                if isinstance(final_infos, dict):
                    self._maybe_add(final_infos)
                else:
                    for item in final_infos:
                        self._maybe_add(item)
                return
            self._maybe_add(info)
            return
        if isinstance(info, (list, tuple)):
            for item in info:
                self._maybe_add(item)

    def pop(self):
        episodes, self._episodes = self._episodes, []
        return episodes


def main():
    parser = argparse.ArgumentParser(description="PPO for GFootball 5v5")
    parser.add_argument("--env-name", default="5_vs_5")
    parser.add_argument("--representation", default="extracted", choices=["extracted", "pixels", "pixels_gray", "simple115", "simple115v2", "raw"])
    parser.add_argument("--rewards", default="scoring,checkpoints,shaping")
    parser.add_argument("--action-set", default="default", choices=["default", "v2", "full"], help="Define o conjunto de acoes (use 'full' para habilitar switch manual de jogador).")
    parser.add_argument("--stacked", dest="stacked", action="store_true", help="Stack 4 frames.")
    parser.add_argument("--no-stacked", dest="stacked", action="store_false", help="Disable frame stacking.")
    parser.set_defaults(stacked=True)
    parser.add_argument("--total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--gamma", type=float, default=0.993)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--controlled-left", type=int, default=1)
    parser.add_argument("--controlled-right", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-difficulty", type=float, default=0.05)
    parser.add_argument("--opponent-difficulty-start", type=float, default=0.01)
    parser.add_argument("--opponent-difficulty-end", type=float, default=0.3)
    parser.add_argument("--curriculum-warmup-updates", type=int, default=300, help="Quantidade de updates antes de iniciar o incremento de dificuldade.")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable mixed precision.")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="policy")
    parser.add_argument("--logdir", type=str, default="runs/ppo_5v5")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--curriculum", dest="curriculum", action="store_true", help="Enable difficulty ramp-up.")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.set_defaults(curriculum=True)
    args = parser.parse_args()

    if args.representation not in {"extracted", "pixels", "pixels_gray"}:
        args.stacked = False

    rank, world_size, device = init_distributed()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    set_seed(args.seed + rank)
    if is_main_process(rank):
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(args.logdir).mkdir(parents=True, exist_ok=True)

    current_difficulty = args.opponent_difficulty_start
    # Avaliacao inicial com dificuldade inicial para sinal mais claro.
    args.eval_difficulty = min(args.eval_difficulty, current_difficulty)

    def build_vec_env(difficulty: float, seed_offset: int = 0):
        env_fns = []
        for idx in range(args.num_envs):
            env_seed = args.seed + seed_offset + rank * args.num_envs + idx
            env_fns.append(
                make_env_factory(
                    env_name=args.env_name,
                    representation=args.representation,
                    rewards=args.rewards,
                    action_set=args.action_set,
                    stacked=args.stacked,
                    seed=env_seed,
                    players_left=args.controlled_left,
                    players_right=args.controlled_right,
                    logdir=args.logdir if is_main_process(rank) else None,
                    difficulty=difficulty,
                    render=args.render and idx == 0 and is_main_process(rank),
                )
            )
        return gym.vector.AsyncVectorEnv(env_fns)

    envs = build_vec_env(current_difficulty)
    single_observation_space = envs.single_observation_space
    single_action_space = envs.single_action_space
    policy = FootballPolicy(single_observation_space, single_action_space).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        policy.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if is_main_process(rank):
            print(f"Resumed from {args.resume}")

    if world_size > 1:
        policy = torch.nn.parallel.DistributedDataParallel(policy, device_ids=None if device.type == "cpu" else [device.index])

    obs = envs.reset()
    global_step = 0
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs * world_size)
    best_eval = -np.inf
    metrics_path = Path(args.logdir) / "train_metrics.jsonl"
    if is_main_process(rank):
        with metrics_path.open("a"):
            pass
    metrics_collector = EpisodeMetricsCollector()

    for update in range(1, num_updates + 1):
        if args.curriculum:
            if update > args.curriculum_warmup_updates:
                progress = min(global_step / float(args.total_timesteps), 1.0)
                target_difficulty = args.opponent_difficulty_start + progress * (args.opponent_difficulty_end - args.opponent_difficulty_start)
                # Atualiza quando a diferença for relevante para evitar rebuild excessivo.
                if target_difficulty - current_difficulty >= 0.02:
                    current_difficulty = target_difficulty
                    envs.close()
                    envs = build_vec_env(current_difficulty, seed_offset=int(global_step))
                    obs = envs.reset()
                    if is_main_process(rank):
                        print(f"[curriculum] step={global_step} difficulty={current_difficulty:.8f}")

        rollout = RolloutBuffer(args.num_steps, args.num_envs, single_observation_space.shape, single_action_space.shape if hasattr(single_action_space, "shape") else (1,))
        for step in range(args.num_steps):
            obs_tensor = torch.as_tensor(obs, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
                action, logprob, _, value = policy.act(obs_tensor)
            actions_np = action.cpu().numpy()
            next_obs, reward, done, info = envs.step(actions_np)
            if is_main_process(rank):
                metrics_collector.add_info(info)

            rollout.add(obs, actions_np, logprob.cpu().numpy(), reward, done, value.cpu().numpy())
            obs = next_obs
            global_step += args.num_envs * world_size

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
            next_value = policy.act(torch.as_tensor(obs, device=device))[3]

        rollout.compute_returns(next_value.cpu().numpy(), done, args.gamma, args.gae_lambda)

        adv_flat = rollout.advantages.reshape(-1)
        adv_mean = float(np.mean(adv_flat))
        adv_std = float(np.std(adv_flat))
        adv_tensor = torch.as_tensor(rollout.advantages, device=device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        rollout.advantages = adv_tensor.cpu().numpy()

        approx_kl = 0.0
        clip_frac = 0.0
        entropy_acc = 0.0
        value_loss_acc = 0.0
        policy_loss_acc = 0.0
        grad_norm_acc = 0.0
        total_minibatches = 0
        for _ in range(args.ppo_epochs):
            for (
                batch_obs,
                batch_actions,
                batch_logprob,
                batch_adv,
                batch_returns,
                _batch_values,
            ) in rollout.get_minibatches(args.minibatch_size):
                optimizer.zero_grad()
                mb_obs = torch.as_tensor(batch_obs, device=device)
                mb_actions = torch.as_tensor(batch_actions, device=device)
                mb_old_logprob = torch.as_tensor(batch_logprob, device=device)
                mb_adv = torch.as_tensor(batch_adv, device=device)
                mb_returns = torch.as_tensor(batch_returns, device=device)

                with torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
                    new_logprob, entropy, value = policy.evaluate(mb_obs, mb_actions)
                    log_ratio = new_logprob - mb_old_logprob
                    ratio = log_ratio.exp()
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value_loss = 0.5 * (mb_returns - value).pow(2).mean()
                    entropy_mean = entropy.mean()
                    loss = pg_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    approx_kl += (mb_old_logprob - new_logprob).mean().item()
                    clip_frac += torch.mean((torch.abs(ratio - 1.0) > args.clip_range).float()).item()
                    entropy_acc += entropy_mean.item()
                    value_loss_acc += value_loss.item()
                    policy_loss_acc += pg_loss.item()
                    grad_norm_acc += float(grad_norm if isinstance(grad_norm, float) else grad_norm.item())
                    total_minibatches += 1

        approx_kl /= (args.ppo_epochs * (args.num_steps * args.num_envs / args.minibatch_size))
        clip_frac /= (args.ppo_epochs * (args.num_steps * args.num_envs / args.minibatch_size))
        total_minibatches = max(total_minibatches, 1)

        if is_main_process(rank):
            metrics = {
                "step": global_step,
                "update": update,
                "kl": approx_kl,
                "kl_divergence": approx_kl,
                "clip_frac": clip_frac,
                "clip_fraction": clip_frac,
                "entropy": float(entropy_acc / total_minibatches),
                "value_loss": float(value_loss_acc / total_minibatches),
                "policy_loss": float(policy_loss_acc / total_minibatches),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "advantage_mean": adv_mean,
                "advantage_std": adv_std,
                "grad_norm": float(grad_norm_acc / total_minibatches),
                "difficulty": current_difficulty,
                "reward_mean": float(np.mean(rollout.rewards)),
                "reward_std": float(np.std(rollout.rewards)),
            }
            episode_summaries = metrics_collector.pop()
            if episode_summaries:
                returns = np.array([m.get("episode_return", 0.0) for m in episode_summaries], dtype=np.float32)
                success = np.array([m.get("success", 0) for m in episode_summaries], dtype=np.float32)
                wins = np.array([m.get("win", 0) for m in episode_summaries], dtype=np.float32)
                goals = np.array([m.get("goals", 0) for m in episode_summaries], dtype=np.float32)
                shots = np.array([m.get("shots_on_target", 0) for m in episode_summaries], dtype=np.float32)
                possession = np.array([m.get("possession_fraction", 0.0) for m in episode_summaries], dtype=np.float32)
                ball_progress = np.array([m.get("ball_progress_avg", 0.0) for m in episode_summaries], dtype=np.float32)
                metrics.update(
                    {
                        "episodic_return_mean": float(returns.mean()),
                        "episodic_return_std": float(returns.std()),
                        "success_rate": float(success.mean()),
                        "goals_per_episode": float(goals.mean()),
                        "shots_on_target_per_episode": float(shots.mean()),
                        "ball_possession_time_mean": float(possession.mean()),
                        "ball_distance_from_goal_mean": float(ball_progress.mean()),
                        "win_rate": float(wins.mean()),
                        "episodes_in_batch": int(len(episode_summaries)),
                    }
                )
            else:
                metrics.update(
                    {
                        "episodic_return_mean": None,
                        "episodic_return_std": None,
                        "success_rate": None,
                        "goals_per_episode": None,
                        "shots_on_target_per_episode": None,
                        "ball_possession_time_mean": None,
                        "ball_distance_from_goal_mean": None,
                        "win_rate": None,
                        "episodes_in_batch": 0,
                    }
                )
            with metrics_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            if update % args.save_interval == 0:
                save_checkpoint(policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy, optimizer, global_step, Path(args.checkpoint_dir) / "last.pt", {"difficulty": current_difficulty})

        if is_main_process(rank) and update % args.eval_interval == 0:
            eval_score = evaluate(policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy, device, args, global_step)
            if eval_score > best_eval:
                best_eval = eval_score
                save_checkpoint(
                    policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy,
                    optimizer,
                    global_step,
                    Path(args.checkpoint_dir) / "best_model.pt",
                    {"eval_score": eval_score, "difficulty": current_difficulty},
                )

    envs.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
