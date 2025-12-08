#!/usr/bin/env python3
"""Rollout helper to evaluate a trained policy."""

import argparse
from pathlib import Path
from typing import Optional

import gym
import numpy as np
import torch

import gfootball.env as football_env
from rl_agents.networks import FootballPolicy


def load_policy(checkpoint: Path, observation_space: gym.Space, action_space: gym.Space, device: torch.device) -> FootballPolicy:
    policy = FootballPolicy(observation_space, action_space).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state", payload)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description="Run inference for a trained Football agent.")
    parser.add_argument("--checkpoint", type=Path, default=Path("policy/best_model.pt"))
    parser.add_argument("--env-name", default="5_vs_5")
    parser.add_argument("--representation", default="extracted", choices=["extracted", "pixels", "pixels_gray", "simple115", "simple115v2", "raw"])
    parser.add_argument("--rewards", default="scoring,checkpoints,shaping")
    parser.add_argument("--action-set", default="full", choices=["default", "v2", "full"], help="Conjunto de acoes; use 'full' para permitir switch manual.")
    parser.add_argument("--stacked", dest="stacked", action="store_true", help="Use stacked observations.")
    parser.add_argument("--no-stacked", dest="stacked", action="store_false")
    parser.set_defaults(stacked=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--difficulty", type=float, default=0.2)
    parser.add_argument("--left-players", type=int, default=1)
    parser.add_argument("--right-players", type=int, default=0)
    parser.add_argument("--record", action="store_true", help="Dump videos to ./videos.")
    parser.add_argument("--video-format", default="webm", choices=["avi", "webm"], help="Video container for recordings.")
    args = parser.parse_args()

    if args.representation not in {"extracted", "pixels", "pixels_gray"}:
        args.stacked = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    logdir: Optional[str] = "videos" if args.record else ""
    if args.record and logdir:
        Path(logdir).mkdir(parents=True, exist_ok=True)
    env = football_env.create_environment(
        env_name=args.env_name,
        representation=args.representation,
        stacked=args.stacked,
        rewards=args.rewards,
        render=args.render,
        write_video=args.record,
        write_full_episode_dumps=args.record,
        write_goal_dumps=args.record,
        dump_frequency=1 if args.record else 0,
        logdir=logdir,
        number_of_left_players_agent_controls=args.left_players,
        number_of_right_players_agent_controls=args.right_players,
        other_config_options={
            "right_team_difficulty": args.difficulty,
            "left_team_difficulty": args.difficulty,
            "video_format": args.video_format,
            "action_set": args.action_set,
        },
    )

    policy = load_policy(args.checkpoint, env.observation_space, env.action_space, device)

    episode_returns = []
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_tensor = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                action, _, _, _ = policy.act(obs_tensor)
            action_np = action.cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            ep_return += float(np.array(reward).sum())
        episode_returns.append(ep_return)
        print(f"[rollout] episode={episode + 1} return={ep_return:.8f}")

    env.close()
    print(f"[rollout] mean_return={np.mean(episode_returns):.8f} over {args.episodes} episodes")


if __name__ == "__main__":
    main()
