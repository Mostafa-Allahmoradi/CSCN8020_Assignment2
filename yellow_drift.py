"""
yellow_drift.py (Assignment 2 Edition with GUI & Headless mode)

Tabular Q-Learning simulator for Yellow Drift.
Integrates with 'yellow_drift_util.py' for environment description and simulation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import gymnasium as gym

# --- IMPORT ASSIGNMENT UTILS ---
try:
    import yellow_drift_util as utils
except ImportError:
    print("Error: yellow_drift_util.py not found. Please ensure it is in the same directory.")
    raise

# ----------------------------
# Utilities
# ----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def moving_average(values: List[float], window: int) -> float:
    if window <= 1:
        return float(values[-1]) if values else 0.0
    if len(values) < window:
        return float(np.mean(values)) if values else 0.0
    return float(np.mean(values[-window:]))

# ----------------------------
# Environment abstraction
# ----------------------------

class DiscreteEnv:
    @property
    def n_states(self) -> int: raise NotImplementedError
    @property
    def n_actions(self) -> int: raise NotImplementedError
    def reset(self, seed: Optional[int] = None) -> int: raise NotImplementedError
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]: raise NotImplementedError
    def render(self) -> None: pass

class GymnasiumWrapper(DiscreteEnv):
    def __init__(self, gym_env: Any, render: str = "none") -> None:
        self.env = gym_env
        self._render = render
        try:
            self._n_states = int(self.env.observation_space.n)
            self._n_actions = int(self.env.action_space.n)
        except Exception as e:
            raise ValueError("Requires discrete observation_space.n and action_space.n") from e

    @property
    def n_states(self) -> int: return self._n_states
    @property
    def n_actions(self) -> int: return self._n_actions

    def reset(self, seed: Optional[int] = None) -> int:
        out = self.env.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out
        return int(obs)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        out = self.env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out
            done = bool(done)
        return int(obs), float(reward), bool(done), dict(info) if info is not None else {}

    def render(self) -> None:
        if self._render != "none":
            try: self.env.render()
            except Exception: pass

# ----------------------------
# Q-Learning agent
# ----------------------------

@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay: float = 0.999

    def value(self, episode_idx: int) -> float:
        eps = self.end + (self.start - self.end) * (self.decay ** episode_idx)
        return float(max(self.end, min(self.start, eps)))

@dataclass
class QLearningConfig:
    gamma: float = 0.99
    alpha: float = 0.8 
    epsilon: float = 0.1
    epsilon_schedule: Optional[EpsilonSchedule] = None
    max_steps_per_episode: int = 200
    seed: int = 42

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, cfg: QLearningConfig) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.cfg = cfg
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        set_global_seed(cfg.seed)

    def get_epsilon(self, episode_idx: int) -> float:
        return self.cfg.epsilon if self.cfg.epsilon_schedule is None else self.cfg.epsilon_schedule.value(episode_idx)

    def act(self, state: int, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self.Q[state]
        m = float(np.max(q))
        candidates = np.flatnonzero(q == m)
        return int(random.choice(candidates))

    def select_action(self, state: int) -> int:
        return self.act(state, epsilon=0.0)

    def update(self, s: int, a: int, r: float, sp: int, done: bool) -> float:
        q_sa = self.Q[s, a]
        target = r if done else (r + self.cfg.gamma * float(np.max(self.Q[sp])))
        td_error = target - q_sa
        self.Q[s, a] = q_sa + self.cfg.alpha * td_error
        return float(td_error)

    def save(self, path: str) -> None:
        np.savez_compressed(path, Q=self.Q, cfg=dataclasses.asdict(self.cfg))

    @staticmethod
    def load(path: str) -> "QLearningAgent":
        data = np.load(path, allow_pickle=True)
        Q = data["Q"]
        cfg_dict = data["cfg"].item() if isinstance(data["cfg"], np.ndarray) else data["cfg"]
        eps_sched = None
        if cfg_dict.get("epsilon_schedule") is not None:
            eps_sched = EpsilonSchedule(**cfg_dict["epsilon_schedule"])
        cfg = QLearningConfig(
            gamma=float(cfg_dict["gamma"]),
            alpha=float(cfg_dict["alpha"]),
            epsilon=float(cfg_dict["epsilon"]),
            epsilon_schedule=eps_sched,
            max_steps_per_episode=int(cfg_dict["max_steps_per_episode"]),
            seed=int(cfg_dict["seed"]),
        )
        agent = QLearningAgent(Q.shape[0], Q.shape[1], cfg)
        agent.Q = Q.astype(np.float64)
        return agent

# ----------------------------
# Simulator
# ----------------------------

@dataclass
class EpisodeStats:
    episode: int
    steps: int
    total_reward: float
    epsilon: float
    td_error_mean_abs: float
    success: bool

class QLearningSimulator:
    def __init__(self, env: DiscreteEnv, agent: QLearningAgent) -> None:
        self.env = env
        self.agent = agent
        self.stats: List[EpisodeStats] = []

    def train(
        self,
        episodes: int,
        render: bool = False,
        log_every: int = 50,
        seed: Optional[int] = None,
        stop_on_goal: bool = False,
        stop_avg_reward: Optional[float] = None,
        ma_window: int = 100,
        train_render_sleep: float = 0.0,
        log_file: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None 
    ) -> List[EpisodeStats]:
        if seed is not None:
            set_global_seed(seed)
        self.stats.clear()

        rewards_hist: List[float] = []

        for ep in range(int(episodes)):
            eps = float(self.agent.get_epsilon(ep))
            s = int(self.env.reset(seed=None))
            total_reward = 0.0
            td_abs: List[float] = []
            success = False

            for t in range(self.agent.cfg.max_steps_per_episode):
                if render:
                    self.env.render()
                    if train_render_sleep > 0:
                        time.sleep(train_render_sleep)

                a = self.agent.act(s, eps)
                sp, r, done, info = self.env.step(a)

                td = self.agent.update(s, a, r, sp, done)
                td_abs.append(abs(td))

                total_reward += r
                s = sp

                if done:
                    term = str(info.get("terminal", ""))
                    success = (r > 0) or (term == "goal")
                    steps = t + 1
                    break
            else:
                steps = self.agent.cfg.max_steps_per_episode

            rewards_hist.append(float(total_reward))

            self.stats.append(EpisodeStats(
                episode=ep,
                steps=int(steps),
                total_reward=float(total_reward),
                epsilon=float(eps),
                td_error_mean_abs=float(np.mean(td_abs) if td_abs else 0.0),
                success=bool(success),
            ))

            if log_every > 0 and (ep + 1) % log_every == 0:
                recent = self.stats[-log_every:]
                avg_r = float(np.mean([x.total_reward for x in recent]))
                avg_len = float(np.mean([x.steps for x in recent]))
                succ_rate = float(np.mean([1.0 if x.success else 0.0 for x in recent]))
                
                log_msg = f"[episode {ep+1:>6}] avg_reward={avg_r:+.3f} avg_len={avg_len:.1f} success_rate={succ_rate:.2%} eps={eps:.3f}"
                print(log_msg, flush=True)
                
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(log_msg + "\n")
                
                if log_callback:
                    log_callback(log_msg)

            if stop_on_goal and success:
                msg = f"Early stop: first success at episode {ep+1} (reward={total_reward:+.3f})."
                print(msg, flush=True)
                if log_callback:
                    log_callback(msg)
                break

            if stop_avg_reward is not None:
                ma = moving_average(rewards_hist, ma_window)
                if ma >= float(stop_avg_reward):
                    msg = f"Early stop: moving-average reward {ma:.3f} >= {float(stop_avg_reward):.3f} (window={ma_window}) at episode {ep+1}."
                    print(msg, flush=True)
                    if log_callback:
                        log_callback(msg)
                    break

        return self.stats

    def evaluate(
        self,
        episodes: int = 100,
        greedy: bool = True,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        if seed is not None:
            set_global_seed(seed)

        rewards: List[float] = []
        lengths: List[int] = []
        successes: List[float] = []

        for _ in range(int(episodes)):
            s = int(self.env.reset(seed=None))
            total = 0.0
            success = False

            for t in range(self.agent.cfg.max_steps_per_episode):
                if greedy:
                    a = self.agent.select_action(s) 
                else:
                    a = self.agent.act(s, self.agent.cfg.epsilon)

                s, r, done, info = self.env.step(a)
                total += r
                if done:
                    term = str(info.get("terminal", ""))
                    success = (r > 0) or (term == "goal")
                    lengths.append(t + 1)
                    break
            else:
                lengths.append(self.agent.cfg.max_steps_per_episode)

            rewards.append(float(total))
            successes.append(1.0 if success else 0.0)

        return {
            "episodes": float(episodes),
            "avg_reward": float(np.mean(rewards) if rewards else 0.0),
            "std_reward": float(np.std(rewards) if rewards else 0.0),
            "avg_length": float(np.mean(lengths) if lengths else 0.0),
            "success_rate": float(np.mean(successes) if successes else 0.0),
        }

# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabular Q-Learning Simulator.")
    p.add_argument("--gym-id", default="Taxi-v3")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon-decay", action="store_true")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.999)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--stop-on-goal", action="store_true")
    p.add_argument("--stop-avg-reward", type=float, default=None)
    p.add_argument("--ma-window", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--save", type=str, default="")
    p.add_argument("--load", type=str, default="")
    p.add_argument("--export-stats", type=str, default="")
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--no-gui", action="store_true", help="Run training silently without tkinter pop-ups or visual simulation.")
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)
    
    # Only initialize tkinter if GUI is allowed
    if not args.no_gui:
        import tkinter as tk
        from tkinter import messagebox
        from tkinter import scrolledtext
        root = tk.Tk()
        root.withdraw()

    # 2. Setup Environment
    raw_env = gym.make(args.gym_id) 
    env = GymnasiumWrapper(raw_env)

    print("\n--- Environment Description (from assignment2_utils) ---")
    if not hasattr(raw_env, "reward_range"):
        raw_env.reward_range = (-float('inf'), float('inf'))
    utils.describe_env(raw_env)
    print("--------------------------------------------------------\n")

    # 3. Load or initialize agent
    if args.load:
        agent = QLearningAgent.load(args.load)
        agent.cfg.epsilon = args.epsilon 
    else:
        eps_sched = EpsilonSchedule(start=args.eps_start, end=args.eps_end, decay=args.eps_decay) if args.epsilon_decay else None
        cfg = QLearningConfig(
            gamma=args.gamma, alpha=args.alpha, epsilon=args.epsilon,
            epsilon_schedule=eps_sched, max_steps_per_episode=args.max_steps, seed=args.seed
        )
        agent = QLearningAgent(env.n_states, env.n_actions, cfg)

    sim = QLearningSimulator(env, agent)

    # 4. TRAINING
    if args.episodes > 0:
        if not args.no_gui:
            # GUI Training Mode
            start_training = messagebox.askyesno(
                "Training Required", 
                "Our taxi driver doesn't know how to drive yet. Let's train the driver before we hit the road.\nFair enough?"
            )
            
            if not start_training:
                print("Training cancelled by user. Exiting.")
                root.destroy()
                return 0
                
            log_win = tk.Toplevel()
            log_win.title("Training Status")
            log_win.geometry("550x350")
            
            tk.Label(log_win, text="Training Driver in Progress... Please Wait", font=("Arial", 12, "bold")).pack(pady=10)
            text_area = scrolledtext.ScrolledText(log_win, wrap=tk.WORD, width=65, height=15)
            text_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
            
            def update_gui_log(msg: str):
                text_area.insert(tk.END, msg + "\n")
                text_area.see(tk.END)
                log_win.update()

            sim.train(
                episodes=args.episodes, render=False, log_every=args.log_every,
                seed=args.seed, stop_on_goal=args.stop_on_goal, 
                stop_avg_reward=args.stop_avg_reward, ma_window=args.ma_window,
                log_file=args.log_file if args.log_file else None,
                log_callback=update_gui_log
            )
            
        else:
            # Silent / Headless Training Mode
            sim.train(
                episodes=args.episodes, render=False, log_every=args.log_every,
                seed=args.seed, stop_on_goal=args.stop_on_goal, 
                stop_avg_reward=args.stop_avg_reward, ma_window=args.ma_window,
                log_file=args.log_file if args.log_file else None
            )

    # 5. Evaluate
    metrics = sim.evaluate(episodes=args.eval_episodes, greedy=True, seed=args.seed)
    print("\nEvaluation (greedy):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, flush=True)

    # 6. VISUAL SIMULATION LOOP
    if not args.no_gui:
        print("\n--- Visual Simulation using assignment2_utils ---")
        visual_env = gym.make(args.gym_id, render_mode="human")
        
        try:
            while True:
                utils.simulate_episodes(visual_env, agent, num_episodes=1)
                keep_playing = messagebox.askyesno("Continue Playing?", "Do you want to play another round?")
                if not keep_playing:
                    print("Simulation ended by user.")
                    break
        except AttributeError as e:
            print(f"Error during simulation: {e}")
            
        visual_env.close()

    # Clean up and export
    if not args.no_gui:
        root.destroy()
    
    if args.export_stats:
        payload = {"config": vars(args), "train_stats": [dataclasses.asdict(s) for s in sim.stats], "eval": metrics}
        os.makedirs(os.path.dirname(args.export_stats) or ".", exist_ok=True)
        with open(args.export_stats, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        agent.save(args.save)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())