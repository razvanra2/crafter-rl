import argparse
import pickle
from pathlib import Path

import torch
import torch.optim as O

from rich import print
from src.crafter_wrapper import Env
from src.util import DuelingNnModel, get_epsilon_schedule, NnModel
from src.memory import ReplayMemory
from src.dqn_agent import DQN
from src.ddqn_agent import DoubleDQN
from src.dueling_agent import DuelingDQN
from src.random_agent import RandomAgent

def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    min_return = episodic_returns.min().item()
    max_return = episodic_returns.max().item()
    ep_str = "{number:06}".format(number=crt_step)
    print(f"[{ep_str}] R/ep={avg_return}, std={episodic_returns.std().item()} eval results: {episodic_returns}.")
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return, "min_return": min_return, "max_return": max_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during trainig.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        obs = obs.reshape(1, obs.size(0), obs.size(1), obs.size(2))
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            obs = obs.reshape(1, obs.size(0), obs.size(1), obs.size(2))
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    
    warmup_steps = opt.steps / 10

    if opt.net == 'dqn':
        print("Using DQN net")
        net = NnModel(env.action_space.n).to(opt.device)
        agent = DQN(
            net,
            ReplayMemory(opt.device, size=1000, batch_size=32),
            O.Adam(net.parameters(), lr=1e-3, eps=1e-4),
            get_epsilon_schedule(start=1.0, end=0.1, steps=opt.steps),
            env.action_space.n,
            warmup_steps=warmup_steps,
            update_steps=1,
        )
    elif opt.net == 'ddqn':
        print("Using Double DQN net")
        net = NnModel(env.action_space.n).to(opt.device)
        agent = DoubleDQN(
            net,
            ReplayMemory(opt.device, size=1000, batch_size=32),
            O.Adam(net.parameters(), lr=1e-3, eps=1e-4),
            get_epsilon_schedule(start=1.0, end=0.1, steps=opt.steps),
            env.action_space.n,
            warmup_steps=warmup_steps,
            update_steps=1,
            update_target_steps=4
        )
    elif opt.net == 'dddqn':
        print("Using Dueling DDQN net")
        net = DuelingNnModel(env.action_space.n).to(opt.device)
        agent = DuelingDQN(
            net,
            ReplayMemory(opt.device, size=1000, batch_size=32),
            O.Adam(net.parameters(), lr=1e-3, eps=1e-4),
            get_epsilon_schedule(start=1.0, end=0.1, steps=opt.steps),
            env.action_space.n,
            warmup_steps=warmup_steps,
            update_steps=1,
            update_target_steps=4
        )
    elif opt.net == 'rand':
        print("Using random agent")
        agent = RandomAgent(env.action_space.n)

    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            state, done = env.reset().clone(), False
            state = state.reshape(1, state.size(0), state.size(1), state.size(2))

        action = agent.step(state)
        state_, reward, done, info = env.step(action)
        state_ = state_.reshape(1, state_.size(0), state_.size(1), state_.size(2))

        agent.learn(state, action, reward, state_, done)

        state = state_.clone()

        step_cnt += 1

        if step_cnt % opt.eval_interval == 0:
            print("[{:06d}] progress={:03.2f}%.".format(step_cnt, 100.0 * step_cnt / opt.steps))

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0 and step_cnt >= warmup_steps:
            eval(agent, eval_env, step_cnt, opt)

def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--net",
        type=str,
        default='dqn',
        metavar="NET",
        help="Type of DQN",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
