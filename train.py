import argparse
import pickle
from pathlib import Path

import torch

from src.crafter_wrapper import Env
from src.random_agent import RandomAgent
from src.ddqn_agent import DdqnAgent
from src.models import DQN

RUN_RANDOM = False

def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during trainig.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
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
    train_env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = DdqnAgent(train_env.action_space.n)

    # main loop
    if RUN_RANDOM:
        ep_cnt, step_cnt, done = 0, 0, True
        while step_cnt < opt.steps or not done:
            if done:
                ep_cnt += 1
                obs, done = train_env.reset(), False

            action = agent.act(obs)
            obs, reward, done, info = train_env.step(action)

            step_cnt += 1

            # evaluate once in a while
            if step_cnt % opt.eval_interval == 0:
                eval(agent, eval_env, step_cnt, opt)
    else:
        agent.learn(
            env=train_env,                   # gym environmnet
            Network=DQN,                     # neural network
            eval=eval,                       # the evaluation callback
            eval_env=eval_env,
            opt=opt,
            batch_size=32,                   # q-network update batch size
            gamma=0.9,                      # discount factor
            replay_buffer_size=100000,       # size of the replay buffer
            learning_starts=opt.steps/4,     # number of initial random actions (exploration)
            learning_freq=6,                 # frequency of the update
            target_update_freq=10,           # number of gradient steps after which the target network is updated
            max_allowed_steps=opt.steps,
            log_every=opt.eval_interval      # logging interval. returns the mean reward per episode.
        )

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
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
