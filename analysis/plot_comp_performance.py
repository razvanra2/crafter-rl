import argparse
import pathlib
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print

def read_pkl(path):
    events = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                events.append(pickle.load(openfile))
            except EOFError:
                break
    return events


def read_crafter_logs(indir, clip=True):
    agent_subdirs = [f.path for f in os.scandir(indir) if f.is_dir()]
    print(agent_subdirs)

    for subdirpath in agent_subdirs:
        subdir = pathlib.Path(subdirpath)
        filenames = sorted(list(subdir.glob("**/*/eval_stats.pkl")))

        runs = []
        for idx, fn in enumerate(filenames):
            df = pd.DataFrame(columns=["step", "avg_return"], data=read_pkl(fn))
            df["run"] = idx
            runs.append(df)

        # some runs might not have finished and you might want to clip all of them
        # to the shortest one.
        if clip:
            min_len = min([len(run) for run in runs])
            runs = [run[:min_len] for run in runs]
            print(f"Clipped al runs to {min_len}.")

        df = pd.concat(runs, ignore_index=True)
        sns.lineplot(x="step", y="avg_return", data=df, legend="full", label = subdirpath.split("/",1)[1])
    plt.savefig("out_figs/comparative_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/",
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    read_crafter_logs(cfg.logdir)
