import argparse
import pathlib

import seaborn as sns
import matplotlib.pyplot as plt
from rich import print

import json

def read_jsonl(path):
    with open(path) as json_file:
        print(path)
        json_list = list(json_file)

    achievements = []
    unique_achievements = []
    for json_str in json_list:
        result = json.loads(json_str)
        achievement_cnt = 0
        unique_achievemnt_cnt = 0
        for atr, val in result.items():
            if atr != "length" and atr != "reward" and val != 0:
                achievement_cnt += val
                unique_achievemnt_cnt += 1
                achievements.append(achievement_cnt)
                unique_achievements.append(unique_achievemnt_cnt)

    return achievements, unique_achievements

def read_crafter_logs(indirpath, clip=True):
    indir = pathlib.Path(indirpath)
    filenames = sorted(list(indir.glob("**/*/stats.jsonl")))
    runs_tot = []
    runs_uni = []
    names = []
    for idx, fn in enumerate(filenames):
        tot, uni = read_jsonl(fn)
        runs_tot.append(tot)
        runs_uni.append(uni)
        names.append(str(fn).split('/')[1].strip())

    if clip:
        min_len = min([len(run) for run in runs_tot])
        runs_tot = [run[:min_len] for run in runs_tot]
        print(f"Clipped al runs to {min_len}.")

        min_len = min([len(run) for run in runs_uni])
        runs_uni = [run[:min_len] for run in runs_uni]
        print(f"Clipped al runs to {min_len}.")

    avg_tot_by_agent = {}
    avg_uni_by_agent = {}
    for idx, name in enumerate(names):
        if name in avg_tot_by_agent and name in avg_uni_by_agent:
            l1 = avg_tot_by_agent[name]
            l2 = runs_tot[idx]
            # TODO FIXME
            new_val = [(a + b) for a, b in zip(l1, l2)]
            avg_tot_by_agent[name] = new_val

            l1 = avg_uni_by_agent[name]
            l2 = runs_uni[idx]
            # TODO FIXME
            new_val = [(a + b) for a, b in zip(l1, l2)]
            avg_uni_by_agent[name] = new_val
        else:
            avg_tot_by_agent[name] = runs_tot[idx]
            avg_uni_by_agent[name] = runs_uni[idx]

    name_cnt = {}
    for name in names:
        if name in name_cnt:
            name_cnt[name] = name_cnt[name] + 1
        else:
            name_cnt[name] = 1

    for name in names:
        avg_tot_by_agent[name] =[el / name_cnt[name] for el in avg_tot_by_agent[name]]
        avg_uni_by_agent[name] =[el / name_cnt[name] for el in avg_uni_by_agent[name]]

    # plot
    for idx, name in enumerate(names):
        avg_tot = avg_tot_by_agent[name]
        avg_uni = avg_uni_by_agent[name]
        sns.lineplot(data=avg_tot, label="avg total achievements")
        sns.lineplot(data=avg_uni, label="avg unique achievements")
        plt.savefig(f'out_figs/achievements_{name}_plot.png')
        plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/random_agent",
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    read_crafter_logs(cfg.logdir)
