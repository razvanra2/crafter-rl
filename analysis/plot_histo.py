import argparse
import pathlib

import seaborn as sns
import matplotlib.pyplot as plt
from rich import print

from matplotlib.pyplot import figure

import json

def read_jsonl(path):
    with open(path) as json_file:
        json_list = list(json_file)
    list_result = []
    for json_str in json_list:
        result = json.loads(json_str)
        list_result.append(result)
    return list_result

def read_crafter_logs(indirpath, clip=True):
    indir = pathlib.Path(indirpath)
    filenames = sorted(list(indir.glob("**/*/stats.jsonl")))
    names = []
    results = []
    for idx, fn in enumerate(filenames):
        result = read_jsonl(fn)
        results.append(result)
        names.append(str(fn).split('/')[1].strip())

    avgs = {}
    for run in results:
        values = {}
        for game in run:
            for key, value in game.items():
                if key in values:
                    values[key].append(value)
                else:
                    values[key] = [value]

        for key in values:
            res_key = values[key]
            if key in avgs:
                l1 = res_key
                l2 = avgs[key]

                avgs[key] = [(a + b) for a, b in zip(l1, l2)]
            else:
                avgs[key] = res_key

    keys_list = []
    for key in avgs:
        keys_list.append(key)
        temp = [x / len(results) for x in avgs[key]]
        avgs[key] = temp

    figure(figsize=(80, 60), dpi=80)

    print(avgs)
    fig, axs = plt.subplots(4,6, figsize=(25,15))
    il = [0,1,2,3]
    jl = [0,1,2,3,4,5]
    fig.suptitle('Achievements histograms')

    for i in il:
        for j in jl:
            idx = i*4 + j
            target = keys_list[idx]
            axs[i][j].plot(avgs[target])
            axs[i][j].set_title(target)
    print(f"out_figs/histo_{names[0]}.png")
    plt.savefig(f"out_figs/histo_{names[0]}.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/random_agent",
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    read_crafter_logs(cfg.logdir)
