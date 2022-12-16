## find best metric
import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300


def get_best_metric(lines: List[str]) -> dict:
    pattern = r"\[(?P<time>.*)\] ppocr INFO: best metric, hmean: (?P<hmean>.+), precision: (?P<precision>.+), recall: (?P<recall>.+), fps: (?P<fps>.+), best_epoch: (?P<best_epoch>.+)"
    results = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            results.append(match.groupdict())
    return results[-1]


def parse_train_log(lines: List[str]) -> pd.DataFrame:
    pattern = r"\[(?P<time>.*)\] ppocr INFO: epoch: \[(?P<epoch>\d+)/(?P<epoch_total>\d+)\], global_step: (?P<global_step>\d+), lr: (?P<lr>.+), loss: (?P<loss>.+), avg_reader_cost: (?P<avg_reader_cost>.+) s, avg_batch_cost: (?P<avg_batch_cost>.+) s, avg_samples: (?P<avg_samples>.+), ips: (?P<ips>.+) samples/s, eta: (?P<eta>.*)"
    results = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            results.append(match.groupdict())

    df = pd.DataFrame(results)
    df["time"] = df["time"].astype("datetime64")
    df["epoch"] = df["epoch"].astype(int)
    df["epoch_total"] = df["epoch_total"].astype(int)
    df["global_step"] = df["global_step"].astype(int)
    df["lr"] = df["lr"].astype(float)
    df["loss"] = df["loss"].astype(float)
    df["avg_reader_cost"] = df["avg_reader_cost"].astype(float)
    df["avg_batch_cost"] = df["avg_batch_cost"].astype(float)
    df["avg_samples"] = df["avg_samples"].astype(float)
    df["ips"] = df["ips"].astype(float)

    return df


def parse_eval_log(lines: List[str], eval_step=300) -> pd.DataFrame:
    pattern = r"\[(?P<time>.*)\] ppocr INFO: cur metric, precision: (?P<precision>.+), recall: (?P<recall>.+), hmean: (?P<hmean>.+), fps: (?P<fps>.+)"
    results = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            results.append(match.groupdict())

    df = pd.DataFrame(results)
    df['global_step'] = np.array(range(1, len(df) + 1)) * eval_step
    df["time"] = df["time"].astype("datetime64")
    df["precision"] = df["precision"].astype(float)
    df["recall"] = df["recall"].astype(float)
    df["hmean"] = df["hmean"].astype(float)
    df["fps"] = df["fps"].astype(float)

    return df


def parse_args():
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file",
                        type=str,
                        required=True,
                        help="log file path")
    parser.add_argument("--save_dir",
                        type=str,
                        default=None,
                        help="save dir path")
    parser.add_argument("--eval_step", type=int, default=300, help="eval step")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    log_file = args.log_file
    save_dir = args.save_dir
    eval_step = args.eval_step

    if not os.path.exists(log_file):
        raise Exception("log file not exists")

    save_dir = Path(log_file).parent if save_dir is None else save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_figure_dir = os.path.join(save_dir, "figrue")
    if not os.path.exists(save_figure_dir):
        os.makedirs(save_figure_dir)

    with open(log_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    best_metric = get_best_metric(lines)
    print("best metric: {}".format(best_metric))

    train_log = parse_train_log(lines)
    train_log.to_csv(os.path.join(save_dir, "train_log.csv"), index=False)
    eval_log = parse_eval_log(lines, eval_step=eval_step)
    eval_log.to_csv(os.path.join(save_dir, "eval_log.csv"), index=False)

    # plot train loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_log["global_step"], train_log["loss"], label="loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_figure_dir, "loss_train.png"), dpi=300)

    # plot train cost
    plt.figure(figsize=(10, 5))
    plt.plot(train_log["global_step"],
             train_log["avg_reader_cost"],
             label="avg_reader_cost")
    plt.plot(train_log["global_step"],
             train_log["avg_batch_cost"],
             label="avg_batch_cost")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig(os.path.join(save_figure_dir, "cost_train.png"), dpi=300)

    # plot ips
    plt.figure(figsize=(10, 5))
    plt.plot(train_log["global_step"], train_log["ips"], label="ips")
    plt.xlabel("Iteration")
    plt.ylabel("ips")
    plt.savefig(os.path.join(save_figure_dir, "ips_train.png"), dpi=300)

    # plot metric
    plt.figure(figsize=(10, 5))
    plt.plot(eval_log["global_step"], eval_log["precision"], label="precision")
    plt.plot(eval_log["global_step"], eval_log["recall"], label="recall")
    plt.plot(eval_log["global_step"], eval_log["hmean"], label="hmean")
    plt.xticks(np.arange(0, max(eval_log["global_step"]), eval_step * 2))
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(save_figure_dir, "metric_eval.png"), dpi=300)

    with open(os.path.join(save_dir, "best_metric.csv"), "w") as f:
        best_metric = pd.DataFrame(best_metric, index=[0])
        best_metric.to_csv(f, index=False)