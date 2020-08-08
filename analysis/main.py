import itertools
from pathlib import Path
from typing import Iterable, List, Any
import sqlite3
from tqdm import tqdm
import charmonium.time_block as ch_time_block

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

output_path = Path("../output")
output_path.mkdir(exist_ok=True)

def starmap(func, it):
    return map(lambda args: func(*args), it)

def read_illixr_table(table_name: str, index_cols: List[str]):
    db_path = Path("..") / "metrics" / (table_name + ".sqlite")
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    return (
        pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
        .sort_values(index_cols)
        .set_index(index_cols)
    )

with ch_time_block.ctx("Read SQLite tables", print_start=False):
    plugin_start               = read_illixr_table("plugin_start"              , ["plugin_id"])
    threadloop_iteration_start = read_illixr_table("threadloop_iteration_start", ["plugin_id", "iteration_no"])
    threadloop_iteration_stop  = read_illixr_table("threadloop_iteration_stop" , ["plugin_id", "iteration_no"])
    threadloop_skip_start      = read_illixr_table("threadloop_skip_start"     , ["plugin_id", "iteration_no", "skip_no"])
    threadloop_skip_stop       = read_illixr_table("threadloop_skip_stop"      , ["plugin_id", "iteration_no", "skip_no"])
    switchboard_callback_start = read_illixr_table("switchboard_callback_start", ["plugin_id", "serial_no"])
    switchboard_callback_stop  = read_illixr_table("switchboard_callback_stop" , ["plugin_id", "serial_no"])
    #switchboard_topic_stop     = read_illixr_table("switchboard_topic_stop"    , ["topic_name"])

def compute_durations(start: pd.DataFrame, stop: pd.DataFrame):
    ts = pd.merge(
        start,
        stop ,
        left_index =True,
        right_index=True,
        suffixes=("_start", "_stop")
    )
    aggs = dict()
    for clock in ["cpu", "wall"]:
        # convert from ns to ms
        start_of_time = min(
            ts[f"{clock}_time_start"].min(),
            ts[f"{clock}_time_stop" ].min(),
        )
        ts[f"{clock}_time_start"] = (ts[f"{clock}_time_start"] - start_of_time) / 1e6
        ts[f"{clock}_time_stop" ] = (ts[f"{clock}_time_stop" ] - start_of_time) / 1e6

        ts[f"{clock}_duration" ] = ts[f"{clock}_time_stop" ] - ts[f"{clock}_time_start" ]

        aggs[f"{clock}_duration"] = ["mean", "sum", "std"]

    # compute periods
    ts["period"] = ts["wall_time_start"].diff()
    aggs["period"] = ["mean", "std"]

    # Special. I will count the number of cpu_duration entries, to get the count of start/stop entries
    # I could have picked any duration record here.
    aggs["cpu_duration"].append("count")

    summary = ts.groupby("plugin_id").agg(aggs)
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.rename(columns={"cpu_duration_count": "count"})

    return ts, summary

with ch_time_block.ctx("Compute joins", print_start=False):
    threadloop_iteration_ts, threadloop_iteration_summary = compute_durations(threadloop_iteration_start, threadloop_iteration_stop)
    threadloop_skip_ts     , threadloop_skip_summary      = compute_durations(threadloop_skip_start     , threadloop_skip_stop)
    switchboard_callback_ts, switchboard_callback_summary = compute_durations(switchboard_callback_start, switchboard_callback_stop)

    def prepare_for_concat(summary: pd.DataFrame, activity: str):
        summary = pd.merge(plugin_start[["plugin_name"]], summary, left_index=True, right_index=True)
        summary["activity"] = activity
        summary = summary.set_index(["plugin_name", "activity"])
        return summary

    all_summaries = pd.concat(starmap(prepare_for_concat, [
        (threadloop_iteration_summary, "iter"),
        (threadloop_skip_summary     , "skip"),
        (switchboard_callback_summary, "cb"  ),
    ]))

    with (output_path / "account_summaries.md").open("w") as f:
        f.write(all_summaries.to_markdown())
        f.write("\n\nAll times are in milliseconds unless otherwise mentioned.")

with ch_time_block.ctx("Pie", print_start=False):
    labels = [" ".join(levels) for levels in all_summaries.index]
    all_summaries = all_summaries.sort_values("cpu_duration_sum")
    ys = all_summaries["cpu_duration_sum"].values
    fig = plt.figure()
    ax = fig.gca()
    ax.pie(ys, labels=labels)
    fig.savefig(output_path / "cpu_time_per_account_pie.png")
    plt.close(fig)

with ch_time_block.ctx("Timeseries", print_start=False):
    all_ts = pd.concat(starmap(prepare_for_concat, [
        (threadloop_iteration_ts, "iter"),
        (threadloop_skip_ts     , "skip"),
        (switchboard_callback_ts, "cb"  ),
    ])).sort_index()

    fig = plt.figure()
    ax = fig.gca()
    for account in set(all_ts.index):
        xs = all_ts.loc[account]["wall_time_start"]
        ys = all_ts.loc[account]["cpu_duration"]
        ax.plot((xs - xs[0]) / 1e3, ys / ys.mean(), label=' '.join(account))
    ax.legend()
    ax.set_title("CPU-time duration, normalized such that each component's mean = 1")
    ax.set_ylabel("CPU-time duration (multiples of component's mean)")
    ax.set_xlabel("time since start (s)")
    fig.savefig(output_path / "account_durations_normalized_timeseries.png")
    plt.close(fig)

def nice_histogram(ax, ys, bins):
    ax.hist(ys, bins=bins, align='mid')
    ax.plot(ys, np.random.randn(*ys.shape) * (ax.get_ylim()[1] * 0.1) + (ax.get_ylim()[1] * 0.5) * np.ones(ys.shape), linestyle='', marker='.', ms=1)

with ch_time_block.ctx("Histograms", print_start=False):
    duration_hist_path = output_path / "duration_hists"
    duration_hist_path.mkdir(exist_ok=True)

    period_hist_path = output_path / "period_hists"
    period_hist_path.mkdir(exist_ok=True)

    for account in tqdm(set(all_ts.index)):
        fig = plt.figure()
        ax = fig.gca()
        ys = all_ts.loc[account]["cpu_duration"]
        nice_histogram(ax, ys, 60)
        ax.set_title(f"Swarm/histogram of CPU-time duration of {' '.join(account)}")
        ax.set_xlabel("CPU-time duration (ms)")
        fig.savefig(duration_hist_path / ("-".join(account) + ".png"))
        plt.close(fig)

        fig = plt.figure()
        ax = fig.gca()
        ys = np.diff(all_ts.loc[account]["wall_time_start"].values)
        nice_histogram(ax, ys, 60)
        ax.set_title(f"Swarm/histogram of periods of {' '.join(account)}")
        ax.set_xlabel("CPU-time duration (ms)")
        fig.savefig(period_hist_path / ("-".join(account) + ".png"))
        plt.close(fig)
