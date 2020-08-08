import itertools
from pathlib import Path
from typing import Iterable, List, Any, Tuple, Optional
import sqlite3
from tqdm import tqdm
import charmonium.time_block as ch_time_block

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set(style="whitegrid")

output_path = Path("../output")
output_path.mkdir(exist_ok=True)

def read_illixr_table(table_name: str, index_cols: List[str]) -> pd.DataFrame:
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
    switchboard_topic_stop     = read_illixr_table("switchboard_topic_stop"    , ["topic_name"])
    switchboard_check_qs_start = read_illixr_table("switchboard_check_queues_start", ["serial_no"])
    switchboard_check_qs_stop  = read_illixr_table("switchboard_check_queues_stop" , ["serial_no"])
    timewarp_gpu               = read_illixr_table("timewarp_gpu"              , ["iteration_no"])
    timewarp_fence_start       = read_illixr_table("timewarp_fence_start"      , ["iteration_no"])
    timewarp_fence_stop        = read_illixr_table("timewarp_fence_stop"       , ["iteration_no"])


def compute_durations(start: pd.DataFrame, stop: pd.DataFrame, account_name: Optional[str], suffix: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.merge(
        start,
        stop,
        left_index=True,
        right_index=True,
        suffixes=("_start", "_stop")
    )
    for clock in ["cpu", "wall"]:
        # convert from ns to ms
        start_of_time = min(
            ts[f"{clock}_time_start"].min(),
            ts[f"{clock}_time_stop" ].min(),
        )
        ts[f"{clock}_time_start"] = (ts[f"{clock}_time_start"] - start_of_time) / 1e6
        ts[f"{clock}_time_stop" ] = (ts[f"{clock}_time_stop" ] - start_of_time) / 1e6

        ts[f"{clock}_duration" ] = ts[f"{clock}_time_stop" ] - ts[f"{clock}_time_start" ]

    # compute periods
    ts["period"] = ts["wall_time_start"].diff()

    aggs = dict(
        cpu_duration =["mean", "sum", "std"],
        wall_duration=["mean", "sum", "std"],
        period=["mean", "std"],
    )
 
    summary = ts.groupby("plugin_id").agg(aggs)

    # flatten column from multiindex (["cpu_duration", "wall_duration", "period"], ["mean", "std", "sum"])
    # to single-level index ("cpu_duration_mean", "cpu_duration_std", "cpu_duration_sum", "wall_duration_mean", ...)
    summary.columns = ["_".join(column) for column in summary.columns]

    summary["count"] = ts.groupby("plugin_id")["cpu_duration"].count()

    for df in [ts, summary]:
        if not account_name and suffix:
            assert suffix
            df["account_name"] = pd.merge(
                plugin_start[["plugin_name"]],
                df,
                left_index=True,
                right_index=True,
            )["plugin_name"].map(str) + f" {suffix}"
        elif account_name and not suffix:
            df["account_name"] = account_name
        else:
            raise ValueError("Provide account_name or suffix, but not both")

        df.set_index(["account_name"], inplace=True)

    return (ts, summary)

with ch_time_block.ctx("Compute joins", print_start=False):
    for df in [switchboard_check_qs_start, switchboard_check_qs_stop]:
        df["plugin_id"] = -1
        df.reset_index(inplace=True)
        df.set_index(["plugin_id", "serial_no"], inplace=True)

    for df in [timewarp_fence_start, timewarp_fence_stop]:
        df["plugin_id"] = -1
        df.reset_index(inplace=True)
        df.set_index(["plugin_id", "iteration_no"], inplace=True)

    threadloop_iteration_ts, threadloop_iteration_summary = compute_durations(threadloop_iteration_start, threadloop_iteration_stop, None, "iter")
    threadloop_skip_ts     , threadloop_skip_summary      = compute_durations(threadloop_skip_start     , threadloop_skip_stop     , None, "skip")
    switchboard_callback_ts, switchboard_callback_summary = compute_durations(switchboard_callback_start, switchboard_callback_stop, None, "cb")
    timewarp_fence_ts      , timewarp_fence_summary       = compute_durations(timewarp_fence_start      , timewarp_fence_stop      , "timewarp_gl fence", None)
    switchboard_check_qs_ts, switchboard_check_qs_summary = compute_durations(switchboard_check_qs_start, switchboard_check_qs_stop, "runtime check_qs", None)

    all_summaries = pd.concat([
        threadloop_iteration_summary,
        threadloop_skip_summary,
        switchboard_callback_summary,
        # timewarp_gpu doesn't count in CPU time
        # timewarp_fence is already counted in threadloop_iteration_summary
        switchboard_check_qs_summary,
    ])

    all_ts = pd.concat([
        threadloop_iteration_ts,
        threadloop_skip_ts,
        switchboard_callback_ts,
        switchboard_check_qs_ts,
    ]).sort_index()


    with (output_path / "account_summaries.md").open("w") as f:
        f.write(all_summaries.to_markdown())
        f.write("\n\nAll times are in milliseconds unless otherwise mentioned.")

    with (output_path / "timewarp_fence.md").open("w") as f:
        f.write(timewarp_fence_summary.to_markdown())
        f.write("\n\nAll times are in milliseconds unless otherwise mentioned.")

with ch_time_block.ctx("Plot bar chart of CPU time share", print_start=False):
    total_cpu_time = all_ts["cpu_duration"].sum()

    # this works, but is too slow
    # counts = pd.merge(all_ts.reset_index(), all_summaries, left_on=["account_name"], right_index=True, how="left")["count"]
    # all_ts["cpu_duration_share"] = all_ts["cpu_duration"] / total_cpu_time * 100
    # #* counts
    # sns.barplot(
    #     x="account_name",
    #     y="cpu_duration_share",
    #     ci=0.99,
    #     estimator=sum,
    #     data=all_ts.reset_index(),
    # )
    # sns.violinplot(
    #     x="account_name",
    #     y="cpu_duration_share",
    #     inner="stick",
    #     data=all_ts.reset_index(),
    # )

    all_summaries.sort_values("cpu_duration_sum", inplace=True)
    all_summaries["cpu_duration_share"] = all_summaries["cpu_duration_sum"] / total_cpu_time * 100
    # I derived this from the Central Limit Theorem
    # I believe sigma_sum^2 = sigma_individ^2 / sqrt(n).
    all_summaries["cpu_duration_std"] = all_summaries["cpu_duration_std"] / all_summaries["count"]**(1/4) / total_cpu_time * all_summaries["count"]
    fig = plt.figure()
    ax = fig.gca()
    ax.bar(
        x=all_summaries.index,
        height=all_summaries["cpu_duration_share"],
        yerr=all_summaries["cpu_duration_std"],
    )
    ax.set_xticklabels(all_summaries.index, rotation=90)
    ax.set_ylabel("% of total CPU time")
    ax.set_title("Breakdown of total CPU time")

    fig.tight_layout()
    fig.savefig(output_path / "cpu_time_per_account.png")
    plt.close(fig)

with ch_time_block.ctx("Plot timeseries charts", print_start=False):
    fig = plt.figure()
    ax = fig.gca()
    for account in set(all_ts.index):
        xs = all_ts.loc[account]["wall_time_start"]
        ys = all_ts.loc[account]["cpu_duration"]
        ax.plot((xs - xs[0]) / 1e3, ys / ys.mean(), label=account)
    ax.legend()
    ax.set_title("Normalized CPU-time duration")
    ax.set_ylabel("CPU-time duration (multiples of account mean)")
    ax.set_xlabel("time since start (s)")
    ax.set_ylim(0, 20)
    fig.savefig(output_path / "account_durations_normalized_timeseries.png")
    plt.close(fig)

def nice_histogram(ys: np.array, label: str, title: str, account: str, path: Path, bins: int):
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(ys, bins=bins, align='mid')
    ax.plot(ys, np.random.randn(*ys.shape) * (ax.get_ylim()[1] * 0.2) + (ax.get_ylim()[1] * 0.5) * np.ones(ys.shape), linestyle='', marker='.', ms=1)
    ax.set_title(title)
    ax.set_xlabel(label)
    fig.savefig(path / f"{account}.png")
    plt.close(fig)

with ch_time_block.ctx("Plot histograms", print_start=False):
    duration_hist_path = output_path / "duration_hists"
    period_hist_path = output_path / "period_hists"

    for path in [duration_hist_path, period_hist_path]:
        path.mkdir(exist_ok=True)

    for account in tqdm(set(all_ts.index)):
        nice_histogram(
            all_ts.loc[account]["cpu_duration"],
            "CPU-time (ms)",
            f"Swarm/histogram of CPU-time duration of {account}",
            account,
            duration_hist_path,
            60,
        )
        nice_histogram(
            np.diff(all_ts.loc[account]["wall_time_start"].values),
            "Period (ms)",
            f"Swarm/histogram of periods of {account}",
            account,
            period_hist_path,
            60,
        )
