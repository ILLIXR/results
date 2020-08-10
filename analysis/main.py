import itertools
from pathlib import Path
from typing import Iterable, List, Any, Tuple, Optional
import sqlite3
import shutil

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
    switchboard_callback_start = read_illixr_table("switchboard_callback_start", ["plugin_id", "iteration_no"])
    switchboard_callback_stop  = read_illixr_table("switchboard_callback_stop" , ["plugin_id", "iteration_no"])
    switchboard_topic_stop     = read_illixr_table("switchboard_topic_stop"    , ["topic_name"])
    switchboard_check_qs_start = read_illixr_table("switchboard_check_queues_start", ["iteration_no"])
    switchboard_check_qs_stop  = read_illixr_table("switchboard_check_queues_stop" , ["iteration_no"])
    timewarp_gpu_start         = read_illixr_table("timewarp_gpu_start"        , ["iteration_no"])
    timewarp_gpu_stop          = read_illixr_table("timewarp_gpu_stop"         , ["iteration_no"])
    offline_imu_cam            = read_illixr_table("offline_imu_cam"           , ["iteration_no"])
    offline_imu_cam["has_camera"] = offline_imu_cam["has_camera"] == 1

def compute_durations(start: pd.DataFrame, stop: pd.DataFrame, account_name: Optional[str], suffix: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.merge(
        start,
        stop,
        left_index=True,
        right_index=True,
        suffixes=("_start", "_stop")
    )
    for clock in ["cpu", "wall", "gpu"]:
        if f"{clock}_time_start" in ts:
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

    if not account_name and suffix:
        assert suffix
        ts["account_name"] = pd.merge(
            plugin_start["plugin_name"],
            ts,
            left_index=True,
            right_index=True,
        )["plugin_name"].map(str) + f" {suffix}"
    elif account_name and not suffix:
        ts["account_name"] = account_name
    else:
        raise ValueError("Provide account_name or suffix, but not both")

    ts = ts.reset_index().set_index(["account_name", "iteration_no"])

    return ts

with ch_time_block.ctx("Compute joins", print_start=False):
    all_ts = pd.concat([
        compute_durations(threadloop_iteration_start, threadloop_iteration_stop, None              , "iter"),
        compute_durations(threadloop_skip_start     , threadloop_skip_stop     , None              , "skip"),
        compute_durations(switchboard_callback_start, switchboard_callback_stop, None              , "cb"),
        compute_durations(switchboard_check_qs_start, switchboard_check_qs_stop, "runtime check_qs", None),
        compute_durations(timewarp_gpu_start        , timewarp_gpu_stop        , "timewarp_gl gpu" , None),
    ]).sort_index()

    # need to reset index to mutate it
    all_ts["account_name"] = all_ts.index.droplevel(1)
    all_ts["iteration_no"] = all_ts.index.droplevel(0)

    bool_mask_cam = all_ts.join(offline_imu_cam)["has_camera"].fillna(value=False)

    offline_imu_cam_mask = all_ts["account_name"] == "offline_imu_cam iter"
    all_ts.loc[offline_imu_cam_mask & bool_mask_cam, "account_name"] = "offline_imu_cam iter cam"
    all_ts.loc[offline_imu_cam_mask &~bool_mask_cam, "account_name"] = "offline_imu_cam iter imu"

    slam2_mask = all_ts["account_name"] == "slam2 cb"
    all_ts.loc[slam2_mask & bool_mask_cam, "account_name"] = "slam2 cb cam"
    all_ts.loc[slam2_mask &~bool_mask_cam, "account_name"] = "slam2 cb imu"

    # all done mutating index.
    # Restore it.
    all_ts = all_ts.reset_index(drop=True).set_index(["account_name", "iteration_no"])
    print(all_ts.head())
    print(all_ts.loc["slam2 cb cam"].head())

    aggs = {
        "period"       : ["mean", "std"],
        "cpu_duration" : ["mean", "std", "sum"],
        "wall_duration": ["mean", "std", "sum"],
        "gpu_duration" : ["mean", "std", "sum"],
    }
    summaries = all_ts.groupby("account_name").agg(aggs)

    # flatten column from multiindex (["cpu_duration", "wall_duration", "period"], ["mean", "std", "sum"])
    # to single-level index ("cpu_duration_mean", "cpu_duration_std", "cpu_duration_sum", "wall_duration_mean", ...)
    summaries.columns = ["_".join(column) for column in summaries.columns]

    summaries["count"] = all_ts.groupby("account_name")["cpu_duration"].count()

    with (output_path / "account_summaries.md").open("w") as f:
        f.write(summaries.to_markdown())
        f.write("\n\nAll times are in milliseconds unless otherwise mentioned.")

with ch_time_block.ctx("Plot bar chart of CPU time share", print_start=False):
    total_cpu_time = all_ts["cpu_duration"].sum()

    # counts = pd.merge(all_ts.reset_index(), summaries, left_on=["account_name"], right_index=True, how="left")["count"]
    # all_ts["cpu_duration_share"] = all_ts["cpu_duration"] / total_cpu_time * 100
    # #* counts
    # ax = sns.violinplot(
    #     x="account_name",
    #     y="cpu_duration_share",
    #     inner="stick",
    #     data=all_ts.reset_index(),
    # )
    # ax.get_figure().savefig(output_path / "cpu_time_violins.png")
    # plt.close(ax.get_figure())

    summaries.sort_values("cpu_duration_sum", inplace=True)
    summaries["cpu_duration_share"] = summaries["cpu_duration_sum"] / total_cpu_time * 100
    # I derived this from the Central Limit Theorem
    # I believe sigma_sum^2 = sigma_individ^2 / sqrt(n).
    summaries["cpu_duration_std"] = summaries["cpu_duration_std"] / summaries["count"]**(1/4) / total_cpu_time * summaries["count"]
    fig = plt.figure()
    ax = fig.gca()
    ax.bar(
        x=summaries.index,
        height=summaries["cpu_duration_share"],
        yerr=summaries["cpu_duration_std"],
    )
    ax.set_xticklabels(summaries.index, rotation=90)
    ax.set_ylabel("% of total CPU time")
    ax.set_title("Breakdown of total CPU time")

    fig.tight_layout()
    fig.savefig(output_path / "cpu_time_per_account.png")
    plt.close(fig)

    accounts = sorted(set(all_ts.index.levels[0]))

with ch_time_block.ctx("Plot timeseries charts", print_start=False):
    fig = plt.figure()
    ax = fig.gca()
    for account in accounts:
        xs = all_ts.loc[account]["wall_time_start"]
        ys = all_ts.loc[account]["cpu_duration"]
        ax.plot((xs - xs.iloc[0]) / 1e3, ys / ys.mean(), label=account)
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
    cpu_duration_hist_path = output_path / "cpu_duration_hists"
    period_hist_path = output_path / "period_hists"
    wall_duration_hist_path = output_path / "wall_duration_hists"
    gpu_duration_hist_path = output_path / "gpu_duration_hists"
    gpu_overhead_hist_path = output_path / "gpu_overhead_hists"
    utilization_hist_path = output_path / "utilization_hists"

    for path in [
            cpu_duration_hist_path,
            wall_duration_hist_path,
            period_hist_path,
            utilization_hist_path,
            gpu_duration_hist_path,
    ]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()

    for account in tqdm(accounts):
        nice_histogram(
            all_ts.loc[account]["wall_duration"],
            "Wall time (ms)",
            f"Wall time duration of {account}",
            account,
            wall_duration_hist_path,
            60,
        )
        nice_histogram(
            all_ts.loc[account]["cpu_duration"],
            "CPU time (ms)",
            f"CPU time duration of {account}",
            account,
            cpu_duration_hist_path,
            60,
        )
        nice_histogram(
            all_ts.loc[account]["cpu_duration"] / all_ts.loc[account]["wall_duration"] * 100,
            "Utilization (%)",
            f"Utilization of {account}",
            account,
            utilization_hist_path,
            60,
        )
        nice_histogram(
            np.diff(all_ts.loc[account]["wall_time_start"]),
            "Period (ms)",
            f"Periods of {account}",
            account,
            period_hist_path,
            60,
        )

nice_histogram(
    all_ts.loc["timewarp_gl gpu"]["gpu_duration"],
    "GPU time (ms)",
    "GPU time of timewarp",
    "timewarp_gl GPU calls",
    gpu_duration_hist_path,
    60,
)

nice_histogram(
    all_ts.loc["timewarp_gl gpu"]["gpu_duration"] / all_ts.loc["timewarp_gl iter"]["cpu_duration"],
    "CPU overhead (%)",
    "CPU overhead of timewarp_gl GPU calls",
    "timewarp_gl GPU calls",
    gpu_overhead_hist_path,
    60,
)
