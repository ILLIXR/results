import collections
import itertools
from pathlib import Path
from typing import Iterable, List, Any, Tuple, Optional, Callable, TypeVar, Dict, Union
import sqlite3
import copy
import re
import shutil
import warnings

from tqdm import tqdm
import charmonium.time_block as ch_time_block
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

verify_integrity = True

def read_illixr_table(metrics_path: Path, table_name: str, index_cols: List[str]) -> pd.DataFrame:
    db_path = metrics_path / (table_name + ".sqlite")
    assert db_path.exists()
    return (
        pd.read_sql_query(f"SELECT * FROM {table_name};", sqlite3.connect(str(db_path)))
        .sort_values(index_cols)
        .set_index(index_cols, verify_integrity=verify_integrity)
        .sort_index()
    )

def read_illixr_csv(metrics_path: Path, record_name: str, index_cols: List[str], other_cols: List[str]) -> pd.DataFrame:
    cols = index_cols + other_cols
    pattern = re.compile(
        r"^{record_name},"
        + r",".join(fr"(?P<{col}>[^,]+)" for col in cols)
        + r"\s+$"
    )

    df_as_dict = {col: [] for col in cols}

    output_log = metrics_path / "output.log"
    if not output_log.exists():
        raise ValueError(f"{metrics_path!s}/ouptut.log does not exist. Pipe the output into it.")

    with output_log.open("r") as f:
        for line in f:
            if match := pattern.match(line):
                for col in cols:
                    df_as_dict[col].append(match.group(col))

    return (
        pd.DataFrame.from_dict(df_as_dict)
        .sort_values(index_cols)
        .set_index(index_cols, verify_integrity=verify_integrity)
        .sort_index()
    )

list_concat = itertools.chain.from_iterable
T = TypeVar("T")
V = TypeVar("V")
def dict_concat(dcts: Iterable[Dict[T, V]], **kwargs) -> Dict[T, V]:
    return dict(list_concat(dct.items() for dct in dcts))

def set_account_name(
        ts: pd.DataFrame,
        account_name: str = "",
        plugin_start: Optional[pd.DataFrame] = None,
        index: Optional[List[str]] = None,
) -> pd.DataFrame:
    if "account_name" in ts:
        pass
    elif plugin_start is not None:
        ts = ts.assign(
            account_name = pd.merge(
                plugin_start["plugin_name"],
                ts,
                left_index=True,
                right_index=True,
            )["plugin_name"].map(str) + f"{account_name}"
        )
    else:
        ts = ts.assign(
            account_name=account_name
        )
    index = ["account_name"] + (index if index else ["iteration_no"])
    return (
        ts
        .reset_index()
        .sort_values(index)
        .set_index(index, verify_integrity=verify_integrity)
        .sort_index()
    )

def compute_durations(ts: pd.DataFrame, account_name: str = "", plugin_start: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    clocks = ["cpu", "wall", "gpu"]
    starts = {
        clock: min(
            ts[f"{clock}_time_start"].min(),
            ts[f"{clock}_time_stop" ].min(),
        )
        for clock in clocks
        if f"{clock}_time_start" in ts
    }

    return ts.assign(
        **dict_concat(
            {
                f"{clock}_time_start": (ts[f"{clock}_time_start"] - starts[clock]) / 1e6 if clock in starts else 0,
                f"{clock}_time_stop" : (ts[f"{clock}_time_stop" ] - starts[clock]) / 1e6 if clock in starts else 0,
                f"{clock}_time_duration": ts[f"{clock}_time_stop"] - ts[f"{clock}_time_start"] if clock in starts else 0,
            }
            for clock in clocks
        ),
        period=lambda ts: ts["wall_time_start"].diff(),
    )

def split_account(ts: pd.DataFrame, account_name: str, new_account_name: str, bool_mask: pd.Series) -> pd.DataFrame:
    # need to reset index to mutate it
    ts = ts.assign(
        account_name=ts.index.droplevel(1),
        iteration_no=ts.index.droplevel(0),
    )

    ts.loc[
        (ts["account_name"] == account_name) & bool_mask, "account_name"
    ] = new_account_name

    # all done mutating index.
    # Restore it.
    return (
        ts
        .reset_index(drop=True)
        .sort_values(["account_name", "iteration_no"])
        .set_index(["account_name", "iteration_no"], verify_integrity=verify_integrity)
        .sort_index()
    )

def concat_accounts(ts: pd.DataFrame, account_name_a: str, account_name_b: str, account_name_out: str) -> pd.DataFrame:
    # Keep in mind a merge won't work, because there could be many:many
    # For example, concat-ing skip to iterations by iteration_no

    new_rows = (
        (
            pd.concat([
                ts.loc[account_name_a].reset_index(),
                ts.loc[account_name_b].reset_index(),
            ])
            .groupby(["iteration_no"])
            .agg(dict(list_concat(
                {
                    f"{clock}_time_duration": "sum",
                    f"{clock}_time_start": "min",
                    f"{clock}_time_stop" : "max",
                }.items()
                for clock in ["cpu", "gpu", "wall"]
            )))
            .assign(
                account_name=account_name_out,
            )
            .reset_index()
            .sort_values(["account_name", "iteration_no"])
            .set_index(["account_name", "iteration_no"], verify_integrity=verify_integrity)
            .sort_index()
        )
    )

    return pd.concat([
        ts.drop(account_name_a, level=0).drop(account_name_b, level=0),
        new_rows,
    ], verify_integrity=verify_integrity).sort_index()

def get_data(metrics_path: Path) -> None:
    plugin_name            = read_illixr_table(metrics_path, "plugin_name"             , ["plugin_id"])
    threadloop_iteration   = read_illixr_table(metrics_path, "threadloop_iteration"    , ["plugin_id", "iteration_no"])
    threadloop_skip        = read_illixr_table(metrics_path, "threadloop_skip"         , ["plugin_id", "iteration_no", "skip_no"])
    switchboard_callback   = read_illixr_table(metrics_path, "switchboard_callback"    , ["plugin_id", "iteration_no"])
    switchboard_topic_stop = read_illixr_table(metrics_path, "switchboard_topic_stop"  , ["topic_name"])
    switchboard_check_qs   = read_illixr_table(metrics_path, "switchboard_check_queues", ["iteration_no"])
    timewarp_gpu           = read_illixr_table(metrics_path, "timewarp_gpu"            , ["iteration_no"])
    imu_cam                = read_illixr_table(metrics_path, "imu_cam"                 , ["iteration_no"])

    # This is an integer in SQLite, because no bool in SQLite.
    # Convert it to a bool.
    imu_cam["has_camera"]  = imu_cam["has_camera"] == 1

    for row in switchboard_topic_stop.itertuples():
        if row.unprocessed / max(1, row.unprocessed + row.processed) > 0.03:
            warnings.warn(f"""
{row.Index} has many ({row.unprocessed} / {(row.unprocessed + row.processed)}) unprocessed events when ILLIXR terminated. Your hardware might be oversubscribed.
""".strip(), UserWarning)

    # stdout_cpu_timer = read_illixr_csv(metrics_path, "cpu_timer", ["account_name", "iteration_no"], ["wall_time_start", "wall_time_stop", "cpu_time_start", "cpu_time_stop"])
    # stdout_gpu_timer = read_illixr_csv(metrics_path, "gpu_timer", ["account_name", "iteration_no"], ["wall_time_start", "wall_time_stop", "gpu_duration"])

    threadloop_skip = (
        set_account_name(compute_durations(threadloop_skip), " skip", plugin_name, ["iteration_no", "skip_no"])
        .groupby(level=[0, 1])
        .agg(dict(list_concat(
            {
                f"{clock}_time_duration": "sum",
                f"{clock}_time_start": "min",
                f"{clock}_time_stop" : "max",
            }.items()
            for clock in ["cpu", "gpu", "wall"]
        )))
    )
    ts = pd.concat([
        compute_durations(pd.concat([
            set_account_name(threadloop_iteration, " iter", plugin_name),
            set_account_name(switchboard_callback, " cb", plugin_name),
            set_account_name(switchboard_check_qs, "runtime_check qs"),
            # set_account_name(stdout_cpu_timer),
        ])),
        threadloop_skip,
    ]).sort_index()

    for account_name in set(threadloop_skip.index.droplevel(1)):
        account_name = account_name.split(" ")[0]
        ts = concat_accounts(ts, f"{account_name} iter", f"{account_name} skip", account_name)

    bool_mask_cam = ts.join(imu_cam)["has_camera"].fillna(value=False)

    ts = split_account(ts, "offline_imu_cam", "offline_imu_cam cam",  bool_mask_cam)
    ts = split_account(ts, "offline_imu_cam", "offline_imu_cam imu", ~bool_mask_cam)

    ts = split_account(ts, "slam2 cb", "slam2 cb cam",  bool_mask_cam)
    ts = split_account(ts, "slam2 cb", "slam2 cb imu", ~bool_mask_cam)

    ts = concat_accounts(ts, "offline_imu_cam cam", "slam2 cb cam", "camera")
    ts = concat_accounts(ts, "offline_imu_cam imu", "slam2 cb imu", "IMU")
    # ts = concat_accounts(ts, "timewarp_gl iter", "timewarp_gl gpu", "timewarp")

    summaries = ts.groupby("account_name").agg({
        "period"       : ["mean", "std"],
        "cpu_time_duration" : ["mean", "std", "sum"],
        "wall_time_duration": ["mean", "std", "sum"],
        "gpu_time_duration" : ["mean", "std", "sum"],
    })

    # flatten column from multiindex (["cpu_duration", "wall_duration", "period"], ["mean", "std", "sum"])
    # to single-level index ("cpu_duration_mean", "cpu_duration_std", "cpu_duration_sum", "wall_duration_mean", ...)
    summaries.columns = ["_".join(column) for column in summaries.columns]

    summaries["count"] = ts.groupby("account_name")["wall_time_duration"].count()

    return ts, summaries

ts, summaries = get_data(Path("..") / "metrics")

output_path = Path("../output")
output_path.mkdir(exist_ok=True)
with (output_path / "account_summaries.md").open("w") as f:
    f.write(summaries.to_markdown())
    f.write("\n\nAll times are in milliseconds unless otherwise mentioned.")

import sys
sys.exit(0)

with ch_time_block.ctx("Plot bar chart of CPU time share", print_start=False):
    total_cpu_time = ts["cpu_duration"].sum()

    # counts = pd.merge(ts.reset_index(), summaries, left_on=["account_name"], right_index=True, how="left")["count"]
    # ts["cpu_duration_share"] = ts["cpu_duration"] / total_cpu_time * 100
    # #* counts
    # ax = sns.violinplot(
    #     x="account_name",
    #     y="cpu_duration_share",
    #     inner="stick",
    #     data=ts.reset_index(),
    # )
    # ax.get_figure().savefig(output_path / "cpu_time_violins.png")
    # plt.close(ax.get_figure())

    summaries.sort_values("cpu_duration_sum", inplace=True)
    summaries["cpu_duration_share"] = summaries["cpu_duration_sum"] / total_cpu_time
    # I derived this from the Central Limit Theorem
    # sigma_sum^2 = sigma_individ^2 / sqrt(n).
    # sigma_sum = sigma_individ / n^(1/4)
    # 
    summaries["cpu_duration_std"] = summaries["cpu_duration_std"] * summaries["count"]**(3/4) / total_cpu_time

    summaries2_share = collections.defaultdict(lambda: 0)
    summaries2_std = collections.defaultdict(lambda: 0)
    account_group_names = []
    for account_name in summaries.index:
        account_group_name = account_name.split(" ")[0].strip()
        summaries2_share[account_group_name] += summaries.loc[account_name]["cpu_duration_share"]
        summaries2_std[account_group_name] = np.sqrt(summaries.loc[account_name]["cpu_duration_std"]**2 + summaries2_std[account_group_name]**2)
        account_group_names.append(account_group_name)

    fig = plt.figure()
    ax = fig.gca()
    ax.bar(
        x=account_group_names,
        height=[100*summaries2_share[account_group_name] for account_group_name in account_group_names],
        yerr  =[100*summaries2_std  [account_group_name] for account_group_name in account_group_names],
    )
    ax.set_ylim(0, 100)
    ax.set_xticklabels(account_group_names, rotation=90)
    ax.set_ylabel("% of total CPU time")
    ax.set_title("Breakdown of total CPU time")

    fig.tight_layout()
    fig.savefig(output_path / "cpu_time_per_account.png")
    plt.close(fig)

    accounts = sorted(set(ts.index.levels[0]))

with ch_time_block.ctx("Plot timeseries charts", print_start=False):
    fig = plt.figure()
    ax = fig.gca()
    for account in accounts:
        xs = ts.loc[account]["wall_time_start"]
        ys = ts.loc[account]["cpu_duration"]
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
    fig.savefig(path / f"{account.replace(' ', '-')}.png")
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
            ts.loc[account]["wall_duration"],
            "Wall time (ms)",
            f"Wall time duration of {account}",
            account,
            wall_duration_hist_path,
            60,
        )
        nice_histogram(
            ts.loc[account]["cpu_duration"],
            "CPU time (ms)",
            f"CPU time duration of {account}",
            account,
            cpu_duration_hist_path,
            60,
        )
        nice_histogram(
            ts.loc[account]["cpu_duration"] / ts.loc[account]["wall_duration"] * 100,
            "Utilization (%)",
            f"Utilization of {account}",
            account,
            utilization_hist_path,
            60,
        )
        nice_histogram(
            np.diff(ts.loc[account]["wall_time_start"]),
            "Period (ms)",
            f"Periods of {account}",
            account,
            period_hist_path,
            60,
        )

nice_histogram(
    ts.loc["timewarp_gl gpu"]["gpu_duration"],
    "GPU time (ms)",
    "GPU time of timewarp",
    "timewarp_gl GPU calls",
    gpu_duration_hist_path,
    60,
)

nice_histogram(
    ts.loc["timewarp_gl gpu"]["gpu_duration"] / ts.loc["timewarp_gl iter"]["cpu_duration"],
    "CPU overhead (%)",
    "CPU overhead of timewarp_gl GPU calls",
    "timewarp_gl GPU calls",
    gpu_overhead_hist_path,
    60,
)
