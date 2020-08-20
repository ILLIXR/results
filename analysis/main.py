
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

T = TypeVar("T")
V = TypeVar("V")
def it_concat(its: Iterable[Iterable[T]]) -> Iterable[T]:
    return itertools.chain.from_iterable(its)

def list_concat(lsts: Iterable[List[T]]) -> List[T]:
    return list(itertools.chain.from_iterable(lsts))

def dict_concat(dcts: Iterable[Dict[T, V]], **kwargs) -> Dict[T, V]:
    return dict(it_concat(dct.items() for dct in dcts), **kwargs)

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

def is_int(string: str) -> bool:
    return bool(re.match(r"^\d+$", string))

def is_float(string: str) -> bool:
    return bool(re.match(r"^\d+.\d+(e[+-]\d+)?$", string))

def read_illixr_csv(metrics_path: Path, record_name: str, index_cols: List[str], other_cols: List[str]) -> pd.DataFrame:
    cols = index_cols + other_cols
    pattern = re.compile(
        rf"^{record_name},"
        + r",".join(fr"(?P<{col}>[^,]+)" for col in cols)
        + r"\s+$"
    )

    df_as_dict: Dict[str, List[str]] = {col: [] for col in cols}

    output_log = metrics_path / "output.log"
    if not output_log.exists():
        raise ValueError(f"{metrics_path!s}/ouptut.log does not exist. Pipe the output into it.")

    close_miss = 0
    with output_log.open("r") as f:
        for line in f:
            if match := pattern.match(line):
                for col in cols:
                    df_as_dict[col].append(match.group(col))
            elif line.startswith(record_name + ","):
                close_miss += 1
    hits = len(df_as_dict[cols[0]])
    if hits + close_miss != 0 and hits / (hits + close_miss) < 0.95:
        warnings.warn(UserWarning(f"Many ({close_miss} / {close_miss + hits}) lines of {record_name} were spliced incorrectly."))

    for key, col in df_as_dict.items():
        if all(map(is_int, col)):
            col = list(map(int, col))
        elif all(map(is_float, col)):
            col = np.array(col, dtype=float)
            assert not np.any(np.isnan(col))

        df_as_dict[key] = col
            
    df = pd.DataFrame.from_dict(df_as_dict)

    if index_cols:
        df = (
            df
            .sort_values(index_cols)
            .set_index(index_cols, verify_integrity=verify_integrity)
            .sort_index()
        )
    return df

def set_account_name(
        ts: pd.DataFrame,
        account_name: str = "",
        plugin_start: Optional[pd.DataFrame] = None,
        index: Optional[List[str]] = None,
) -> pd.DataFrame:
    if "account_name" in list(ts.index.names) + list(ts.columns):
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
    ts = (
        ts
        .reset_index()
        .sort_values(index)
        .set_index(index, verify_integrity=verify_integrity)
        .sort_index()
    )
    if "plugin_id" in ts.columns:
        ts = ts.drop(columns=["plugin_id"])
    return ts

clocks = ["cpu", "wall", "gpu"]

def compute_durations(
        ts: pd.DataFrame,
        account_name: str = "",
        plugin_start: Optional[pd.DataFrame] = None,
        fix_periods: bool = True,
) -> pd.DataFrame:
    # for clock in clocks:
    #     for series in [f"{clock}_time_start", f"{clock}_time_stop"]:
    #         if series in ts:
    #             nans = pd.to_numeric(ts[series], errors='coerce').isna()
    #             if nans.sum():
    #                 print("nans", series)
    #                 print(ts.dtypes)
    #                 print(nans.index[nans])
    #                 print(ts[series])
    #                 raise ValueError()
    ts = ts.assign(**dict_concat(
        {
            f"{clock}_time_start": pd.to_numeric(ts[f"{clock}_time_start"]),
            f"{clock}_time_stop" : pd.to_numeric(ts[f"{clock}_time_stop" ]),
        }
        for clock in clocks
        if f"{clock}_time_start" in ts
    ))

    starts = {
        clock: min(
            ts[f"{clock}_time_start"].min(),
            ts[f"{clock}_time_stop" ].min(),
        )
        for clock in clocks
        if f"{clock}_time_start" in ts
    }

    ts = ts.assign(**dict_concat(
        {
            f"{clock}_time_start": (ts[f"{clock}_time_start"] - starts[clock]) / 1e6 if clock in starts else 0,
            f"{clock}_time_stop" : (ts[f"{clock}_time_stop" ] - starts[clock]) / 1e6 if clock in starts else 0,
        }
        for clock in clocks
    ))
    ts = ts.assign(**dict_concat(
            {
                f"{clock}_time_duration": ts[f"{clock}_time_stop"] - ts[f"{clock}_time_start"] if clock in starts else 0,
            }
            for clock in clocks
        ),
        period=ts["wall_time_start"].diff(),
    )
    if fix_periods:
        for account_name in ts.index.levels[0]:
            mask = ts.index.droplevel(1) == account_name
            ts.loc[mask, "period"] = ts.loc[mask, "wall_time_start"].diff()
    return ts

def split_account(
        ts: pd.DataFrame,
        account_name: str,
        new_account_name: str,
        bool_mask: pd.Series,
) -> pd.DataFrame:
    # need to reset index to mutate it
    ts = ts.assign(
        account_name=ts.index.droplevel(1),
        iteration_no=ts.index.droplevel(0),
    )

    index_mask = ts["account_name"] == account_name
    assert (not verify_integrity) or index_mask.sum()
    assert (not verify_integrity) or bool_mask.sum()
    assert (not verify_integrity) or (index_mask & bool_mask).sum()

    with ch_time_block.ctx("change_index", print_start=False):
        ts["account_name"].mask(
            cond=index_mask & bool_mask,
            other=new_account_name,
            inplace=True,
        )

    # all done mutating index.
    # Restore it.
    return (
        ts
        .reset_index(drop=True)
        .sort_values(["account_name", "iteration_no"])
        .set_index(["account_name", "iteration_no"], verify_integrity=verify_integrity)
        .sort_index()
    )

def concat_accounts(
        ts: pd.DataFrame,
        account_name_a: str,
        account_name_b: str,
        account_name_out: str,
) -> pd.DataFrame:
    new_rows = (
        pd.merge(
            ts.loc[account_name_a],
            ts.loc[account_name_b],
            left_index=True ,
            right_index=True,
            suffixes=("_a", "_b"),
        )
    )
    l_a = len(ts.loc[account_name_a])
    l_b = len(ts.loc[account_name_b])
    l_ab = len(new_rows)
    if l_a < l_b * 0.95 or l_b < l_a * 0.95:
        warnings.warn(UserWarning(f"Merging {account_name_a} ({l_a} rows) with {account_name_b} ({l_b} rows), despite row-count mismatch"))
    if l_ab < l_a * 0.95:
        warnings.warn(UserWarning(f"Merging {account_name_a} ({l_a} rows) with {account_name_b} ({l_b} rows) have only {l_ab} rows in common"))
    new_rows = (
        new_rows
        .assign(
            **dict_concat(
                {
                    f"{clock}_time_start"   : new_rows[[f"{clock}_time_start_a"   , f"{clock}_time_start_b"   ]].min(axis=1),
                    f"{clock}_time_stop"    : new_rows[[f"{clock}_time_stop_a"    , f"{clock}_time_stop_b"    ]].max(axis=1),
                }
                for clock in ["cpu", "gpu", "wall"]
            ),
            account_name=account_name_out,
            period=lambda df: df["wall_time_start"].diff(),
        )
    )
    new_rows = (
        new_rows
        .assign(
            cpu_time_duration=new_rows[f"cpu_time_duration_a"] + new_rows[f"cpu_time_duration_b"],
            wall_time_duration=new_rows[f"wall_time_duration_b"],
            gpu_time_duration=new_rows[f"gpu_time_duration_a"] + new_rows[f"gpu_time_duration_b"],
        )
    )
    new_rows = (
        new_rows
        .drop(columns=[col for col in new_rows.columns if col.endswith("_a") or col.endswith("_b")])
        .sort_values(["account_name", "iteration_no"])
        .reset_index()
        .set_index(["account_name", "iteration_no"], verify_integrity=verify_integrity)
        .sort_index()
    )

    with ch_time_block.ctx("concat", print_start=False):
        return pd.concat([
            ts.drop(account_name_a, level=0).drop(account_name_b, level=0),
            new_rows,
        ], verify_integrity=verify_integrity).sort_index()

def rename_accounts(ts: pd.DataFrame, renames: Dict[str, str]) -> pd.DataFrame:
    # need to reset index to mutate it
    ts = ts.assign(
        account_name=ts.index.droplevel(1),
        iteration_no=ts.index.droplevel(0),
    )

    for old_account_name, new_account_name in renames.items():
        ts.loc[ts["account_name"] == old_account_name, "account_name"] = new_account_name

    return (
        ts
        .reset_index(drop=True)
        .sort_values(["account_name", "iteration_no"])
        .set_index(["account_name", "iteration_no"], verify_integrity=verify_integrity)
        .sort_index()
    )

def sub_accounts(ts: pd.DataFrame, account_a: str, account_b: str) -> pd.DataFrame:
    ts = ts.copy()
    ts.loc[account_a, "cpu_time_duration"] -= ts.loc[account_b, "cpu_time_duration"]
    return ts

def reindex(ts: pd.DataFrame, accounts: Optional[List[str]] = None) -> pd.DataFrame:
    ts = (
        ts
        .reset_index()
    )
    accounts = accounts if accounts is not None else ts["account_name"].unique()
    for account_name in accounts:
        account_mask = ts["account_name"] == account_name
        ts.loc[account_mask, "iteration_no"] = range(account_mask.sum())

    return (
        ts
        .reset_index(drop=True)
        .sort_values(["account_name", "iteration_no"])
        .set_index(["account_name", "iteration_no"], verify_integrity=verify_integrity)
        .sort_index()
    )

@ch_time_block.decor(print_start=False)
def get_data(metrics_path: Path) -> Dict[str, pd.DataFrame]:
    with ch_time_block.ctx("load sqlite", print_start=False):
        plugin_name            = read_illixr_table(metrics_path, "plugin_name"             , ["plugin_id"])
        threadloop_iteration   = read_illixr_table(metrics_path, "threadloop_iteration"    , ["plugin_id", "iteration_no"])
        switchboard_callback   = read_illixr_table(metrics_path, "switchboard_callback"    , ["plugin_id", "iteration_no"])
        switchboard_topic_stop = read_illixr_table(metrics_path, "switchboard_topic_stop"  , ["topic_name"])
        switchboard_check_qs   = read_illixr_table(metrics_path, "switchboard_check_queues", ["iteration_no"])
        timewarp_gpu           = read_illixr_table(metrics_path, "timewarp_gpu"            , ["iteration_no"])
        imu_cam                = read_illixr_table(metrics_path, "imu_cam"                 , ["iteration_no"])
        camera_cvtfmt          = read_illixr_table(metrics_path, "camera_cvtfmt"           , ["iteration_no"])

        # This is an integer in SQLite, because no bool in SQLite.
        # Convert it to a bool.
        imu_cam["has_camera"]  = imu_cam["has_camera"] == 1


    with ch_time_block.ctx("load csv", print_start=False):
        thread_ids = read_illixr_csv(metrics_path, "thread", [], ["thread_id", "name", "sub_name"])
        stdout_cpu_timer = read_illixr_csv(metrics_path, "cpu_timer", ["account_name", "iteration_no"], ["wall_time_start", "wall_time_stop", "cpu_time_start", "cpu_time_stop"])
        stdout_gpu_timer = read_illixr_csv(metrics_path, "gpu_timer", ["account_name", "iteration_no"], ["wall_time_start", "wall_time_stop", "gpu_time_duration"])
        for df in [stdout_gpu_timer, timewarp_gpu]:
            # TODO: not this
            if not df.empty:
                df["gpu_time_start"] = df["gpu_time_duration"].cumsum().shift(1)
                df["gpu_time_stop" ] = df["gpu_time_duration"].cumsum()
                del df["gpu_time_duration"]
                df.drop(index=df.index[0], inplace=True)

        stdout_cpu_timer2 = read_illixr_csv(metrics_path, "cpu_timer2", ["iteration_no", "thread_id", "sub_iteration_no"], ["wall_time_start", "wall_time_stop", "cpu_time_start", "cpu_time_stop"])
        stdout_cpu_timer2 = (
            compute_durations(stdout_cpu_timer2, fix_periods=False)
        )
        stdout_cpu_timer2 = (
            stdout_cpu_timer2
            .reset_index()
            .assign(
                iteration_no=lambda df: df["iteration_no"].mask(df["iteration_no"] == "null", "-1").astype(int)
            )
            .sort_values(["iteration_no", "thread_id", "sub_iteration_no"])
            .set_index(["iteration_no", "thread_id", "sub_iteration_no"], verify_integrity=verify_integrity)
            .sort_index()
        )
        stdout_cpu_timer2 = (stdout_cpu_timer2
            .groupby(level=[0])
            # just groupby iteration_no, coalescing thread_id and sub_iteration_no
            .agg({
                **dict_concat(
                    {
                        f"{clock}_time_duration": sum,
                        f"{clock}_time_start"   : min,
                        f"{clock}_time_stop"    : max,
                    }
                    for clock in clocks
                ),
            })
            .assign(
                account_name="opencv",
                period=lambda df: df["wall_time_start"].diff(),
            )
        )
        stdout_cpu_timer2 = (stdout_cpu_timer2
            .reset_index()
            .sort_values(["account_name", "iteration_no"])
            .set_index(["account_name", "iteration_no"], verify_integrity=verify_integrity)
            .sort_index()
        )

    switchboard_topic_stop = switchboard_topic_stop.assign(
        completion = lambda df: df["processed"] / (df["unprocessed"] + df["processed"]).clip(lower=1)
    )
    for row in switchboard_topic_stop.itertuples():
        if row.completion < 0.95 and row.unprocessed > 0:
            warnings.warn("\n".join([
                f"{row.Index} has many ({row.unprocessed} / {(row.unprocessed + row.processed)}) unprocessed events when ILLIXR terminated.",
                "Your hardware resources might be oversubscribed.",
            ]) , UserWarning)

    with ch_time_block.ctx("concat data", print_start=False):
        ts = compute_durations(pd.concat([
            set_account_name(threadloop_iteration, " iter", plugin_name),
            set_account_name(switchboard_callback, " cb", plugin_name),
            set_account_name(switchboard_check_qs, "runtime check_qs"),
            reindex(set_account_name(stdout_cpu_timer)),
            set_account_name(timewarp_gpu, "timewarp_gl gpu"),
            set_account_name(camera_cvtfmt, "camera_cvtfmt"),
            set_account_name(stdout_gpu_timer, " gpu", None),
        ])).sort_index()
        ts = pd.concat([ts, stdout_cpu_timer2]).sort_index()

    with ch_time_block.ctx("misc", print_start=False):
        account_names = ts.index.levels[0]
        thread_ids["missing_cpu_time_usage"] = np.NaN
        thread_ids[  "total_cpu_time_usage"] = np.NaN

        used = 0
        start = np.inf
        stop = 0
        for account_name in account_names:
            if account_name.endswith(" cb"):
                used += ts.loc[account_name, "cpu_time_duration"].sum()
                start = min(start, ts.loc[account_name, "cpu_time_start"].min())
                stop  = max(stop , ts.loc[account_name, "cpu_time_start"].max())
        used += ts.loc["runtime check_qs", "cpu_time_duration"].sum()
        start = min(start, ts.loc["runtime check_qs", "cpu_time_start"].min())
        stop  = max(stop , ts.loc["runtime check_qs", "cpu_time_stop" ].max())
        thread_ids.loc[thread_ids["sub_name"] == "0", "total_cpu_time_usage"] = stop - start
        thread_ids.loc[thread_ids["sub_name"] == "0", "missing_cpu_time_usage"] = stop - start - used

        for account_name in account_names:
            if account_name.endswith(" iter"):
                thread_name = account_name.split(" ")[0]
                used  = ts.loc[account_name, "cpu_time_duration"].sum()
                start = ts.loc[account_name, "cpu_time_start"   ].min()
                stop  = ts.loc[account_name, "cpu_time_stop"    ].max()
                thread_ids.loc[thread_ids["sub_name"] == thread_name, "total_cpu_time_usage"] = stop - start
                thread_ids.loc[thread_ids["sub_name"] == thread_name, "missing_cpu_time_usage"] = stop - start - used

    with ch_time_block.ctx("split accounts", print_start=False):

        bool_mask_cam = ts.join(imu_cam)["has_camera"].fillna(value=False)

        has_zed = "zed_camera_thread iter" in ts.index.levels[0]
        has_gldemo = "gldemo iter" in ts.index.levels[0]
        if not has_zed:
            ts = split_account(ts, "offline_imu_cam iter", "offline_imu_cam cam",  bool_mask_cam)
            ts = split_account(ts, "offline_imu_cam iter", "offline_imu_cam imu", ~bool_mask_cam)
            assert "offline_imu_cam" not in ts.index.levels[0]

        ts = split_account(ts, "slam2 cb", "slam2 cb cam",  bool_mask_cam)
        ts = split_account(ts, "slam2 cb", "slam2 cb imu", ~bool_mask_cam)
        ts = reindex(ts, ["slam2 cb cam"])

    with ch_time_block.ctx("concat accounts", print_start=False):
        # ts = concat_accounts(ts, "timewarp_gl iter", "timewarp_gl gpu", "Timewarp")
        ts = concat_accounts(ts, "slam2 hist l", "slam2 cb cam", "slam2 cb cam")
        ts = concat_accounts(ts, "slam2 hist r", "slam2 cb cam", "slam2 cb cam")
        ts = concat_accounts(ts, "slam2 matching l", "slam2 cb cam", "slam2 cb cam")
        ts = concat_accounts(ts, "slam2 matching r", "slam2 cb cam", "slam2 cb cam")
        ts = concat_accounts(ts, "slam2 pyramid l", "slam2 cb cam", "slam2 cb cam")
        ts = concat_accounts(ts, "slam2 pyramid r", "slam2 cb cam", "slam2 cb cam")
        # ts = concat_accounts(ts, "opencv", "slam2 cb cam", "slam2 cb cam")

    with ch_time_block.ctx("rename accounts", print_start=False):
        ts = rename_accounts(ts, {
            "runtime check_qs": "Runtime",
            **({
                "gldemo iter": "Application (GL Demo)"
            } if has_gldemo else {}),
            # "camera_cvtfmt": "Camera convert-format",
            "audio_encoding": "Audio Encoding",
            "audio_decoding": "Audio Decoding",
            "slam2 cb imu": "OpenVINS IMU",
            "slam2 cb cam": "OpenVINS Camera",
            **({
                "offline_imu_cam cam": "Camera loading",
                "offline_imu_cam imu": "IMU loading",
            } if not has_zed else {
                "zed_camera_thread": "ZED Camera (part)",
                "zed_imu_thread": "ZED IMU (part)",
            })
        })

    summaries = ts.groupby("account_name").agg({
        "period"            : ["mean", "std"],
        "cpu_time_duration" : ["mean", "std", "sum"],
        "wall_time_duration": ["mean", "std", "sum"],
        "gpu_time_duration" : ["mean", "std", "sum"],
    })

    # flatten column from multiindex (["cpu_duration", "wall_duration", "period"], ["mean", "std", "sum"])
    # to single-level index ("cpu_duration_mean", "cpu_duration_std", "cpu_duration_sum", "wall_duration_mean", ...)
    summaries.columns = ["_".join(column) for column in summaries.columns]

    summaries["count"] = ts.groupby("account_name")["wall_time_duration"].count()

    return ts, summaries, switchboard_topic_stop, thread_ids

ts, summaries, switchboard_topic_stop, thread_ids = get_data(Path("..") / "metrics")

output_path = Path("../output")
output_path.mkdir(exist_ok=True)
with (output_path / "account_summaries.md").open("w") as f:
    f.write("# Summaries\n\n")
    columns = ["count"] + [col for col in summaries.columns if col != "count"]
    floatfmt = ["", ".0f", ".1f", ".1f"] + list_concat(["e", "e", "e"] for clock in clocks)
    f.write(summaries[columns].to_markdown(floatfmt=floatfmt))
    f.write("\n\n")
    f.write("# Thread IDs (of long-running threads)\n\n")
    f.write(thread_ids.sort_values(["name", "sub_name"]).to_markdown())
    f.write("\n\n")
    f.write("# Switchboard topic stops\n\n")
    f.write(switchboard_topic_stop.to_markdown(floatfmt=["", ".0f", ".0f", ".2f"]))
    f.write("\n\n")
    f.write("# Notes\n\n")
    f.write("- All times are in milliseconds unless otherwise mentioned.\n")
    f.write("- Total wall runtime = {:.1f} sec\n".format(
        (ts["wall_time_stop"].max() - ts["wall_time_start"].min()) / 1e3
    ))
    f.write("- Next are the first and last few rows of each account.")
    f.write("\n\n")
    account_names = ts.index.levels[0]
    columns = ["period"] + [col for col in ts.columns if col != "period"]
    for account_name in account_names:
        f.write(f"# {account_name}\n\n")
        df = ts.loc[account_name]
        f.write(pd.concat([df.head(20), df.tail(20)]).to_markdown())
        f.write("\n\n")

    f, (ax, ax2, ax3) = plt.subplots(3, sharex=True)
    f.tight_layout(pad=2.0)
    plt.rcParams.update({'font.size': 8})

    # plot the same data on both axes
    ax.plot(ts.loc["OpenVINS Camera", "cpu_time_stop"], ts.loc["OpenVINS Camera", "cpu_time_duration"], 'tab:orange', linewidth=.5)
    ax.set_title("OpenVINS Camera CPU Time Timeseries")
    ax.set(ylabel='CPU Time (ms)')

    ax2.plot(ts.loc["OpenVINS IMU", "cpu_time_stop"], ts.loc["OpenVINS IMU", "cpu_time_duration"], 'tab:green', linewidth=.5)
    ax2.set_title("OpenVINS IMU CPU Time Timeseries")
    ax2.set(ylabel='CPU Time (ms)')

    if "debugview cb" in account_names:
        ax3.plot(ts.loc["debugview cb", "cpu_time_stop"], ts.loc["debugview cb", "cpu_time_duration"], 'tab:blue', linewidth=.5)
        ax3.set_title("Debugview CB CPU Time Timeseries")
        ax3.set(ylabel='CPU Time (ms)')

    plt.xlabel("Timestamp (ms)")
    plt.savefig("test.png")

    # import IPython; IPython.embed()

    # Add here 
    print(summaries["cpu_time_duration_sum"].to_csv())

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

cpu_duration_hist_path = output_path / "cpu_duration_hists"
period_hist_path = output_path / "period_hists"
wall_duration_hist_path = output_path / "wall_duration_hists"
gpu_duration_hist_path = output_path / "gpu_duration_hists"
gpu_overhead_hist_path = output_path / "gpu_overhead_hists"
utilization_hist_path = output_path / "utilization_hists"

with ch_time_block.ctx("Plot histograms", print_start=False):
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
