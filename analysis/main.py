import yaml
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
import charmonium.cache as ch_cache
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
from util import list_concat, it_concat, dict_concat, is_int, is_float, TrialConditions, PerTrialData
verify_integrity = True
from per_trial_analysis import analysis as per_trial_analysis
from inter_trial_analysis import analysis as inter_trial_analysis
import plot_graphs

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
        if f"{clock}_time_start" in ts and clock != "gpu"
    ))

    starts = {
        clock: min(
            ts[f"{clock}_time_start"].min(),
            ts[f"{clock}_time_stop" ].min(),
        )
        for clock in clocks
        if f"{clock}_time_start" in ts and clock != "gpu"
    }

    ts = ts.assign(**dict_concat(
        {
            f"{clock}_time_start": (ts[f"{clock}_time_start"] - starts[clock]) / 1e6 if clock in starts else 0,
            f"{clock}_time_stop" : (ts[f"{clock}_time_stop" ] - starts[clock]) / 1e6 if clock in starts else 0,
        }
        for clock in clocks
    ))
    ts = ts.assign(
        **dict_concat(
            {
                f"{clock}_time_duration": ts[f"{clock}_time_stop"] - ts[f"{clock}_time_start"] if clock in starts else 0,
            }
            for clock in clocks
            if clock != "gpu"
        ),
        period=ts["wall_time_start"].diff(),
        gpu_time_duration=ts["gpu_time_duration"] / 1e6 if "gpu_time_duration" in ts else 0,
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

def read_illixr_power(metrics_path: str):
    if not Path(metrics_path + "/perf-results.txt").exists():
        ## get jetson data
        read_pp_cmd = "cat {}/output.log | grep -A5 'Power' | sed 's/[^0-9.]*//g'".format(metrics_path)
        res = subprocess.run(read_pp_cmd, shell=True, stdout=subprocess.PIPE)
        pp_res = res.stdout.splitlines()
        return [float(val)/1000.0 for val in pp_res]
    else:
        read_perf_cmd = "cat {}/perf-results.txt | grep 'energy-pkg\|energy-ram\|seconds' | sed 's/[^0-9.]*//g'".format(metrics_path)
        res = subprocess.run(read_perf_cmd, shell=True, stdout=subprocess.PIPE)
        perf_res = res.stdout.splitlines()
        cpu_energy = float(perf_res[0])
        ddr_energy = float(perf_res[1])
        perf_time = float(perf_res[2])
        read_nvidia_cmd = 'cat {}/nvidia-smi.txt | grep -A 5 "Power Readings" | grep "Power Draw" | sed \'s/[^0-9.]*//g\''.format(metrics_path)
        res = subprocess.run(read_nvidia_cmd, shell=True, stdout=subprocess.PIPE)
        nvidia_res = res.stdout.splitlines()
        # print(nvidia_res)
        # Ditch the last 70 values (~5 seconds worth) because they correspond to sqlite, not the app
        nvidia_power = sum([float(val) for val in nvidia_res[:-70]])/len(nvidia_res[:-70])
        return (nvidia_power, perf_time, cpu_energy, ddr_energy)

@ch_time_block.decor(print_start=False, print_args=True)
def get_data(metrics_path: Path) -> Tuple[Any]:
    power_data = read_illixr_power(str(metrics_path))

    with warnings.catch_warnings(record=True) as warnings_log:
        with ch_time_block.ctx("load sqlite", print_start=False):
            plugin_name            = read_illixr_table(metrics_path, "plugin_name"             , ["plugin_id"])
            threadloop_iteration   = read_illixr_table(metrics_path, "threadloop_iteration"    , ["plugin_id", "iteration_no"])
            switchboard_callback   = read_illixr_table(metrics_path, "switchboard_callback"    , ["plugin_id", "iteration_no"])
            switchboard_topic_stop = read_illixr_table(metrics_path, "switchboard_topic_stop"  , ["topic_name"])
            switchboard_check_qs   = read_illixr_table(metrics_path, "switchboard_check_queues", ["iteration_no"])
            timewarp_gpu           = read_illixr_table(metrics_path, "timewarp_gpu"            , ["iteration_no"])
            imu_cam                = read_illixr_table(metrics_path, "imu_cam"                 , ["iteration_no"])
            #camera_cvtfmt          = read_illixr_table(metrics_path, "camera_cvtfmt"           , ["iteration_no"])
            try:
                # try...except for "backwards compatibility reasons"
                mtp                = read_illixr_table(metrics_path, "mtp_record"              , ["iteration_no"])
                if "imu_to_display" not in mtp:
                    raise NotImplementedError
            except Exception:
                warnings.warn("Using fake data for mtp")
                mtp = pd.DataFrame.from_dict(dict(
                    iteration_no=[0, 1, 2],
                    imu_to_display=[10e6, 20e6, 14e6],
                    predict_to_display=[3e6, 4e6, 4e6],
                    render_to_display=[5e6, 466, 7e6],
                    vsync=[1e9, 1.1e9, 1.2e9],
                ))

            # This is an integer in SQLite, because no bool in SQLite.
            # Convert it to a bool.
            imu_cam["has_camera"]  = imu_cam["has_camera"] == 1


        with ch_time_block.ctx("load csv", print_start=False):
            thread_ids = read_illixr_csv(metrics_path, "thread", [], ["thread_id", "name", "sub_name"])
            stdout_cpu_timer = read_illixr_csv(metrics_path, "cpu_timer", ["account_name", "iteration_no"], ["wall_time_start", "wall_time_stop", "cpu_time_start", "cpu_time_stop"])
            stdout_gpu_timer = read_illixr_csv(metrics_path, "gpu_timer", ["account_name", "iteration_no"], ["wall_time_start", "wall_time_stop", "gpu_time_duration"])
            for df in [timewarp_gpu, stdout_gpu_timer]:
                df["gpu_time_duration"] = pd.to_numeric(df["gpu_time_duration"])
                splice_mask = df["gpu_time_duration"] > 1e10
                if splice_mask.sum() > 5:
                    warnings.warn(UserWarning(
                        f"gpu_timer had {splice_mask.sum()} splices, which is kind of high."
                    ))
                df.drop(df.index[splice_mask], inplace=True)
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
                .agg(dict_concat(
                    {
                        f"{clock}_time_duration": sum,
                        f"{clock}_time_start"   : min,
                        f"{clock}_time_stop"    : max,
                    }
                    for clock in clocks
                ))
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
                #set_account_name(camera_cvtfmt, "camera_cvtfmt"),
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
            if "runtime check_qs" in ts:
                used += ts.loc["runtime check_qs", "cpu_time_duration"].sum()
                start = min(start, ts.loc["runtime check_qs", "cpu_time_start"].min())
                stop  = max(stop , ts.loc["runtime check_qs", "cpu_time_stop" ].max())
                thread_ids.loc[thread_ids["sub_name"] == "0", "total_cpu_time_usage"] = stop - start
                thread_ids.loc[thread_ids["sub_name"] == "0", "missing_cpu_time_usage"] = stop - start - used
            else:
                warnings.warn("'runtime check_qs' account not found. Skipping.")

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
            if not has_zed:
                assert "offline_imu_cam iter" in ts.index.levels[0]
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
            ts.loc["opencv", "wall_time_start"] += ts.loc["slam2 cb cam", "wall_time_start"].iloc[0]
            ts = concat_accounts(ts, "opencv", "slam2 cb cam", "slam2 cb cam")

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

        mtp["imu_to_display"] = mtp["imu_to_display"] / 1e6
        mtp["predict_to_display"] = mtp["predict_to_display"] / 1e6
        mtp["render_to_display"] = mtp["render_to_display"] / 1e6
        mtp["wall_time"] = (mtp["vsync"] - mtp["vsync"].iloc[0]) / 1e6
        del mtp["vsync"]

        return ts, summaries, switchboard_topic_stop, thread_ids, warnings_log, power_data, mtp

# @ch_cache.decor(ch_cache.FileStore.create(Path(".cache/")))
def get_data_cached(metrics_path: Path) -> Tuple[Any]:
    return get_data(metrics_path)

# TODO: use ../metrics/blah/ instead of ../metrics-blah/
run_list = [
    path.name.partition("-")[2]
    for path in list(Path("..").iterdir())
    if path.name.endswith("metrics-")
]

sponza_list = [
    "metrics-jetsonlp-sponza",
    "metrics-jetsonhp-sponza",
    "metrics-desktop-sponza",
]
materials_list = [
    "metrics-jetsonlp-materials",
    "metrics-jetsonhp-materials",
    "metrics-desktop-materials",
]
platformer_list = [
    "metrics-jetsonlp-platformer",
    "metrics-jetsonhp-platformer",
    "metrics-desktop-platformer",
]
demo_list = [
    "metrics-desktop-demo",
    "metrics-jetsonhp-demo",
    "metrics-jetsonlp-demo",
]

fps_spreadsheet_sponza = pd.DataFrame()
fps_spreadsheet_materials = pd.DataFrame() 
fps_spreadsheet_platformer = pd.DataFrame() 
fps_spreadsheet_demo = pd.DataFrame() 

power_spreadsheet = pd.DataFrame() 
cpu_spreadsheet = pd.DataFrame() 
gpu_spreadsheet = pd.DataFrame() 
timeseries_spreadsheet = pd.DataFrame() 

replaced_names = {
    'app': 'Application',
    'zed_imu_thread iter': 'IMU',
    'zed_camera_thread iter': 'Camera',
    'timewarp_gl iter': 'Reprojection',
    'hologram iter': 'Hologram',
    'audio_encoding iter': 'Encoding',
    'audio_decoding iter': 'Playback',
    'imu_integrator iter': 'IMU Integrator',

    # GPU Values
    'timewarp_gl gpu': 'Reprojection',
    'hologram': 'Hologram',
}


def write_graphs(
        name_list: List[str],
        ignore_accounts=List[str],
) -> None:
    for run_name in tqdm(name_list):
        metrics_path = Path("../metrics") / f"{run_name}"
        ts, summaries, switchboard_topic_stop, thread_ids, warnings_log, power_data, mtp = get_data_cached(metrics_path)

        accounts = ts.index.levels[0]

        fig = plt.figure()
        ax = fig.subplots()
        for account in set(accounts) - set(ignore_accounts):
            xs = (ts.loc[account, "wall_time_start"] - ts.loc[account, "wall_time_start"].iloc[0]) / 1e3
            ys = ts.loc[account, "wall_time_duration"]
            xs = xs.iloc[10:]
            ys = ys.iloc[10:]
            ax.plot(xs, ys, label=account)
            ax.set_xlabel("Time (seconds after program start)")
            ax.set_ylabel("Wall-time Duration (ms)")
            #print(account)
            #print(ys)
            #ys.to_csv(f"{account}.csv", index=False)
        ax.legend()
        ax.set_title("Wall-Time Duration by Component")
        fig.savefig(metrics_path / "wall_time_durations.png")


# trials: List[PerTrialData] = []
# for metrics_path in Path("../metrics").iterdir():
#     if not (metrics_path / "trial_conditions.yaml").exists():
#         warnings.warn(f"{metrics_path!s} does not contain `trial_conditions.yaml`. Skipping analysis.")
#         continue
#     with (metrics_path / "trial_conditions.yaml").open() as f: 
#         conditions: Dict[str, str] = yaml.safe_load(f)
#         conditions_obj = TrialConditions(**conditions)

#     ts, summaries, switchboard_topic_stop, thread_ids, warnings_log, power_data, mtp = get_data(metrics_path)
#     output_path = Path("../output") / metrics_path.name
#     output_path.mkdir(exist_ok=True, parents=True)
#     trial = PerTrialData(ts = ts, summaries = summaries, thread_ids = thread_ids, output_path = output_path, switchboard_topic_stop = switchboard_topic_stop, mtp = mtp, warnings_log = warnings_log, conditions = conditions_obj, power_data = power_data)
#     trials.append(trial)
#     # per_trial_analysis(trial)
# inter_trial_analysis(trials, replaced_names)

# plot_graphs.plot_fps(0)
# plot_graphs.plot_fps(1)
# plot_graphs.plot_fps(2)
# plot_graphs.plot_cpu()
# plot_graphs.plot_gpu()
# plot_graphs.plot_power()
# plot_graphs.plot_mtp(['desktop-sponza', 'jetsonhp-sponza', 'jetsonlp-sponza'], 'Sponza', 180)
# plot_graphs.plot_mtp(['desktop-materials', 'jetsonhp-materials', 'jetsonlp-materials'], 'Materials', 180)
# plot_graphs.plot_mtp(['desktop-platformer', 'jetsonhp-platformer', 'jetsonlp-platformer'], 'Platformer', 120)
# plot_graphs.plot_mtp(['desktop-demo', 'jetsonhp-demo', 'jetsonlp-demo'], 'AR', 60)
plot_graphs.plot_frame_time(0)