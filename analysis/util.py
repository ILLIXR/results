from typing import TypeVar, Iterable, List, Dict
from typing import NamedTuple
import re 
import itertools
import pandas as pd 
from pathlib import Path
from warnings import WarningMessage

T = TypeVar("T")
V = TypeVar("V")

def it_concat(its: Iterable[Iterable[T]]) -> Iterable[T]:
    return itertools.chain.from_iterable(its)

def list_concat(lsts: Iterable[List[T]]) -> List[T]:
    return list(itertools.chain.from_iterable(lsts))

def dict_concat(dcts: Iterable[Dict[T, V]], **kwargs: V) -> Dict[T, V]:
    return dict(it_concat(dct.items() for dct in dcts), **kwargs)
    
def is_int(string: str) -> bool:
    return bool(re.match(r"^\d+$", string))

def is_float(string: str) -> bool:
    return bool(re.match(r"^\d+.\d+(e[+-]\d+)?$", string))

class TrialConditions (NamedTuple):

    # Something like "desktop", "jetson-lp", or "jetson-hp"
    machine: str 

    # Something like "platformer", or "sponza"
    application: str

    # Run `git rev-parse HEAD` in the ILLIXR repo to get this.
    illixr_commit: str 

class PerTrialData(NamedTuple):

    # [t]ime[s]eries of each account (account is like "OpenVINS", "timewarp_gl", or "runtime").
    # Each row has "iteration_no" (index), "start_wall_time", "end_wall_time", etc.
    ts: pd.DataFrame

    # summaries of every account in the system
    # Each row has "account_name" (index), "avg_wall_time", "avg_cpu_time", etc.
    summaries: pd.DataFrame

    # thread_ids discovered in the system; Useful for correlation with external profiling tools.
    # Each row has "thread_id", "name", "sub_name".
    thread_ids: pd.DataFrame

    # Each row has "topic_name" (index), "processed", "unprocessed", "completion" (completion ratio)
    # This is useful to tell if frames were still in the switchboard buffer at shutdown-time.
    switchboard_topic_stop: pd.DataFrame

    # [m]otion-[t]o-[p]hoton
    # Each row (conceptually a "vsync-graphics-cycle") has "imu_to_display", "predict_to_display", "render_to_display", and "wall_time" (of vsync)
    mtp: pd.DataFrame

    # Power data parsed out of external tools.
    # See analysis/main.py:read_illixr_power for definition
    power_data: List[float]

    # The path that analysis output should go.
    output_path: Path

    # List of warnings that occured when parsing the data.
    warnings_log: List[WarningMessage]

    # TrialConditions parsed from the YAML description.
    conditions: TrialConditions
