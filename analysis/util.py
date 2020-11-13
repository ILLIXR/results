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
    machine: str 
    application: str
    illixr_commit: str 

class PerTrialData(NamedTuple):
    ts: pd.DataFrame
    summaries: pd.DataFrame
    thread_ids: pd.DataFrame
    output_path: Path
    switchboard_topic_stop: pd.DataFrame
    mtp: pd.DataFrame
    warnings_log: List[WarningMessage]
    conditions: TrialConditions
    power_data: List[float]
