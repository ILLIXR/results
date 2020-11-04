from typing import TypeVar, Iterable, List, Dict
import re 
import itertools
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
