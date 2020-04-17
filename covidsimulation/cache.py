from typing import Tuple
import hashlib
from pathlib import Path
import pickle

from .parameters import Parameters

CACHE_BASE_PATH = Path('saved').joinpath('cache')


def get_from_cache(args):
    fpath = get_cache_path(args)
    if fpath.exists():
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    return None


def save_to_cache(args, content):
    fpath = get_cache_path(args)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, 'wb') as f:
        pickle.dump((args, content), f)


def get_cache_path(args):
    signature = get_signature(args)
    return CACHE_BASE_PATH.joinpath(signature[:1]).joinpath(signature[1:3]).joinpath(signature[3:] + '.pkl')


def get_signature(args: Tuple[int, Parameters]):
    return hashlib.blake2b(str(args).encode(), digest_size=8).hexdigest()
