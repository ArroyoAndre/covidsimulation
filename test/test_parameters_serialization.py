import pytest
from multiprocessing import Pool

from covidsimulation.cache import get_signature


def get_parameters(arg):
    from covidsimulation.regions.br_saopaulo import params as br_saopaulo_params
    return br_saopaulo_params


def test_parameters_have_stable_signature():
    p = Pool(2)
    params = p.map(get_parameters, [0, 1])
    assert get_signature(params[0]) == get_signature(params[1])
