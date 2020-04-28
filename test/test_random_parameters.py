import pytest
import numpy as np
from covidsimulation.random import RandomParametersState, UniformParameter, ParameterSum
from dataclasses import dataclass
from typing import List, Any


def test_random_parameters_materialization():

    @dataclass
    class TopClass:
        subclasses: List
        static_value: Any
        random_value: Any

    @dataclass
    class SubClass:
        static_value: Any
        random_value: Any
        random_value_dict: Any
        random_value_list: Any

        def dont_call_me(self):
            assert False

    random_state = RandomParametersState()
    top = TopClass(
        subclasses=[SubClass(
            static_value='45',
            random_value=UniformParameter('val1', 0.0, 1.0),
            random_value_dict={'my_val': UniformParameter('val2', 0.0, 1.0)},
            random_value_list=[UniformParameter('val3', 0.0, 1.0)],
        )],
        static_value='myname',
        random_value=np.array([UniformParameter('val4', 0.0, 1.0)])
    )
    assert str(top).find('Uniform') > 0
    random_state.materialize_object(top)
    assert 'val1' in random_state.state.keys()
    assert 'val2' in random_state.state.keys()
    assert 'val3' in random_state.state.keys()
    assert 'val4' in random_state.state.keys()
    assert str(top).find('Uniform') == -1


def test_random_sum():
    state = RandomParametersState()
    v = (1 + UniformParameter('p1', 0.0, 2.0) + 2) * 2 + 3 * UniformParameter('p2', 0.0, 1.0)
    k = [v]
    state.materialize_object(k)
    assert 'p1' in state.state.keys()
    assert 'p2' in state.state.keys()
    assert isinstance(k[0], float)
