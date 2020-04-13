import pytest
import numpy as np
from covidsimulation import run_simulations
from covidsimulation.regions import br_saopaulo


@pytest.fixture
def default_simulation_params():
    return {
        "sim_params": br_saopaulo.params,
        "n": 1,
        "duration": 20,
        "simulation_size": 50000
    }

def test_run_simulations(default_simulation_params):
    run_simulations(isolations=[], **default_simulation_params)


def test_isolation(default_simulation_params):
    with_isolation = run_simulations(isolations=[[0, 0.9]], **default_simulation_params)
    without_isolation = run_simulations(isolations=[], **default_simulation_params)

    metrics = ["deaths", "infected"]
    for metric in metrics:
        assert with_isolation.get_metric(metric)[1][-1] < \
               without_isolation.get_metric(metric)[1][-1]
