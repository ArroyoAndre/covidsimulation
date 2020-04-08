import pytest
import numpy as np
from covidsimulation import run_simulations
from covidsimulation.regions import br_saopaulo

def test_run_simulations():
    run_simulations(br_saopaulo.params, isolations=[], n=1, duration=10, 
                    simulation_size=50000)
