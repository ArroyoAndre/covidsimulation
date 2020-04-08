import pytest
import numpy as np
from covidsimulation import run_simulations
from covidsimulation.regions import br_saopaulo


def test_run_simulations():
    run_simulations(br_saopaulo.params, isolamentos=[], n=1, duracao=10, 
                    tamanho_simulacao=50000)
