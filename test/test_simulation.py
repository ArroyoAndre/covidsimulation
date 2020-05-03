import pytest
import numpy as np
from covidsimulation.regions.br_saopaulo import params as br_saopaulo_params
from covidsimulation.intervention import HygieneAdoption, MaskUsage
from covidsimulation import run_simulations, plot, Stats
from tqdm import tqdm


def test_run_simulations():
    stats = run_simulations(
        sim_params=br_saopaulo_params, 
        simulate_capacity=False,
        duration=10, 
        number_of_simulations=1, 
        simulation_size=5000, 
        fpath='saved/teste.pkl',
        use_cache=False,
        tqdm=tqdm,
        )
    average_infected = stats.get_metric('infected').mean.tolist()
    assert average_infected[9] > average_infected[0] 

def test_run_simulations_cached():
    stats = run_simulations(
        sim_params=br_saopaulo_params, 
        simulate_capacity=False,
        duration=5, 
        number_of_simulations=1, 
        simulation_size=5000, 
        fpath='saved/teste.pkl',
        use_cache=True,
        tqdm=tqdm,
        )
    stats2 = run_simulations(
        sim_params=br_saopaulo_params, 
        simulate_capacity=False,
        duration=5, 
        number_of_simulations=1, 
        simulation_size=5000, 
        fpath='saved/teste.pkl',
        use_cache=True,
        tqdm=tqdm,
        )
    assert stats.stats.tobytes() == stats2.stats.tobytes()

def test_run_simulations_with_hygiene_interventions():
    params = br_saopaulo_params.clone()
    params.interventions.extend([
        HygieneAdoption(0, 0.65),
        MaskUsage(1, 0.6),
    ])
    stats = run_simulations(
        sim_params=params,
        simulate_capacity=False,
        duration=10,
        number_of_simulations=1,
        simulation_size=5000,
        fpath='saved/teste.pkl',
        use_cache=False,
        tqdm=tqdm,
        )
    average_infected = stats.get_metric('infected').mean.tolist()
    assert average_infected[9] > average_infected[0]
