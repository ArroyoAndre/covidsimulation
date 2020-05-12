from typing import List, Tuple, Union, Iterable, Callable, Optional
import numpy as np
from functools import partial
from copy import deepcopy
from itertools import product
from multiprocessing import Manager, Pool, cpu_count

from . import Parameters, Stats
from .utils import get_date_from_isoformat
from .random import RandomParametersState
from .simulation_engine import simulate_wrapped, combine_stats, show_progress, get_sim_params_list


LSE_REGULARIZATOR = 60.0  # Logarithmic Squared Error regularization factor, to diminish the weight


# of errors between small quantities, in which random noise might represent
# a large portion of observed deviations


def calibrate_parameters(
        parameters_to_try: List[Tuple[Callable, Iterable]],
        score_function: Callable,
        sim_params: Parameters,
        simulate_capacity=False,
        duration: int = 80,
        simulation_size: int = 100000,
        n: int = 10,
        use_cache=True,
        tqdm=None,
):
    """
    parameters_to_try: List of tuples (setting_fn, [values]). The function setting_fn must take 2
        parameters (sim_params: Parameters, value) and must modify sim_params.
    score_fn: Function that receives a Stats object and returns a double that needs to be minimized.
    sim_params: Parameters that will be used as a template for each simulation.
    """
    sim_params_list, combinations = get_simulation_parameters(sim_params, parameters_to_try, n)

    if tqdm:
        manager = Manager()
        creation_queue = manager.Queue()
        simulation_queue = manager.Queue()


    simulate_with_params = partial(simulate_wrapped,
                                   simulation_size=simulation_size,
                                   duration=duration,
                                   simulate_capacity=simulate_capacity,
                                   use_cache=use_cache,
                                   creation_queue=creation_queue if tqdm else None,
                                   simulation_queue=simulation_queue if tqdm else None,
                                   )
    try:
        pool = Pool(min(cpu_count(), len(sim_params_list)))
        all_stats = pool.imap(simulate_with_params, sim_params_list)
        if tqdm:
            creation_bar, simulation_bar = show_progress(tqdm, creation_queue, simulation_queue, simulation_size,
                                                         len(sim_params_list), duration)
            creation_bar.start()
            simulation_bar.start()
        all_stats = list(all_stats)
        scores = [score_function(combine_stats(stats, sim_params)) for stats in grouper(all_stats, n)]
    finally:
        pool.close()
        pool.join()
        if tqdm:
            creation_bar.stop()
            creation_bar.join()
            simulation_bar.stop()
            simulation_bar.join()

    best = np.argsort(np.array(scores))
    return np.array(combinations)[best[:8]]


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip(*args)


def get_simulation_parameters(sim_params: Parameters, parameters_to_try: List[Tuple[Callable, Iterable]], n):
    sim_params_list = []
    parameters_settings = []
    for _, parameter_values in parameters_to_try:
        parameters_settings.append(list())
        for parameter_value in parameter_values:
            parameters_settings[-1].append(parameter_value)
    combinations = list(product(*parameters_settings))
    for combination in combinations:
        params = deepcopy(sim_params)
        for i, setting in enumerate(combination):
            parameters_to_try[i][0](params, setting)
        for i in range(n):
            sim_params_list.append((i, params))
    return sim_params_list, combinations


def score_reported_deaths(stats: Stats, expected_deaths: List[Tuple[Union[int, str], float]]):
    metric = stats.get_metric('confirmed_deaths').mean
    lse = 0.0
    for day, reporded_deaths in expected_deaths:
        if isinstance(day, str):
            day = get_day_from_isoformat(day, stats.start_date)
        le = np.log((metric[day] + LSE_REGULARIZATOR) / (reporded_deaths + LSE_REGULARIZATOR))
        lse += le ** 2
    return lse


def get_day_from_isoformat(isoformat_date: str, start_date: str):
    return (get_date_from_isoformat(isoformat_date) - get_date_from_isoformat(start_date)).days


def get_best_random_states(
        score_function: Callable,
        sim_params: Parameters,
        random_states: Optional[List[RandomParametersState]],
        simulate_capacity=False,
        duration: int = 80,
        simulation_size: int = 100000,
        n: int = 10,
        p: float = 0.1,
        use_cache=True,
        tqdm=None,
) -> List[RandomParametersState]:
    """
    parameters_to_try: List of tuples (setting_fn, [values]). The function setting_fn must take 2
        parameters (sim_params: Parameters, value) and must modify sim_params.
    score_fn: Function that receives a Stats object and returns a double that needs to be minimized.
    sim_params: Parameters that will be used as a template for each simulation.
    """

    sim_params_list = get_sim_params_list(sim_params, random_states, n)

    if tqdm:
        manager = Manager()
        creation_queue = manager.Queue()
        simulation_queue = manager.Queue()


    simulate_with_params = partial(simulate_wrapped,
                                   simulation_size=simulation_size,
                                   duration=duration,
                                   simulate_capacity=simulate_capacity,
                                   use_cache=use_cache,
                                   creation_queue=creation_queue if tqdm else None,
                                   simulation_queue=simulation_queue if tqdm else None,
                                   )
    try:
        pool = Pool(min(cpu_count(), len(sim_params_list)))
        all_stats = pool.imap(simulate_with_params, sim_params_list)
        if tqdm:
            creation_bar, simulation_bar = show_progress(tqdm, creation_queue, simulation_queue, simulation_size,
                                                         len(sim_params_list), duration)
            creation_bar.start()
            simulation_bar.start()
        all_stats = list(all_stats)
        scores = [score_function(combine_stats(stats, sim_params)) for stats in grouper(all_stats, 1)]
    finally:
        pool.close()
        pool.join()
        if tqdm:
            creation_bar.stop()
            creation_bar.join()
            simulation_bar.stop()
            simulation_bar.join()

    best = np.argsort(np.array(scores))
    num_scores = len(scores) - int((1.0 - p) * len(scores))
    return [all_stats[i][1] for i in best[:num_scores]]


def randomize_states(random_states, multiplier: int=2):
    new_states = []
    for _ in range(1, multiplier):
        for state in random_states:
            new_state = deepcopy(state)
            for key in np.random.choice(list(new_state.state.keys()), 2):
                if key in new_state.state:
                    del new_state.state[key]
            new_states.append(new_state)
    return random_states + new_states
