
from typing import List, Dict, Tuple, Optional
from copy import copy, deepcopy
from functools import partial
from multiprocessing import Manager, Pool, Queue, cpu_count


import numpy as np
import simpy
from copy import copy
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from . import simulation as cs
from .cache import get_from_cache, save_to_cache
from .lab import laboratory
from .parameters import Parameters
from .population import Population
from .progress import ProgressBar
from .stats import Stats
from .metrics import METRICS

SIMULATION_ENGINE_VERSION = '0.0.1'

def get_stats_matrix(env, duration):
    num_populations = len(env.populations)
    num_metrics = len(cs.MEASUREMENTS)
    num_ages = len(cs.age_str)
    stats = np.zeros([num_populations, num_metrics, num_ages, duration])
    return stats


def track_population(env):
    while True:
        yield env.timeout(1.0)
        if env.d0 is None:
            if env.sim_params.d0_infections * env.scaling < np.array([p.infected for p in env.people]).sum():
                env.d0 = int(env.now+0.01)
            else:
                continue
        if int(env.now + 0.01) - env.d0 >= env.duration:
            return
        cs.log_stats(env)
        if env.simulation_queue:
            env.simulation_queue.put(1)


def get_house_size(house_sizes):  # Number of people living in the same house
    return np.random.choice(len(house_sizes), p=house_sizes)


def get_age_group(age_probabilities, age_risk):
    return np.random.choice(age_risk, p=age_probabilities)



def set_initial_infection(env, people):
    success = False
    while not success:
        someone = cs.choice(people, 1)[0]
        if someone.age_group.index < env.sim_params.min_age_group_initially_infected:
            continue
        success = someone.expose_to_virus()


def get_population(env, population_params):
    people = []
    n = int(population_params.inhabitants * env.scaling)
    initially_infected = population_params.seed_infections
    while len(people) < n:
        people.extend(generate_people_in_new_house(env, population_params))
    for _ in range(initially_infected):
        set_initial_infection(env, people)
    return people


def generate_people_in_new_house(env, population_params):
    house_size = get_house_size(population_params.home_size_probabilities)
    house = cs.Home(population_params.geosocial_displacement)
    age_probabilities = population_params.age_probabilities
    age_groups = population_params.age_groups
    age_group_house = get_age_group(age_probabilities, age_groups)
    for _ in range(house_size):
        age_group = (age_group_house
                     if np.random.random() < env.sim_params.home_age_cofactor
                     else get_age_group(age_probabilities, age_groups)
                     )
        yield cs.Person(env, age_group, house)


def apply_isolation(env, start_date, isolation_factor):
    yield env.timeout(start_date - env.now)
    while env.d0 is None or start_date > env.now - env.d0:
        yield env.timeout(1.0)

    logit_deviation = (env.isolation_deviation - 0.5) / 5.0
    isolation_factor = np.power(isolation_factor, 0.65)
    env.isolation_factor = cs.logit_transform_value(
        isolation_factor, logit_deviation)
    for person in env.people:
        person.in_isolation = person.home.isolation_propensity < env.isolation_factor


def create_populations(env):
    populations = {}
    for population_params in env.sim_params.population_segments:
        for i, age_group in enumerate(population_params.age_groups):
            age_group_cp = copy(age_group)
            severity = np.array(age_group_cp.severity)
            age_bias = env.severity_bias * (i - 4)
            new_odds = np.exp(np.log(severity / (1.0 - severity)
                                     ) - env.severity_deviation + age_bias)
            age_group_cp.severity = new_odds / (1.0 + new_odds)
            population_params.age_groups[i] = age_group_cp
        populations[population_params.name] = get_population(env, population_params)
    return populations

def simulate(
        sim_number,
        sim_params,
        simulation_size,
        duration,
        simulate_capacity,
        add_noise,
        use_cache,
        creation_queue: Optional[Queue] = None,
        simulation_queue: Optional[Queue] = None,
):
    if use_cache:
        args = (
            sim_number, sim_params, simulation_size, duration, simulate_capacity, add_noise, SIMULATION_ENGINE_VERSION)
        results = get_from_cache(args)
        if results:
            if creation_queue:
                creation_queue.put(simulation_size)
            if simulation_queue:
                simulation_queue.put(duration)
            return results[1]
 
    cs.seed(sim_number)
    np.random.seed(sim_number)
    env = simpy.Environment()
    env.creation_queue = creation_queue
    env.simulation_queue = simulation_queue
    env.sim_params = deepcopy(sim_params)
    env.duration = duration
    env.sim_number = sim_number
    scaling = simulation_size / sim_params.total_inhabitants
    env.sim_number = sim_number
    env.d0 = None
    env.simulate_capacity = simulate_capacity
    env.severity_deviation = (np.random.random() +
                             np.random.random() - 1.0) * 0.2
    env.severity_bias = (np.random.random() - 0.5) * 0.2
    env.isolation_deviation = np.random.random()  # uncertainty regarding isolation effectiveness
    env.serial_interval = sim_params.transmission_scale_days + \
        (np.random.random() - 0.5) * 0.1
    env.isolation_factor = 0.0
    env.scaling = scaling
    env.attention = simpy.resources.resource.PriorityResource(
        env, capacity=int(sim_params.capacity_hospital_max * scaling))
    env.hospital_bed = simpy.resources.resource.PriorityResource(
        env, capacity=int(sim_params.capacity_hospital_beds * scaling))
    env.ventilator = simpy.resources.resource.PriorityResource(
        env, capacity=int(sim_params.capacity_ventilators * scaling))
    env.icu = simpy.resources.resource.PriorityResource(
        env, capacity=int(sim_params.capacity_icu * scaling))
    env.process(laboratory(env))
    env.populations = create_populations(env)
    env.stats = get_stats_matrix(env, duration)
    env.people = []
    for people in env.populations.values():
        env.people.extend(people)
    env.process(track_population(env))
    for start_date, isolation_factor in sim_params.distancing:
        env.process(apply_isolation(env, start_date, isolation_factor))
    while not env.d0:
        env.run(until=env.now + 1)
    env.run(until=duration + env.d0 + 0.011)
    stats = env.stats / env.scaling
    if use_cache:
        save_to_cache(args, stats)
    return stats


def run_simulations(
        sim_params: Parameters,
        distancing_list: Optional[List[Tuple[float, float]]] = None,  # Set to override sim_param's default distancing
        simulate_capacity=False,
        duration: int = 80,
        number_of_simulations: int = 4,  # For final presentation purposes, a value greater than 10 is recommended
        simulation_size: int = 100000,  # For final presentation purposes, a value greater than 500000 is recommended
        fpath=None,
        add_noise=True,  # Simulate uncertainty about main parameters and constants
        use_cache=True,
        tqdm=None,  # Optional tqdm function to display progress
):
    if not distancing_list is None:
        sim_params = deepcopy(sim_params)
        sim_params.distancing = distancing_list

    if tqdm:
        manager = Manager()
        creation_queue = manager.Queue()
        simulation_queue = manager.Queue()

    simulate_with_params = partial(simulate,
                                   sim_params=sim_params,
                                   simulation_size=simulation_size,
                                   duration=duration,
                                   simulate_capacity=simulate_capacity,
                                   add_noise=add_noise,
                                   use_cache=use_cache,
                                   creation_queue=creation_queue if tqdm else None,
                                   simulation_queue=simulation_queue if tqdm else None,
                                   )
    try:
        pool = Pool(min(cpu_count(), number_of_simulations))
        all_stats = pool.imap(simulate_with_params, range(number_of_simulations))
        if tqdm:
            creation_bar, simulation_bar = show_progress(tqdm, creation_queue, simulation_queue, simulation_size,
                                                         number_of_simulations, duration)
            creation_bar.start()
            simulation_bar.start()
        all_stats = list(all_stats)
    finally:
        pool.close()
        pool.join()
        if tqdm:
            creation_bar.stop()
            creation_bar.join()
            simulation_bar.stop()
            simulation_bar.join()
    stats = combine_stats(all_stats, sim_params)
    if fpath:
        stats.save(fpath)
    return stats


def combine_stats(all_stats: List[np.ndarray], sim_params: Parameters):
    mstats = np.stack(all_stats)
    population_names = tuple(p.name for p in sim_params.population_segments)
    return Stats(mstats, cs.MEASUREMENTS, METRICS, population_names, cs.age_str, start_date=sim_params.start_date)


def show_progress(tqdm, creation_queue: Queue, simulation_queue: Queue, simulation_size: int,
                  number_of_simulations: int, duration: int):
    creation_bar = ProgressBar(tqdm, creation_queue, simulation_size * number_of_simulations, 0, 'Population')
    simulation_bar = ProgressBar(tqdm, simulation_queue, duration * number_of_simulations, 1, 'Simulation')
    return creation_bar, simulation_bar