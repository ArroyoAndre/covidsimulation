import numpy as np
import simpy
from copy import copy
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from . import simulation as cs
from .lab import laboratory
from .parameters import Parameters
from .stats import Stats
from .metrics import METRICS


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
        if (env.sim_number % 16) == 0:
            print(int(env.now+0.01))


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
    n = int(population_params['inhabitants'] * env.scaling)
    initially_infected = population_params['initially_infected']
    while len(people) < n:
        people.extend(generate_people_in_new_house(env, population_params))
    for _ in range(initially_infected):
        set_initial_infection(env, people)
    return people


def generate_people_in_new_house(env, population_params):
    house_size = get_house_size(population_params['house_sizes'])
    house = cs.Home(population_params['deslocamento'])
    age_probabilities = population_params['age_probabilities']
    age_risk = population_params['age_risk']
    age_group_house = get_age_group(age_probabilities, age_risk)
    for _ in range(house_size):
        age_group = (age_group_house
                     if np.random.random() < env.sim_params.home_age_cofactor
                     else get_age_group(age_probabilities, age_risk)
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
        person = person.home.isolation_propensity < env.isolation_factor


def create_populations(env):
    populations = {}
    for population_name, population_params in env.sim_params.population_segments.items():
        for i, age_group in enumerate(population_params['age_risk']):
            age_group_cp = copy(age_group)
            severity = np.array(age_group_cp.severidades)
            age_bias = env.severity_bias * (i - 4)
            new_odds = np.exp(np.log(severity / (1.0 - severity)
                                     ) - env.severity_deviation + age_bias)
            age_group_cp.severity = new_odds / (1.0 + new_odds)
            population_params['age_risk'][i] = age_group_cp
        populations[population_name] = get_population(env, population_params)
    return populations


def simulate(params):
    simulation_size, duration, isolations, simulate_capacity, sim_params, sim_number = params
    for _ in range(sim_number):
        np.random.random()
    cs.seed(sim_number)
    env = simpy.Environment()
    env.sim_params = sim_params
    env.duration = duration
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
        env, capacity=int(sim_params.capacity_intensive_care * scaling))
    env.process(laboratory(env))
    env.populations = create_populations(env)
    env.stats = get_stats_matrix(env, duration)
    env.people = []
    for people in env.populations.values():
        env.people.extend(people)
    env.process(track_population(env))
    for start_date, isolation_factor in isolations:
        env.process(apply_isolation(env, start_date, isolation_factor))
    env.run(until=duration)
    env.run(until=duration+env.d0+0.011)
    return env.stats / env.scaling


def run_simulations(
        sim_params: Parameters,
        isolations: List[Tuple[float, float]],
        simulate_capacity=False,
        duration: int=80,
        n: int=4,  # For final presentation purposes, a value greater than 10 is recommended
        simulation_size: int=100000,  # For final presentation purposes, a value greater than 500000 is recommended
        fpath=None,
):
    params = [simulation_size, duration,
              isolations, simulate_capacity, sim_params]
    try:
        pool = Pool(min(cpu_count(), n))
        all_stats = pool.map(simulate, [params + [i] for i in range(n)])
    finally:
        pool.close()
        pool.join()
    stats = combine_stats(all_stats, sim_params)
    if fpath:
        stats.save(fpath)
    return stats


def combine_stats(all_stats: List[np.ndarray], sim_params: Parameters):
    mstats = np.stack(all_stats)
    population_names = list(sim_params.population_segments.keys())
    return Stats(mstats, cs.MEASUREMENTS, METRICS, population_names, cs.age_str, start_date=sim_params.start_date)
