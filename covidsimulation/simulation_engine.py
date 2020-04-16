from typing import List, Dict, Tuple, Optional
from copy import copy, deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import simpy

from . import simulation as cs
from .cache import get_from_cache, save_to_cache
from .lab import laboratorio
from .parameters import Parameters
from .population import Population
from .stats import Stats
from .metrics import METRICS

SIMULATION_ENGINE_VERSION = '0.0.1'


def get_matriz_estatisticas(env, duracao):
    num_populacoes = len(env.populacoes)
    num_metricas = len(cs.MEASUREMENTS)
    num_faixas = len(cs.age_str)
    stats = np.zeros([num_populacoes, num_metricas, num_faixas, duracao])
    return stats


def monitorar_populacao(env):
    while True:
        yield env.timeout(1.0)
        if env.d0 is None:
            if env.sim_params.d0_infections * env.scaling < np.array([p.infectado for p in env.pessoas]).sum():
                env.d0 = int(env.now + 0.01)
            else:
                continue
        if int(env.now + 0.01) - env.d0 >= env.duracao:
            return
        cs.log_estatisticas(env)
        if env.tqdm:
            env.tqdm.update(1)


#        if (env.sim_number % 16) == 0:  # Don't show progress
#            print(int(env.now+0.01))


def get_tamanho_casa(tamanho_casas):  # Número de pessoas morando na mesma casa
    return np.random.choice(len(tamanho_casas), p=tamanho_casas)


def get_grupo_idade(probabilidade_faixas, risco_faixas):
    return np.random.choice(risco_faixas, p=probabilidade_faixas)


def setar_infeccao_inicial(env, pessoas):
    sucesso = False
    while not sucesso:
        alguem = cs.choice(pessoas, 1)[0]
        if alguem.age_group.indice_faixa < env.sim_params.min_age_group_initially_infected:
            continue
        sucesso = alguem.expose_to_virus()


def get_populacao(env, param_populacao: Population):
    pessoas = []
    n = int(param_populacao.inhabitants * env.scaling)
    infectados_iniciais = param_populacao.seed_infections
    while len(pessoas) < n:
        pessoas.extend(generate_pessoas_em_nova_casa(env, param_populacao))
    for _ in range(infectados_iniciais):
        setar_infeccao_inicial(env, pessoas)
    return pessoas


def generate_pessoas_em_nova_casa(env, param_populacao: Population):
    tamanho_casa = get_tamanho_casa(param_populacao.home_size_probabilities)
    casa = cs.Home(param_populacao.geosocial_displacement)
    probabilidade_faixas = param_populacao.age_probabilities
    risco_faixas = param_populacao.age_groups
    grupo_idade_casa = get_grupo_idade(probabilidade_faixas, risco_faixas)
    for _ in range(tamanho_casa):
        grupo_idade = (grupo_idade_casa
                       if np.random.random() < env.sim_params.home_age_cofactor
                       else get_grupo_idade(probabilidade_faixas, risco_faixas)
                       )
        yield cs.Person(env, grupo_idade, casa)


def aplica_isolamento(env, dia_inicio, fator_isolamento):
    yield env.timeout(dia_inicio - env.now)
    while env.d0 is None or dia_inicio > env.now - env.d0:
        yield env.timeout(1.0)
    desvio_logit = (env.desvio_isolamento - 0.5) / 5.0
    fator_isolamento = np.power(fator_isolamento, 0.65)
    env.fator_isolamento = cs.logit_transform_value(fator_isolamento, desvio_logit)
    for pessoa in env.pessoas:
        pessoa.em_isolamento = pessoa.home.isolation_propensity < env.fator_isolamento


def cria_populacoes(env):
    populacoes = {}
    for param_populacao in env.sim_params.population_segments:
        for i, grupo_idade in enumerate(param_populacao.age_groups):
            grupo_idade_modificado = copy(grupo_idade)
            severidades = np.array(grupo_idade_modificado.severidades)
            desvio_faixa = env.inclinacao_severidade * (i - 4)
            novos_odds = np.exp(np.log(severidades / (1.0 - severidades)) - env.desvio_severidade + desvio_faixa)
            grupo_idade_modificado.severidades = novos_odds / (1.0 + novos_odds)
            grupo_idade_modificado.adesao_isolamento += param_populacao.isolation_propensity_increase
            param_populacao.age_groups[i] = grupo_idade_modificado
        populacoes[param_populacao.name] = get_populacao(env, param_populacao)
    return populacoes


def simulate(
        sim_number,
        sim_params,
        simulation_size,
        duration,
        simulate_capacity,
        add_noise,
        use_cache,
        tqdm=None,
):
    if use_cache:
        args = (
        sim_number, sim_params, simulation_size, duration, simulate_capacity, add_noise, SIMULATION_ENGINE_VERSION)
        results = get_from_cache(args)
        if results:
            return results[1]
    cs.seed(sim_number)
    np.random.seed(sim_number)
    env = simpy.Environment()
    env.sim_params = deepcopy(sim_params)
    env.duracao = duration
    env.sim_number = sim_number
    print('', end='', flush=True)  # Trick for tqdm in Jupyter notebooks
    if tqdm:
        env.tqdm = tqdm(total=env.duracao, position=env.sim_number + 1)
        env.tqdm.update(0)
    else:
        env.tqdm = None
    scaling = simulation_size / sim_params.total_inhabitants
    env.d0 = None  # Esperando o dia para iniciar logs
    env.simula_capacidade = simulate_capacity
    env.desvio_severidade = (np.random.random() + np.random.random() - 1.0) * 0.2 if add_noise else 0.0
    env.inclinacao_severidade = (np.random.random() - 0.5) * 0.2 if add_noise else 0.0
    env.desvio_isolamento = np.random.random() if add_noise else 0.0  # incerteza na efetividade do isolamento
    env.tempo_medio_entre_contagios = sim_params.transmission_scale_days + (
        (np.random.random() - 0.5) * 0.1 if add_noise else 0.0)
    env.fator_isolamento = 0.0
    env.scaling = scaling
    env.atencao = simpy.resources.resource.PriorityResource(env,
                                                            capacity=int(sim_params.capacity_hospital_max * scaling))
    env.leito = simpy.resources.resource.PriorityResource(env,
                                                          capacity=int(sim_params.capacity_hospital_beds * scaling))
    env.ventilacao = simpy.resources.resource.PriorityResource(env,
                                                               capacity=int(sim_params.capacity_ventilators * scaling))
    env.uti = simpy.resources.resource.PriorityResource(env, capacity=int(sim_params.capacity_intensive_care * scaling))
    env.process(laboratorio(env))
    env.populacoes = cria_populacoes(env)
    env.stats = get_matriz_estatisticas(env, duration)
    env.pessoas = []
    for pessoas in env.populacoes.values():
        env.pessoas.extend(pessoas)
    env.process(monitorar_populacao(env))
    for dia_inicio, fator_isolamento in sim_params.distancing:
        env.process(aplica_isolamento(env, dia_inicio, fator_isolamento))
    env.run(until=duration)
    env.run(until=duration + env.d0 + 0.011)
    if env.tqdm:
        env.tqdm.close()
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
    simulate_with_params = partial(simulate,
                                   sim_params=sim_params,
                                   simulation_size=simulation_size,
                                   duration=duration,
                                   simulate_capacity=simulate_capacity,
                                   add_noise=add_noise,
                                   use_cache=use_cache,
                                   tqdm=tqdm if number_of_simulations <= cpu_count() else None,
                                   )
    try:
        pool = Pool(min(cpu_count(), number_of_simulations))
        all_stats = pool.imap(simulate_with_params, range(number_of_simulations))
        if tqdm:
            all_stats = tqdm(all_stats, total=number_of_simulations, position=0)
        all_stats = list(all_stats)
    finally:
        pool.close()
        pool.join()
    stats = combina_stats(all_stats, sim_params)
    if fpath:
        stats.save(fpath)
    return stats


def combina_stats(all_stats: List[np.ndarray], sim_params: Parameters):
    mstats = np.stack(all_stats)
    population_names = tuple(p.name for p in sim_params.population_segments)
    return Stats(mstats, cs.MEASUREMENTS, METRICS, population_names, cs.age_str, start_date=sim_params.start_date)
