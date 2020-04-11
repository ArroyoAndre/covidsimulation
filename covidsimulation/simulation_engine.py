from typing import List, Dict, Tuple
from copy import copy
from multiprocessing import Pool, cpu_count

import numpy as np
import simpy

from . import simulation as cs
from .lab import laboratorio
from .parameters import Parameters
from .stats import Stats
from .metrics import METRICS


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
                env.d0 = int(env.now+0.01)
            else:
                continue
        if int(env.now +0.01) - env.d0 >= env.duracao:
            return
        cs.log_estatisticas(env)
        if (env.sim_number % 16) == 0:
            print(int(env.now+0.01))


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


def get_populacao(env, param_populacao):
  pessoas = []
  n = int(param_populacao['habitantes'] * env.scaling)
  infectados_iniciais = param_populacao['infectados_iniciais']
  while len(pessoas) < n:
    pessoas.extend(generate_pessoas_em_nova_casa(env, param_populacao))
  for _ in range(infectados_iniciais):
    setar_infeccao_inicial(env, pessoas)
  return pessoas


def generate_pessoas_em_nova_casa(env, param_populacao):
    tamanho_casa = get_tamanho_casa(param_populacao['tamanho_casas'])
    casa = cs.Home(param_populacao['deslocamento'])
    probabilidade_faixas = param_populacao['probabilidade_faixas']
    risco_faixas = param_populacao['risco_faixas']
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
  for nome_populacao, param_populacao in env.sim_params.population_segments.items():
    for i, grupo_idade in enumerate(param_populacao['risco_faixas']):
        grupo_idade_modificado = copy(grupo_idade)
        severidades = np.array(grupo_idade_modificado.severidades)
        desvio_faixa = env.inclinacao_severidade * (i - 4)
        novos_odds = np.exp(np.log(severidades / (1.0 - severidades)) - env.desvio_severidade + desvio_faixa)
        grupo_idade_modificado.severidades = novos_odds / (1.0 + novos_odds)
        param_populacao['risco_faixas'][i] = grupo_idade_modificado
    populacoes[nome_populacao] = get_populacao(env, param_populacao)
  return populacoes


def simula(params):
  tamanho_simulacao, duracao, isolamentos, simula_capacidade, sim_params, add_noise, sim_number = params
  for _ in range(sim_number):
    np.random.random()
  cs.seed(sim_number)
  np.random.seed(sim_number)
  env = simpy.Environment()
  env.sim_params = sim_params
  env.duracao = duracao
  scaling = tamanho_simulacao / sim_params.total_inhabitants
  env.sim_number = sim_number
  env.d0 = None  # Esperando o dia para iniciar logs
  env.simula_capacidade = simula_capacidade
  env.desvio_severidade = (np.random.random() + np.random.random() - 1.0) * 0.2 if add_noise else 0.0
  env.inclinacao_severidade = (np.random.random() - 0.5) * 0.2 if add_noise else 0.0
  env.desvio_isolamento = np.random.random() if add_noise else 0.0 # incerteza na efetividade do isolamento
  env.tempo_medio_entre_contagios = sim_params.transmission_scale_days + (
      (np.random.random() - 0.5) * 0.1 if add_noise else 0.0) 
  env.fator_isolamento = 0.0
  env.scaling = scaling
  env.atencao = simpy.resources.resource.PriorityResource(env, capacity=int(sim_params.capacity_hospital_max * scaling))
  env.leito = simpy.resources.resource.PriorityResource(env, capacity=int(sim_params.capacity_hospital_beds * scaling))
  env.ventilacao = simpy.resources.resource.PriorityResource(env, capacity=int(sim_params.capacity_ventilators * scaling))
  env.uti = simpy.resources.resource.PriorityResource(env, capacity=int(sim_params.capacity_intensive_care * scaling))
  env.process(laboratorio(env))
  env.populacoes = cria_populacoes(env)
  env.stats = get_matriz_estatisticas(env, duracao)
  env.pessoas = []
  for pessoas in env.populacoes.values():
    env.pessoas.extend(pessoas)
  env.process(monitorar_populacao(env))
  for dia_inicio, fator_isolamento in isolamentos:
    env.process(aplica_isolamento(env, dia_inicio, fator_isolamento))
  env.run(until=duracao)
  env.run(until=duracao+env.d0+0.011)
  return env.stats / env.scaling


def run_simulations(
        sim_params: Parameters, 
        isolamentos: List[Tuple[float, float]], 
        simula_capacidade=False, 
        duracao: int=80, 
        n: int=4, # For final presentation purposes, a value greater than 10 is recommended 
        tamanho_simulacao: int=100000,  # For final presentation purposes, a value greater than 500000 is recommended 
        nome_simulacao=None,
        add_noise=True,  # Simulate uncertainty about main parameters and constants
):
    params = [tamanho_simulacao, duracao, isolamentos, simula_capacidade, sim_params, add_noise]
    try:
        pool = Pool(min(cpu_count(), n))
        all_stats = pool.map(simula, [params + [i] for i in range(n)])
    finally:
        pool.close()
        pool.join()
    stats = combina_stats(all_stats, sim_params)
    if nome_simulacao:
        stats.save(nome_simulacao)
    return stats


def combina_stats(all_stats: List[np.ndarray], sim_params: Parameters):
  mstats = np.stack(all_stats)
  population_names = list(sim_params.population_segments.keys())
  return Stats(mstats, cs.MEASUREMENTS, METRICS, population_names, cs.age_str, start_date=sim_params.start_date)
