from queue import PriorityQueue

import numpy as np

CAPACIDADE_INICIAL = 200
FATOR_DE_TAMANHO_MAXIMO_DA_FILA = 5

DEMANDA_EXAMES_INFLUENZA = 100
CHANCE_EXAME_INFLUENZA_PRIORITARIO = 0.3


def demanda_exames_influenza(env):
    while True:
        yield env.timeout(1.0 / (DEMANDA_EXAMES_INFLUENZA * env.scaling))
        prioridade = 0.0 if np.random.random() < CHANCE_EXAME_INFLUENZA_PRIORITARIO else 1.0
        env.solicitar_exame(prioridade, None)


def laboratorio(env):
    env.lab_queue = PriorityQueue()
    env.lab_capacity = CAPACIDADE_INICIAL
    env.exames_solicitados = 0
    env.exames_positivos = 0

    def solicitar_exame(prioridade, pessoa=None):
        env.exames_solicitados += 1
        prioridade = prioridade + np.random.random() / 2.0
        if env.lab_queue.qsize() > (env.lab_capacity) * (FATOR_DE_TAMANHO_MAXIMO_DA_FILA - prioridade):
            return  # Desistencia do exame
        env.lab_queue.put((prioridade, pessoa))

    env.solicitar_exame = solicitar_exame

    env.process(demanda_exames_influenza(env))

    while True:
        capacity = env.lab_capacity * env.scaling
        yield env.timeout(1.0 / env.lab_capacity)
        if env.lab_queue.empty():
            continue
        _, test = env.lab_queue.get()
        if test:
            test.diagnosticado = True
            test.data_diagnostico = env.now
            env.exames_positivos += 1
