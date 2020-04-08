import numpy as np
from queue import PriorityQueue


INITIAL_CAPACITY = 200
MAXIMUM_QUEUE_SIZE_FACTOR = 5

INFLUENZA_EXAM_DEMAND = 100
PRIORITY_INFLUENZA_EXAM_ODDS = 0.3


def influenza_exams_demand(env):
    while True:
        yield env.timeout(1.0 / (INFLUENZA_EXAM_DEMAND * env.scaling))
        priority = 0.0 if np.random.random() < PRIORITY_INFLUENZA_EXAM_ODDS else 1.0
        env.request_exam(priority, None)


def laboratory(env):
    env.lab_queue = PriorityQueue()
    env.lab_capacity = INITIAL_CAPACITY
    env.requested_exams = 0
    env.positive_exams = 0

    def request_exam(priority, person=None):
        env.requested_exams += 1
        priority = priority + np.random.random() / 2.0
        if env.lab_queue.qsize() > (env.lab_capacity) * (MAXIMUM_QUEUE_SIZE_FACTOR - priority):
            return  # Give up on the exam
        env.lab_queue.put((priority, person))

    env.request_exam = request_exam

    env.process(influenza_exams_demand(env))

    while True:
        capacity = env.lab_capacity * env.scaling
        yield env.timeout(1.0 / env.lab_capacity)
        if env.lab_queue.empty():
            continue
        _, test = env.lab_queue.get()
        if test:
            test.diagnosed = True
            test.diagnostic_date = env.now
            env.positive_exams += 1
