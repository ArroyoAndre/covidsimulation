from dataclasses import dataclass
from queue import PriorityQueue

import numpy as np
import simpy

INITIAL_CAPACITY = 200
MAXIMUM_QUEUE_SIZE_FACTOR = 5

INFLUENZA_EXAM_DEMAND = 100
PRIORITY_INFLUENZA_EXAM_ODDS = 0.3


@dataclass
class Lab:
    queue: PriorityQueue
    capacity: float
    env: simpy.Environment
    scaling: float
    requested_exams: int = 0
    positive_exams: int = 0

    def request_exam(self, priority, person=None):
        self.requested_exams += 1
        priority = priority + np.random.random() / 2.0
        if self.queue.qsize() > (self.capacity) * (MAXIMUM_QUEUE_SIZE_FACTOR - priority):
            return  # Give up on the exam
        self.queue.put((priority, person))

    def run(self):
        while True:
            yield self.env.timeout(1.0 / self.capacity)
            if self.queue.empty():
                continue
            _, test = self.queue.get()
            if test:
                test.diagnosed = True
                test.diagnosis_date = self.env.now
                self.positive_exams += 1


def influenza_exams_demand(lab: Lab):
    while True:
        yield lab.env.timeout(1.0 / (INFLUENZA_EXAM_DEMAND * lab.scaling))
        priority = 0.0 if np.random.random() < PRIORITY_INFLUENZA_EXAM_ODDS else 1.0
        lab.request_exam(priority, None)


def laboratory(env: simpy.Environment, scaling: float) -> Lab:
    lab = Lab(
        queue=PriorityQueue(),
        capacity=INITIAL_CAPACITY,
        env=env,
        scaling=scaling,
    )

    env.process(influenza_exams_demand(lab))
    env.process(lab.run())
    return lab
