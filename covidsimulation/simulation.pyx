# distutils: language = c++
#cython: language_level=3

# Copyright 2020 André Arroyo and contributors
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
# and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
# and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or
# promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from libcpp cimport bool, float, int
import numpy as np
cimport numpy as np
from libc.math cimport log as clog, exp as cexp
from libc.stdlib cimport rand, RAND_MAX, srand
from libcpp.vector cimport vector
import simpy


cpdef enum Outcome:
    NO_INFECTION = 0
    NO_SYMPTOMS = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
    INTENSIVE_CARE = 5
    VENTILATION = 6
    DEATH = 7


cpdef enum Age:
    A0_9 = 0
    A10_19 = 1
    A20_29 = 2
    A30_39 = 3
    A40_49 = 4
    A50_59 = 5
    A60_69 = 6
    A70_79 = 7
    A80plus = 8


age_str = [
    '0-9',
    '10-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80+',
]


cdef extern from *:
    """
    /* This is C code which will be put
     * in the .c file output by Cython */
    long c_mod(long x, int m) {return x % m;}
    """
    long c_mod(long x, int m)



cdef int get_outcome(np.ndarray severity):
    cdef double p = (rand() / (RAND_MAX + 1.0))
    if p > severity[0]:
        return Outcome.NO_INFECTION
    elif p > severity[1]:
        return Outcome.NO_SYMPTOMS
    elif p > severity[2]:
        return Outcome.MILD
    elif p > severity[3]:
        return Outcome.MODERATE
    elif p > severity[4]:
        return Outcome.SEVERE
    elif p > severity[5]:
        return Outcome.INTENSIVE_CARE
    elif p > severity[6]:
        return Outcome.VENTILATION
    return Outcome.DEATH

LOCALITY = 60  # The bigger this constant, the higher will be the probability of finding someone geographically closer


def seed(unsigned i):
    """Initialize C's random number generator
    """
    srand(i)


cpdef double logit_transform_value(double p, double adjustment_logit):
    """Take a uniform-distributed value p in ]0.0, 1.0[, and move it in the logistic distribution curve by adjustment_logit 
    """
    cdef double odds = p / (1.0 - p)
    cdef float logit = clog(odds)
    cdef float corrected_logit = logit - adjustment_logit
    cdef double corrected_odds = cexp(corrected_logit)
    cdef double corrected_p = corrected_odds / (1.0 + corrected_odds)
    return corrected_p


cdef double get_uniform():
    return (rand() + 1.0) / (RAND_MAX + 2.0)


cdef double sample_from_logit_uniform(double adjustment_logit):
    cdef double p = 1.0
    while not (0.0 < p < 1.0):
        p = get_uniform()
    return logit_transform_value(p, adjustment_logit)


def choice(arr, sample_size):
    return _choice(arr, sample_size)


def p_choice(np.ndarray p):
    """
    Return a random index from an array of probability weights. Much faster than np.random.choice.
    p: numpy.ndarray - Probabilities of each class. Must sum to 1.0.
    """
    cdef double total = 0
    cdef double luck = (rand()) / (RAND_MAX + 1.0)
    for i in range(len(p)):
        total += p[i]
        if luck < total:
            return i
    return len(p) - 1


cdef sample_indices_with_replacement(size_t population_len, size_t sample_size):
    cdef vector[size_t] indices
    indices.reserve(sample_size)
    for _ in range(sample_size):
        indices.push_back(<size_t>(rand() / (RAND_MAX  + 1.0) * population_len))
    return indices


cdef _choice(list arr, size_t sample_size):
    return [arr[i] for i in sample_indices_with_replacement
(len(arr), sample_size)]


cdef choose_contact_on_street(object person, object people, size_t sample_size=LOCALITY):
    sample = _choice(people, sample_size)  
    distances = np.zeros([sample_size])
    cdef float px = person.home.coords[0]
    cdef float py = person.home.coords[1]
    cdef float pz = person.home.coords[2]
    cdef float index = person.age_group.index
    cdef float[3] icoords
    for i, individual in enumerate(sample):
        icoords = individual.home.coords
        distances[i] = (px - icoords[0]) ** 2 + (py - icoords[1]) ** 2 + (pz - icoords[2]) ** 2
    chosen = sample[np.argmin(distances)]
    return chosen


cdef choose_contact_from_social_group(object person, object people, size_t sample_size=LOCALITY*2):
    sample = _choice(people, sample_size)  
    distances = np.zeros([sample_size])
    cdef float px = person.home.coords[0]
    cdef float py = person.home.coords[1]
    cdef float pz = person.home.coords[2]
    cdef float index = person.age_group.index
    cdef bint find_non_isolated_person = c_mod(rand(), 2)
    cdef float[3] icoords
    for i, individual in enumerate(sample):
        icoords = individual.home.coords
        distances[i] = (px - icoords[0]) ** 2 + (py - icoords[1]) ** 2 + (pz - icoords[2]) ** 2
        if individual.age_group.index != index:
            distances[i] += 1.0
        if find_non_isolated_person and individual.in_isolation:
            distances[i] += 1.0            
    chosen = sample[np.argmin(distances)]
    return chosen


###
## SimulationConstants and defaults
###

cdef class SimulationConstants:
    cdef public float home_contamination_daily_probability
    cdef public float survival_probability_in_severe_overcapacity
    cdef public float survival_probability_without_hospital_bed
    cdef public float survival_probability_without_intensive_care_bed
    cdef public float survival_probability_without_ventilator
    cdef public float symptoms_delay_shape
    cdef public float symptoms_delay_scale
    cdef public float incubation_to_symptoms_variable_fraction
    cdef public float contagion_duration_shape
    cdef public float contagion_duration_scale
    cdef public float time_to_outcome_severe_scale
    cdef public float time_to_outcome_severe_shape
    cdef public float time_to_hospitalization_severe_proportion
    cdef public float immunization_period
    cdef public float individual_daily_influence_in_social_distancing
    cdef public float death_confirmation_delay

    def __cinit__(self, *args, **kwargs):  # DEFAULT SIMULATION PARAMENTERS are set here
        self.home_contamination_daily_probability = 0.3
        self.survival_probability_in_severe_overcapacity = 0.3
        self.survival_probability_without_hospital_bed = 0.9
        self.survival_probability_without_intensive_care_bed = 0.8
        self.survival_probability_without_ventilator = 0.1
        self.symptoms_delay_shape = 4.0
        self.symptoms_delay_scale = 6.0
        self.incubation_to_symptoms_variable_fraction = 0.3
        self.contagion_duration_shape = 2.0
        self.contagion_duration_scale = 4.0
        self.time_to_outcome_severe_scale = 12.0
        self.time_to_outcome_severe_shape = 2.0
        self.time_to_hospitalization_severe_proportion = 0.5
        self.immunization_period = 0.0  # Mean immunization duration. Permanent if 0.0
        self.individual_daily_influence_in_social_distancing = 0.2
        self.death_confirmation_delay = 1.5

    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError('SimulationConstants does not accept positional arguments')
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        vals = ', '.join(['%s=%s' % (at, round(getattr(self, at), 2)) for at in dir(self) if not at.startswith('_') ])
        return 'SimulationConstants(%s)' % vals

    def __reduce__(self):
        return (SimulationConstants, tuple(), self.__getstate__())

    def __getstate__(self):
        return {at: round(getattr(self, at), 2) for at in dir(self) if not at.startswith('_')}

    def __setstate__(self, state):
        for at, value in state.items():
            setattr(self, at, value)

    def __hash__(self):
        return hash(self.__repr__())


####
## Home - a place with geometric coordinates where people live
####

cdef class Home:
    cdef public float isolation_propensity
    cdef public list residents
    cdef public float[3] coords
    
    def __cinit__(self, float displacement):
        cdef float z = np.random.uniform(-1.0, 1.0)
        self.coords[2] = z
        cdef float theta = np.random.uniform(0.0, 2 * np.pi)
        cdef diameter = np.sqrt(1.0 - z**2) 
        self.coords[0] = diameter * np.sin(theta) + displacement
        self.coords[1] = diameter * np.cos(theta)
        self.residents = []

    def add_person(self, Person person):
        self.residents.append(person)
        person.home = self
        isolation_adherence = np.array([p.age_group.isolation_adherence for p in self.residents]).mean()
        self.isolation_propensity = sample_from_logit_uniform(isolation_adherence)

####
## Person - an individual being simulated, with all characteristics about the disease
####


cdef class Person:
    cdef object senv  # SimulationEnvironment
    cdef object env  # simpy Environment
    cdef object process  # simpy async process
    cdef object timeout  # simpy environment timeout
    cdef public object age_group
    cdef Py_ssize_t age_group_index
    cdef SimulationConstants sim_consts
    cdef size_t expected_outcome
    cdef public Home home

    cdef public bool susceptible
    cdef public bool infected
    cdef public bool hospitalized
    cdef public bool recovered
    cdef public bool in_incubation
    cdef public bool contagious
    cdef public bool in_isolation
    cdef public bool dead
    cdef public bool confirmed_death
    cdef public bool active
    cdef public bool diagnosed
    cdef public float infection_date
    cdef public float diagnosis_date
    cdef public float hospitalization_date
    cdef public float death_date
    cdef public float recovery_date
    cdef public float masks_usage
    cdef public float hygiene_adoption
    cdef public size_t transmited
    cdef public bool in_hospital_bed
    cdef public bool in_icu
    cdef public bool in_ventilator
    cdef public bool avoidable_death
    cdef float time_until_symptoms
    cdef float incubation_time
    cdef object hospital_bed_req
    cdef object attention_req
    cdef object icu_req
    cdef object ventilator_req

    def __cinit__(self, object senv, object age_group, Home home):
        self.senv = senv
        self.env = senv.env
        self.timeout = senv.env.timeout
        self.process = senv.env.process
        self.sim_consts = senv.sim_params.constants
        self.susceptible = True 
        self.infected = False
        self.hospitalized = False
        self.recovered = False
        self.in_incubation = False
        self.contagious = False
        self.in_isolation = False
        self.dead = False
        self.confirmed_death = False
        self.active = False  # infected that neither died nor recovered yet
        self.diagnosed = False
        self.infection_date = 0.0
        self.diagnosis_date = 0.0
        self.hospitalization_date = 0.0
        self.death_date = 0.0
        self.recovery_date = 0.0
        self.masks_usage = 0.0
        self.hygiene_adoption = 0.0
        self.transmited = 0
        self.in_hospital_bed = False
        self.in_icu = False
        self.in_ventilator = False
        self.avoidable_death = False
        self.attention_req = None
        self.hospital_bed_req = None
        self.icu_req = None
        self.ventilator_req = None
        self.age_group = age_group
        self.age_group_index = age_group.index
        self.expected_outcome = 0
        home.add_person(self)

    cdef void calculate_case_params(self):
        self.time_until_symptoms = np.random.weibull(
            self.sim_consts.symptoms_delay_shape
            ) * self.sim_consts.symptoms_delay_scale
        symptoms_delay = np.random.random()
        self.incubation_time = self.time_until_symptoms * (
            1.0 - symptoms_delay * self.sim_consts.incubation_to_symptoms_variable_fraction)

    def expose_to_virus(self):
        if not self.susceptible:
            return False
        self.susceptible = False
        self.infection_date = self.env.now
        cdef size_t expected_outcome = get_outcome(self.age_group.severity)
        if expected_outcome == Outcome.NO_INFECTION:
            return False
        else:
            self.expected_outcome = expected_outcome
        self.calculate_case_params()
        self.infected = True
        self.active = True
        self.process(self.run_contagion())
        return True

    def run_contagion(self):
        self.in_incubation = True
        yield self.timeout(self.incubation_time)
        self.in_incubation = False
        self.contagious = True
        self.process(self.run_contagion_home())
        self.process(self.run_contagion_street())
        self.process(self.run_contagion_social_group())
        contagion_duration = np.random.weibull(
            self.sim_consts.contagion_duration_shape) * self.sim_consts.contagion_duration_scale
        self.configure_evolution()
        yield self.timeout(contagion_duration)
        self.contagious = False

    cdef void configure_evolution(self):
        if self.expected_outcome == Outcome.DEATH:
            self.configure_evolution_death()
        elif self.expected_outcome == Outcome.VENTILATION:
            self.configure_evolution_ventilation()
        elif self.expected_outcome == Outcome.INTENSIVE_CARE:
            self.configure_evolution_icu()
        elif self.expected_outcome == Outcome.SEVERE:
            self.configure_evolution_hospitalization()
        elif self.expected_outcome == Outcome.MODERATE:
            self.configure_evolution_moderate_at_home()
        else:
            self.configure_evolution_mild_at_home()

    cdef void configure_evolution_death(self):
        time_until_outcome = np.random.weibull(2) * 17  # 15 dias
        time_until_hospitalization = time_until_outcome * 0.4  # 6 dias
        self.process(self.run_hospitalization(time_until_hospitalization))
        time_until_icu_and_ventilation = time_until_outcome * 0.47  # 8 dias na ventilacao
        self.process(self.run_ventilation(time_until_icu_and_ventilation))
        self.process(self.run_death(time_until_outcome))

    cdef void configure_evolution_ventilation(self):
        time_until_outcome = np.random.weibull(2) * 36  # 32 dias
        time_until_hospitalization = time_until_outcome * 0.2  # 6 dias
        self.process(self.run_hospitalization(time_until_hospitalization))
        time_until_icu_and_ventilation = time_until_outcome * 0.2627  # 8 dias
        self.process(self.run_ventilation(time_until_icu_and_ventilation))
        time_until_icu_without_ventilation = time_until_outcome * 0.5113  # 16 dias
        self.process(self.run_leave_ventilation(time_until_icu_without_ventilation))
        time_until_icu_ends = time_until_outcome * 0.7125  # 22 dias
        self.process(self.run_leave_icu(time_until_icu_ends))
        self.process(self.run_leave_hospital(time_until_outcome))

    cdef void configure_evolution_icu(self):
        time_until_outcome = np.random.weibull(2) * 23  # 20 dias
        time_until_hospitalization = time_until_outcome * 0.3  # 6 dias
        self.process(self.run_hospitalization(time_until_hospitalization))
        time_until_icu = time_until_outcome * 0.4  # 8 dias
        self.process(self.run_enter_icu(time_until_icu))
        time_until_icu_ends = time_until_outcome * 0.7
        self.process(self.run_leave_icu(time_until_icu_ends))
        self.process(self.run_leave_hospital(time_until_outcome))

    cdef void configure_evolution_hospitalization(self):
        time_until_outcome = np.random.weibull(
            self.sim_consts.time_to_outcome_severe_shape) * self.sim_consts.time_to_outcome_severe_scale
        time_until_hospitalization = time_until_outcome * self.sim_consts.time_to_hospitalization_severe_proportion  # 6 dias
        self.process(self.run_hospitalization(time_until_hospitalization))
        self.process(self.run_leave_hospital(time_until_outcome))

    cdef void configure_evolution_moderate_at_home(self):
        time_until_outcome = np.random.weibull(2) * 20  # 18 dias  
        self.process(self.run_cure(time_until_outcome))
        if self.age_group.chance_of_diagnosis_if_moderate > get_uniform():
            self.request_diagnosis()

    cdef void configure_evolution_mild_at_home(self):
        time_until_outcome = np.random.weibull(2) * 15  # 18 dias  
        self.process(self.run_cure(time_until_outcome))
    
    cdef void request_diagnosis(self):
        diagnosis_delay = self.age_group.diagnosis_delay
        if diagnosis_delay is None:
            self.senv.lab.request_exam(1, self)
        else:
            self.process(self.wait_for_diagnosis(diagnosis_delay))

    def wait_for_diagnosis(self, float diagnosis_delay):
        cdef float time_for_diagnosis = np.random.weibull(4.0) * diagnosis_delay
        yield self.timeout(diagnosis_delay)
        self.diagnosed = True
        if self.dead:
            self.process(self.wait_for_death_confirmation())

    def wait_for_death_confirmation(self):
        cdef float time_for_confirmation = np.random.weibull(4.0) * self.sim_consts.death_confirmation_delay
        yield self.timeout(time_for_confirmation)
        self.confirmed_death = True

    def run_hospitalization(self, time_until_hospitalization):
        yield self.timeout(time_until_hospitalization)
        if self.dead:
            return
        if self.senv.simulate_capacity:
            self.process(self.request_attention())
            yield from self.request_hospital_bed()
        else:
            self.hospitalized = True
            self.in_hospital_bed = True
            self.hospitalization_date = self.env.now
            self.request_diagnosis()
  
    def request_attention(self):
        attention_req = self.senv.attention.request(Outcome.DEATH - self.expected_outcome)
        request = attention_req.__enter__()
        self.attention_req = attention_req
        result = yield request | self.timeout(np.random.exponential(2.0))
        if request in result:
            if self.attention_req:
                self.hospitalized = True
                self.hospitalization_date = self.env.now
            self.request_diagnosis()
        else:
            if self.attention_req:
                attention_req.__exit__(None, None, None)
                self.attention_req = None
            if self.expected_outcome == Outcome.SEVERE and np.random.random() < self.sim_consts.survival_probability_in_severe_overcapacity:
                return  # Luck - patient recovered by itself
            self.avoidable_death = True
            yield from self.run_death(0)


    def request_hospital_bed(self):
        hospital_bed_req = self.senv.hospital_bed.request(8 - self.expected_outcome)
        request = hospital_bed_req.__enter__()
        self.hospital_bed_req = hospital_bed_req
        result = yield request | self.timeout(np.random.exponential(2.0))
        if request in result:
            if self.hospital_bed_req:
                self.in_hospital_bed = True
        else:
            if self.hospital_bed_req:
                hospital_bed_req.__exit__(None, None, None)
                self.hospital_bed_req = None
            if self.expected_outcome == Outcome.SEVERE and np.random.random() < self.sim_consts.survival_probability_without_hospital_bed:
                return  # Sorte - o paciente se recuperou em home
            self.avoidable_death = True
            yield from self.run_death(0)

    def request_icu(self):
        icu_req = self.senv.icu.request(8 - self.expected_outcome)
        request = icu_req.__enter__()
        self.icu_req = icu_req
        result = yield request | self.timeout(np.random.exponential(1.0))
        if request in result:
            if self.icu_req:
                self.in_icu = True
        else:
            if self.icu_req:
                icu_req.__exit__(None, None, None)
                self.icu_req = None
            if np.random.random() < self.sim_consts.survival_probability_without_intensive_care_bed:
                return  # Luck - patient recovered by itself
            self.avoidable_death = True
            yield from self.run_death(0)

    def request_ventilator(self):
        ventilator_req = self.senv.ventilator.request(8 - self.expected_outcome)
        request = ventilator_req.__enter__()
        self.ventilator_req = ventilator_req
        result = yield request | self.timeout(np.random.exponential(1.0))
        if request in result:
            if self.ventilator_req:
                self.in_ventilator = True
        else:
            if self.ventilator_req:
                ventilator_req.__exit__(None, None, None)
                self.ventilator_req = None
            if np.random.random() < self.sim_consts.survival_probability_without_ventilator:
                return  # Luck - patient recovered by itself
            self.avoidable_death = True
            yield from self.run_death(0)

    def run_ventilation(self, tempo_ate_ventilacao):
        yield self.timeout(tempo_ate_ventilacao)
        if self.dead:
            return
        if self.senv.simulate_capacity:
            self.process(self.request_icu())
            self.process(self.request_ventilator())
        else:
            self.in_ventilator = True
            self.in_icu = True
    
    def run_leave_ventilation(self, time_until_icu_without_ventilation):
        yield self.timeout(time_until_icu_without_ventilation)
        if self.dead:
            return
        if self.ventilator_req:
            self.ventilator_req.__exit__(None, None, None)
            self.ventilator_req = None
        self.in_ventilator = False
    
    def run_leave_icu(self, time_until_icu_ends):
        yield self.timeout(time_until_icu_ends)
        if self.dead:
            return
        if self.icu_req:
            self.icu_req.__exit__(None, None, None)
            self.icu_req = None
        self.in_icu = False

    def run_enter_icu(self, tempo_ate_icu):
        yield self.timeout(tempo_ate_icu)
        if self.dead:
            return
        if self.senv.simulate_capacity:
            yield from self.request_icu()
        else:
            self.in_icu = True

    def run_leave_hospital(self, time_until_discharge_from_hospital):
        yield self.timeout(time_until_discharge_from_hospital)
        if self.dead:
            return
        self.active = False
        self.setup_remove_immunization()
        if self.hospital_bed_req:
            self.hospital_bed_req.__exit__(None, None, None)
            self.hospital_bed_req = None
        if self.attention_req:
            self.attention_req.__exit__(None, None, None)
            self.attention_req = None
        if self.hospitalized:    
            self.recovery_date = self.env.now
        self.hospitalized = False
        self.in_hospital_bed = False

    def run_cure(self, time_until_discharge_from_hospital):
        yield self.timeout(time_until_discharge_from_hospital)
        if self.dead:
            return
        self.active = False
        self.setup_remove_immunization()
    
    def run_death(self, time_until_death):
        yield self.timeout(time_until_death)
        if self.dead:
            return
        self.active = False 
        self.contagious = False
        if self.senv.simulate_capacity:
            if self.attention_req:
                self.attention_req.__exit__(None, None, None)
                self.attention_req = None        
            if self.hospital_bed_req:
                self.hospital_bed_req.__exit__(None, None, None)
                self.hospital_bed_req = None
            if self.icu_req:
                self.icu_req.__exit__(None, None, None)
                self.icu_req = None
            if self.ventilator_req:
                self.ventilator_req.__exit__(None, None, None)
                self.ventilator_req = None
        self.hospitalized = False
        self.in_hospital_bed = False
        self.in_icu = False
        self.in_ventilator = False
        self.susceptible = False
        self.dead = True
        self.death_date = self.env.now
        if self.diagnosed:
            self.process(self.wait_for_death_confirmation())
    
    def run_contagion_home(self):
        while self.contagious and not self.hospitalized:
            self.infect_in_home()
            yield self.timeout(1.0)

    cdef infect_in_home(self):
        for person in self.home.residents:
            if not (person is self):
                if get_uniform() < self.sim_consts.home_contamination_daily_probability:
                    transmited = person.expose_to_virus()
                    if transmited:
                        self.transmited += 1

    cdef bint test_isolation(self):
        """
        Test if a person's isolation can avoid a transmission and/or infection situation
        """
        cdef bint isolation
        if self.sim_consts.individual_daily_influence_in_social_distancing > get_uniform():
            isolation = sample_from_logit_uniform(self.age_group.isolation_adherence) < self.senv.isolation_factor
        else:
            isolation = self.in_isolation
        return isolation and get_uniform() < self.age_group.isolation_effectiveness

    cdef bint test_mask_transmission(self):
        """
        Test if a person's mask usage can prevent transmission
        """
        if self.masks_usage:
            if self.masks_usage > get_uniform():  # mask was being used
                if self.age_group.mask_transmission_reduction > get_uniform():  # mask was effective
                    return 0
        return 1

    cdef bint test_mask_infection(self):
        """
        Test if a person's mask usage can avoid infection
        """
        if self.masks_usage:
            if self.masks_usage > get_uniform():  # mask was being used
                if self.age_group.mask_infection_reduction > get_uniform():  # mask was effective
                    return 0
        return 1

    cdef bint test_hygiene_infection(self):
        """
        Test if a person's hygiene measures can avoid infection
        """
        if self.hygiene_adoption:
            if self.hygiene_adoption > get_uniform():  # hygiene was being practicised
                if self.age_group.hygiene_infection_reduction > get_uniform():  # hygiene was effective
                    return 0
        return 1

    cdef bint test_street_transmission(self):
        """
        Test if a person can transmit to others in the street, given person's containment measures
        """
        return (
            (not self.test_isolation()) 
            and self.test_mask_transmission()
        )

    cdef bint test_street_infection(self):
        """
        Test if a person can be infected in the street, given person's containment measures
        """
        return (
            self.susceptible 
            and (not self.test_isolation()) 
            and self.test_mask_infection()
            and self.test_hygiene_infection()
        )

    cdef bint test_social_group_transmission(self):
        """
        Test if a person can transmit to others in its social group, given person's containment measures
        """
        return (
            (not self.test_isolation())
            and self.test_mask_transmission()
        )

    cdef bint test_social_group_infection(self):
        """
        Test if a person can be infected in its social group, given person's containment measures
        """
        return (
            self.susceptible
            and (not self.test_isolation())
            and self.test_mask_infection()
            and self.test_hygiene_infection()
        )

    def run_contagion_street(self):
        cdef Person contact_on_street
        yield self.timeout(
            np.random.exponential(self.senv.street_expositions_interval)
            )
        while self.contagious and not self.hospitalized:
            if self.test_street_transmission():
                contact_on_street = choose_contact_on_street(self, self.senv.people)
                if contact_on_street.test_street_infection():
                    if contact_on_street.expose_to_virus():
                        self.transmited += 1
            yield self.timeout(
                np.random.exponential(self.senv.street_expositions_interval)
            )

    def run_contagion_social_group(self):
        cdef Person contact_on_group
        yield self.timeout(
            np.random.exponential(self.senv.social_group_expositions_interval)
            )
        while self.contagious and not self.hospitalized:
            if self.test_social_group_transmission():
                contact_on_group = choose_contact_from_social_group(self, self.senv.people)
                if contact_on_group.test_social_group_infection():
                    if contact_on_group.expose_to_virus():
                        self.transmited += 1
            yield self.timeout(
                np.random.exponential(self.senv.social_group_expositions_interval)
            )

    cdef void setup_remove_immunization(self):
        if self.sim_consts.immunization_period:
            self.process(self.run_remove_immunization())

    def run_remove_immunization(self):
        immunization_timeout = np.random.exponential(self.sim_consts.immunization_period)
        yield self.timeout(immunization_timeout)
        if not self.dead:
            self.susceptible = True
        

########
##  Logs
########

cdef int get_person(Person person):
    return 1

cdef int get_infected(Person person):
    return 1 if person.infected else 0

cdef int get_in_isolation(Person person):
    return 1 if person.in_isolation else 0

cdef int get_diagnosed(Person person):
    return 1 if person.diagnosed else 0

cdef int get_deaths(Person person):
    return 1 if person.dead else 0

cdef int get_confirmed_deaths(Person person):
    return 1 if person.confirmed_death else 0

cdef int get_hospitalized(Person person):
    return 1 if person.hospitalized else 0

cdef int get_in_ventilator(Person person):
    return 1 if person.in_ventilator else 0

cdef int get_in_icu(Person person):
    return 1 if person.in_icu else 0

cdef int get_in_hospital_bed(Person person):
    return 1 if person.in_hospital_bed else 0

cdef int get_in_ward_bed(Person person):
    return 1 if person.in_hospital_bed and not person.in_icu else 0

cdef int get_contagious(Person person):
    return 1 if person.contagious else 0

cdef int get_contagion_ended(Person person):
    return 1 if person.infected and not (person.contagious or person.in_incubation) else 0

cdef int get_rt(Person person):
    return person.transmited if person.infected and not (person.contagious or person.in_incubation) else 0

cdef int get_susceptible(Person person):
    return 1 if person.susceptible else 0

cdef int get_confirmed_inpatients(Person person):
    return 1 if person.hospitalized and person.diagnosed else 0

cdef int get_confirmed_in_icu(Person person):
    return 1 if person.in_icu and person.diagnosed else 0

cdef int get_confirmed_in_ward_bed(Person person):
    return 1 if person.in_hospital_bed and person.diagnosed and not person.in_icu else 0

ctypedef int (*int_from_person)(Person)

cdef enum:
    NUM_FUNCTIONS = 18

cdef int_from_person[NUM_FUNCTIONS] fmetrics

fmetrics[0] = &get_person
fmetrics[1] = &get_infected
fmetrics[2] = &get_in_isolation
fmetrics[3] = &get_diagnosed
fmetrics[4] = &get_deaths
fmetrics[5] = &get_confirmed_deaths
fmetrics[6] = &get_hospitalized
fmetrics[7] = &get_in_ventilator
fmetrics[8] = &get_in_icu
fmetrics[9] = &get_in_hospital_bed
fmetrics[10] = &get_contagious
fmetrics[11] = &get_contagion_ended
fmetrics[12] = &get_rt
fmetrics[13] = &get_susceptible
fmetrics[14] = &get_in_ward_bed
fmetrics[15] = &get_confirmed_in_icu
fmetrics[16] = &get_confirmed_inpatients
fmetrics[17] = &get_confirmed_in_ward_bed


MEASUREMENTS = [
    'population',
    'infected',
    'in_isolation',
    'diagnosed',
    'deaths',
    'confirmed_deaths',
    'inpatients',
    'ventilated',
    'in_intensive_care',
    'in_hospital_bed',
    'contagious',
    'contagion_ended',
    'transmitted',
    'susceptible',
    'in_ward_bed',
    'confirmed_in_intensive_care',
    'confirmed_inpatients',
    'confirmed_in_ward_bed',
]


assert len(MEASUREMENTS) == NUM_FUNCTIONS


cdef _log_stats(size_t day, np.ndarray stats, object populations):
    global fmetrics
    cdef Py_ssize_t stop = NUM_FUNCTIONS
    cdef Py_ssize_t age_index
    cdef Py_ssize_t i
    cdef int value
    cdef double[:, :, :, :] stats_view = stats
    cdef Py_ssize_t population_index
    cdef Py_ssize_t metric_index
    cdef Person person

    for population_index, people in enumerate(populations.values()):
        for person in people:
            age_index = person.age_group_index
            metric_index = 0
            while metric_index < stop:
                value = fmetrics[metric_index](person)
                if value:
                    stats_view[population_index, metric_index, age_index, day] += value
                metric_index += 1

                
def log_stats(senv):
    day = int(senv.env.now+0.1) - senv.d0
    _log_stats(day, senv.stats, senv.populations)
