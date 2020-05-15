# covidsimulation
Covid Epidemics Simulation - Individual-based dynamic model in python and cython with Simpy

_"All models are wrong, but some are useful." __George Box___

# About the model

This is a detailed individual-based simulation of the Covid-19 epidemic within a region. It is possible to model different age-groups and different sub-populations, that behave differently with regards to transmission and interventions. Different interventions, or sequences of interventions, can be simulated as well. Most parameters can be set as a random value within a distribution, allowing to model our current uncertainty. A calibration process is provided, in which parameters or random distributions can be optimized, effectively updating our belief about them. It is possible to plot curves for many output variables of the simulation, like expected official death counts, expected real death counts, intensive-care usage, reproductive number, etc.

The model mainly is intended for:
- providing a reallistic uncertainty range about the development of the epidemic
- studying the possible effects of interventions
- providing insights about what factors can be contributing to the overall results observed.

An example of how to do a basic simulation can be found in https://towardsdatascience.com/model-the-covid-19-epidemic-in-detail-with-python-98f0d13f3a0e

# Instructions
1. Clone this repo
1. Run `setup.sh`  (It might require making it executable: `chmod 777 setup.sh`)
1. Check `Example.ipynb` to understand how it works
1. Setup your own regions and modify simulation parameters and constants
1. Whenever you change Cython code, don't forget to compile it with 
`python3 setup.py build_ext --inplace` or simply call `setup.sh` again

# Troubleshooting
In case you have issues with clogged outputs or in case graphs don't show up, you can try
enabling some nbextensions in your jupyter:
_(If you are not in a virtualenv, use `--user` instead of `--sys-prefix`)_
```
jupyter nbextension install widgetsnbextension --sys-prefix --py
jupyter nbextension install plotlywidget --sys-prefix --py
jupyter nbextension enable widgetsnbextension --sys-prefix --py
jupyter nbextension enable plotlywidget --sys-prefix --py
```


# Understanding the code

This is an IBDM for Covid-19.
The model is based on the concept of `Person`. Each `Person`has several state variables
indicating her health, home location, age, etc, and also indicating the progression of
the disease. All happening with uncertainty. Statistics about the whole population are
gathered at the end of each day and stored as `measurements`.

## Simpy
The simulation happens using `Simpy`'s processes and resources. When `env.process(do_something())`
is called, `do_something()` is executed assynchronously (in simulation time). Within `do_something()`,
executing runs until being blocked by `yield env.timeout(duration)`, and resumes after `duration` 
days. `PriorityResource` manages capacity and a priority queue to use it.

## Geosocial model
We expect a `Person` to have a higher chance of interaction with people who live close to them. Also,
we expect most people will interact more often with other people from the same social group, and even
age group. At the moment, this was achieved approximating those interactions as sampling a few random
individuals, and then choosing the one who is the closest in a multidimensional space. This 
simplification can be replaced by a more accurate model in the future.

## Cython
To achieve a good resolution, a few hundreds of thousands of `Person`s need to be simulated. Using
pure Python would be slow and memory bound. By using Cython to define `Person` objects, speed was
significantly improved, and it is possible to simulate 1M people with less than 1GB RAM per core.

# ToDo's
1. Better document model setup, training and usage.
1. Model more regions, from different countries.
1. Model more possible interventions (e.g. intermitent social distancing, rotational work schedules, etc.)
1. Improve the probabilistic (calibration) model, modeling the information gain from different checks, and enabling transfer of believes between regions.
1. Model hospital transmission to health workers and to other patients.
1. Improve model of interaction between people:
  - Specifying different types of interaction between people
  - Creating sub-population and groups within which interactions can occur (e.g. school class)
  - Specifying different rates of social interaction for each type and age group
...
