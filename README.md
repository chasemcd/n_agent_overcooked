# N Agent Overcooked


Repository to set up and run Overcooked with an arbitrary number of agents and arbitrary layout design using [CoGrid](https://cogrid.readthedocs.io). CoGrid environments follow the PettingZoo `ParallelEnv` API.


## Getting Started

### Install Dependencies

The primary dependency is `cogrid`, our library for creating multi-agent grid-based environments. We've included this in the `requirements.txt` file.

```bash
$ conda create -n n_agent_overcooked python=3.10
$ pip install -r requirements.txt
```


### Constructing the environment

CoGrid is based on a "registry" system, so everything
that could be part of an environment is defined as an object and registered in the CoGrid ecosystem:

- Objects: All "objects" that appear in the environment (a `GridObject`), are registered and have associated names and ASCII character representations. For Overcooked, this includes Pots, Onions, Delivery Zones,
Counters, etc. 
- Observation Spaces/Features: We also define the feature space through individually registered `Feature` classes, which have 
associated generator functions to build observations for each agent. Importantly, CoGrid observation spaces are `Dict` spaces, so we can define a feature space for each agent and track them by name. The `observations`output of `env.step()` is a `Dict` of `{agent_id: {"feature_name": feature_value}}`.
- Rewards: We also define individual `Reward` classes, which have associated generator functions to build rewards for each agent. Overcooked, by default, just uses a `delivery_reward` which is the typical common reward of +1 for all agents when a dish is delivered. In the original paper, they also use plating rewards and onion-in-pot rewards for reward shaping, which can be included here (see `run_env.py`). 
- Layout: We also define the layout of the environment, which is a constant layout that will be used to instantiate the environment from our registered ASCII layout. You could alternatively usa a "layout_fn", which could generate a layout dynamically.


All of the above are done already in the existing Overcooked implementation in CoGrid and we've demonstrated feature space construction in `overcooked_features.py`. Everything is brought together to construct an environment in `run_env.py`, which will display the environment shown below:

![4-agent Overcooked environment](assets/env_image.png)
