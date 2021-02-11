# IMPORTANT NOTE

**Due to feedback, the API of the batcher has been improved with the following consequences:**
* The EpisodeBatcher and Batcher, associated with the Agent classes are still working but will print a deprecated message
* The new Batcher is called RL_Batcher (see documentation) and associated with a RL_Agent with a simplified API
* Learning algorithms using the old version are now in 'rlalgos/deprecated' and the algorithms using the new API are located in 'rlalgos/'

What you gain by using the new API:
* The batcher returns a **Trajectories** object that contains both fixed information (as a DictTensor e.g agent_info, env_info, agent state at the beginning of the trajectories, ...) and sequence of transitions in a TemporalDictTensor
* We provide a *replay* function to replay an agent on trajectories, allowing a faster and simpler implementation of algorithms

All these changes are documented in the HTML documentation at [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures)

We consider that the new provided API is faster and simpler to use.

# TL;DR

TL;DR: RLStructures is a lightweight Python library that provides simple APIs as well as data structures that make as few assumptions as possible about the structure of your agent or your task, while allowing for transparently executing multiple policies on multiple environments in parallel (incl. multiple GPUs). It thus facilitates the implementation of RL algorithms while avoiding complex abstractions.

# Why/What?

RL research addresses multiple aspects of RL like hierarchical policies, option-based policies, goal-oriented policies, structured input/output spaces, transformer-based policies, etc. while many available tools are specific to particular settings.

We propose RLStructures as a way to i) simulate multiple policies, multiple models and multiple environments simultaneously at scale ii) define complex loss functions and iii) quickly implement various policy architectures.

The main principle of RLStructures is to allow the user to delegate the sampling of trajectories and episodes to the library so they can spend most of their time on the interesting part of RL research: developing new models and algorithms.

RLStructures is easy to use: it has very few simple interfaces that can be learned in one hour by reading the tutorials. It is not a RL Alorithms catalog, but a library to do RL Research. For illustration purposes it comes with multiple RL algorithms including A2C, PPO, DDQN and SAC.

## Installation

Install from source by running the following inside the repo:
```
pip install .
```

## Learning RLStructures

* [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures)

The complete documentation is available at [http://facebookresearch.github.io/rlstructures](http://facebookresearch.github.io/rlstructures). The example algorithms are provided in *raglos/*.

A facebook group is also open for discussion.

# List of Papers using rlstructures

* [Learning Adaptive Exploration Strategies in Dynamic Environments Through Informed Policy Regularization](https://arxiv.org/abs/2005.02934)
* More to come...


# Citing RLStructures

Please use this bibtex if you want to cite this repository in your publications:

```
    @misc{rlstructures,
        author = {L. Denoyer, D. Rothermel and X. Martinet},
        title = {{RLStructures - A simple library for RL research}},
        year = {2021},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://gitHub.com/facebookresearch/rlstructures}},
    }

```

* Author: Ludovic Denoyer
* Co-authors: Danielle Rothermel, Xavier Martinet
* Other contributors: many....

## License

`rlstructures` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
