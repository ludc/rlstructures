rlstructures
============

TL;DR
-----
`rlstructures` is a lightweight Python library that provides simple APIs as well as data structures that make as few assumptions as possible on the structure of your agent or your task while allowing the transparent execution of multiple policies on multiple environments in parallel (incl. multiple GPUs).

Important Note (Feb 2021)
-------------------------

Due to feedback, we have made changed over the API. The old API is still working, but we encourage you to move to the new one. The modifications are:

* There is now only one Batcher class (called `RL_Batcher`)

  * The format of the trajectories returned by the batcher is different (see the `Getting Started` section)
* The Agent API (`RL_Agent`) is different and simplified

  * We also include a `replay` function to facilitate loss computation
* The principles are exaclty the same, and adaptation is easy (and we can help !)
* The API will not change anymore during the next months.

Why/What?
---------
RL research addresses multiple aspects of RL like hierarchical policies, option-based policies, goal-oriented policies, structured input/output spaces, transformers-based policies, etc., and there are currently few tools to handle this diversity of research projects.

We propose `rlstructures` as a way to:

* Simulate multiple policies, multiple models and multiple environments simultaneously at scale

* Define complex loss functions

* Quickly implement various policy architectures.

The main RLStructures principle is that the users delegates the sampling of trajectories and episodes to the library so they can spend most of their time on the interesting part of RL research: developing new models and algorithms.

`rlstructures` is easy to use: it has very few simple interfaces that can be learned in one hour by reading the tutorials.

It comes with multiple RL algorithms as examples including A2C, PPO, DDQN and SAC.

Please reach out to us if you intend to use it. We will be happy to help, and potentially to implement missing functionalities.

Where?
------

* Github: http://github.com/facebookresearch/rlstructures
* Discussion Group: https://www.facebook.com/groups/834804787067021
* Medium posts/Tutorials: https://medium.com/@ludovic.den

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   overview
   gettingstarted/index
   algorithms/index
   api/index
   foireaq/foireaq.rst
   deprecated/index.rst
