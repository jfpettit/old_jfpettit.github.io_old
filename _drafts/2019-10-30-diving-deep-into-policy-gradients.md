---
layout: post
author: Jacob Pettit
comments: true
tags: long-post reinforcement-learning
---

In this post, we'll do a deep dive into the policy gradient theorem and the REINFORCE policy gradient algorithm. An implementation of the algorithm will also be shared in a [Colab](https://colab.research.google.com/) notebook and parts of it will be discussed here. This will be post 1 of $$N$$, where I do a deep dive into some theory behind an RL algorithm.

# What are policy gradients

Policy gradients are a class of methods used to tackle RL problems. They work by finding a function which determines what actions to take at each step in the RL environment.
