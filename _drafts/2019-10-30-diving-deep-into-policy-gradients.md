---
layout: post
author: Jacob Pettit
comments: true
tags: long-post reinforcement-learning
---

**In this post, we'll do a deep dive into the policy gradient theorem and the REINFORCE policy gradient algorithm. I'll be assuming familiarity with at least the basics of RL. If you need a refresher, I have a blog post on this [here](https://jfpettit.github.io/blog/2019/11/03/fundamentals-of-reinforcement-learning). We'll cover the math behind REINFORCE and will write some code to train a REINFORCE agent.**

{: class="table-of-content"}
* TOC
{:toc}

# What are policy gradients

Policy gradients are a class of methods used to tackle RL problems. They work by finding a function determines what actions to take at each step in an RL environment. A learned policy gradient can be stochastic or deterministic.

## Stochastic vs. Deterministic policy gradients

A stochastic policy gradient learns a distribution over actions given a current state. This is written as:

$$ p(a | s) = 
