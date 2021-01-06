---
layout: post
author: Jacob Pettit
comments: true
tags: reinforcement-learning code
title: Why does PPO work so well?
---

**[Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) is the state-of-the-art for on-policy policy gradient (PG) RL algorithms. But why does it work better than other on-policy PG algorithms, like REINFORCE and A2C? In this post, I'm going to look at a couple of components of PPO that seem like likely candidates for its superior performance.**

{: class="table-of-content"}
* TOC
{:toc}

# The Goal

Specifically, I want to compare PPO to REINFORCE and A2C. The reasons for choosing these algorithms are that REINFORCE is the first on-policy policy gradient (PG) algorithm that I'm aware of, so it is a good baseline to compare to; to my knowledge, A2C was the next big step in on-policy PG work and was state-of-the-art (SOTA) for a bit, it seems to me like a good midway step between REINFORCE and PPO; PPO is the SOTA (as far as I know) in pure on-policy PG algorithms. 

I've decided to skip TRPO since it's way more complicated to implement than these other algorithms and it was superseded by PPO shortly after it came out. TRPO's paper was published in 2015 and PPO was published in 2017.

So, I'm curious about why PPO works better than these other algorithms. It's easy to see why A2C and PPO work better than REINFORCE, since both of the latter methods make use of value function baselines and advantage estimation. Maybe it's better to say that I want to pick apart PPO and see what components of the algorithm contribute most to its performance.

As a disclaimer, I'm going to try to be a good scholar and reference earlier work, but this is also a blog post so I'm not going to do as thorough of a literature review as I would for a paper. So if I'm wrong about something or didn't link to earlier work somewhere that I should've, feel free to call me out.

# Experiment Plan

I've decided there are a couple of ablations to run. 
1. PPO allows multiple steps over a batch of data by using a KL constraint on the policy. Remove this multiple steps and test performance.
2. Keep the multiple steps but remove the KL constraint.
3. Add the KL constraint to REINFORCE and A2C.
4. Remove the value function from PPO.

Running this set of experiments should let me test several different things.
1. Doing number 1 above tests how much better the PPO loss function is than the REINFORCE and A2C loss functions.
2. This should probe a (anticipated) weakness of the PPO loss function. Without hard KL constraints, the policy updates might get catastrophically large and lead to worse performance.
3. Again, this is a more apples-to-apples comparison of quality of PPO loss function vs. REINFORCE and A2C loss.
4. This might be a really stupid experiment, since it really only compares PPO directly to REINFORCE, but maybe it's worth a shot.

## Hypotheses

And this is my set of hypotheses.
1. PPO without KL-constrained passes over the data will be better than A2C and REINFORCE, but not by much. 
2. PPO without the KL constraint and still doing multiple steps over the data will be better than A2C and REINFORCE on easy environments, worse on hard ones.
3. PPO will still be better, but adding KL-constrained steps to A2C and REINFORCE will greatly improve their performance and sample-efficiency.
4. PPO loss will do slightly better than REINFORCE loss.

## Environments to use

I think 4 environments, each of varying difficulty, should be sufficient. I'm going to stick to continuous control problems, and I'm going to use all environments from [PyBullet](https://pybullet.org/wordpress). The four I've picked are:
1. The continuous inverted pendulum. This is the easy one. It's pretty much just a continuous version of [CartPole](https://gym.openai.com/envs/CartPole-v0/) and it looks kind of [like this](https://gym.openai.com/envs/InvertedPendulum-v2/).
2. The HalfCheetah environment. This environment represents the moderate-difficulty case. It's an open-source version of the [MuJoCo HalfCheetah](https://gym.openai.com/envs/HalfCheetah-v2/).
3. The Ant environment. This one represents the moderately-hard-difficulty case. This is also an open-source version of the [MuJoCo Ant](https://gym.openai.com/envs/Ant-v2/).
4. The Humanoid environment. This environment will be the hard-difficulty case. Again, it's an open-source version of the [MuJoCo Humanoid](https://gym.openai.com/envs/Humanoid-v2/).

## Computational Constraints

For this project, I've got limited compute at my disposal. Especially because I don't want to spend money on cloud compute. So, I'm going to aim to run everything on my laptop (a 2020 MacBook Pro 13 inch) and I'm going to limit each run in an environment to 1 million timesteps. I'll likely do 3 random seeds per environment, and if I use more than that I'll specify it. There will be no random seed tuning. I won't perform any hyperparameter tuning either, and instead will generally use hyperparameters specified in academic papers. Should those hyperparameters be unavailable (e.g. I'm not sure if I'll find hyperparameters for REINFORCE) then I'll estimate best practice from my own experience and from other open-source implementations. Of course, all of the code I write and run will be open-sourced.

