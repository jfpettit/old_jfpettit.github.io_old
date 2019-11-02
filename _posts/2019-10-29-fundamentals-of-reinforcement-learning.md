---
layout: post
author: Jacob Pettit
comments: true
title: Looking at the Fundamentals of Reinforcement Learning
tags: long-post reinforcement-learning
---

In this post, we'll get into the weeds with some of the fundamentals of reinforcement learning. Hopefully, this will serve as a quick overview of the basics for someone who is curious but not looking to invest tons of time into learning lots of background reinforcement learning theory.

{: class="table-of-content"}
* TOC
{:toc}

# Markov Decision Processes
In this section, I'll briefly cover some fundamentals of Markov Decision Processes (MDPs), and of reinforcement learning (RL). In RL, we want to solve an MDP by figuring out how to take actions that maximize the reward received. The actor in an MDP is called an agent. We want the agent to learn how to take optimal actions. Optimal actions are those that maximize the *expected* reward received. Since we can't see into the future, we have to maximize the expectation of future reward. Below we have a diagram of the classic MDP formulation of agent-environment interaction.

![mdp-rl-interaction-loop](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

At initialization, the environment outputs some state $$s_t$$ and reward $$r_t$$. The agent observes this state and in response takes an action, $$a_t$$. This action is then applied to the environment, the environment is stepped forward in response to the action taken, and it yields a new state, $$s_{t+1}$$ and reward signal $$r_{t+1}$$ to the agent. This loop continues until the episode (period of agent-environment interaction) terminates.

Some examples of a state might be a frame of an Atari game or the current layout of pieces in a board game such as chess. The reward is a scalar signal calculated following a reward function, and the action can be something like which piece to move where on a chess board, or which direction to go in an Atari game.

## Return

Return is the cumulative sum of all reward earned in an episode. Let's denote return with $$G_t$$. Then, the return of an episode is defined with:

$$ G_t \stackrel{.}{=} r_t + r_{t+1} + r_{t+2} + r_{t+3} + \dotsb + r_T $$

$$r_T$$ is the reward earned at the final step of an episode. We can rewrite the above expression more concisely:

$$ G_t \stackrel{.}{=} \sum_{t=0}^T r_t $$

In RL, we want to train the agent to maximize expected future return. If the agent always takes the action with the most expected future return, then it is behaving optimally.

Sometimes, we might want an agent to solve a task that doesn't easily break up into episodes. This is called a *continuing* task. The reward formulation above doesn't work for continuing tasks, because as the number of time steps goes to infinity, so does the reward. We need a formulation of reward that will converge to a value as $$t \rightarrow \infty$$.

*Note to self: really make sure to clearly explain gamma and discounted rewards, maybe even tie in GAE-lambda*

# Policies and Value functions

## Policies

The policy is a mapping from states to actions (or to a probability distribution over actions), and can be written like so:

$$\pi : s_t \mapsto a_t$$

Where $$\pi$$ represents the policy, $$s_t$$ represents the current state, and $$a_t$$ is the chosen action to take while in state $$s_t$$. The subscript $$t$$ denotes the state or action at time step $$t$$.

Policies can be stochastic or deterministic. In the case of a stochastic policy, the output would be a probability distribution over actions, with the action the policy believes to be optimal having the highest probability of occurring. In a deterministic policy, the output is directly what action to take, and this action is the one that the policy believes is optimal. These can be written mathematically:
- Stochastic policy: $$\pi(a \vert s) = P_{\pi} [A = a \vert S = s]$$
- Deterministic policy: $$\pi(s) = a$$

Let's break this down. In the stochastic policy, $$\pi(a \vert s)$$ is telling us that the output of the policy $$\pi$$ is conditioned on the state $$s$$. $$P_{\pi} [A = a \vert S = s]$$ says that the probability of the action $$a$$ being equal to $$A$$ depends on $$s$$ equaling $$S$$. The deterministic policy simply tells us that the policy $$\pi$$ takes in state $$s$$ and maps to an action $$a$$.

## Value functions

The value of a state is determined by how much reward is expected to follow it. If there's lots of reward expected after state $$s_t$$, then that must be a good state. But, if there is very little reward expected to follow state $$s_{t+5}$$, then that's a bad state to be in. The state-value function is a function that, given a state, outputs an estimate of the expected future return following that state. An action-value function will take in a state and an action and will output an estimate of the expected future return following that state-action pair. Value functions are defined with respect to policies. The value of a state under policy $$\pi$$ is written with $$v_\pi (s)$$. In English, the value of a state is the expected return when an agent starts in state $$s$$ and from then on acts according to $$\pi$$. Mathematically:

$$v_\pi(s) \stackrel{.}{=} \mathbb{E}_\pi[G_t \vert S_t = s] = \mathbb{E}_\pi[\sum_{i=1}^N \gamma^i R_{i+t+1} \vert S_t = s]$$

## Agent-Environment Interaction Trajectories

An agent-environment interaction trajectory is the sequence of observations seen by the agent, actions the agent takes, and then new states and rewards. Agent-environment interaction occurs in discrete time steps, $$t = 0, 1, 2, 3, \dots$$. As we know, at every time step our agent takes an action $$a_t$$ and receives a new observation $$s_{t+1}$$ and reward $$r_{t+1}$$. Writing this out more directly, an interaction trajectory progresses like this:

$$s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, r_3, s_3, a_3, \dots$$

We can denote interaction trajectories by $$\tau$$, the Greek letter Tau.

# Observations and actions

## Observation spaces

An observation is the state, or portion of the state, as it is observed by the agent. In fully-observable MDPs, the state and the observation can be identical, but sometimes are not (for example, in the Atari [DQN paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), the observations to the agent are the last few frames of the game, concatenated together), and in partially-observable MDPs, the agent cannot see the entirety of the environment's state. Therefore, the distinction between the *state* of an environment and the *observations* an agent receives is an important one.

Observation spaces can be discretely or continuously valued. An example of a discrete observation space is the layout of a tic-tac-toe board, where empty space is represented by a zero, X pieces are represented by a 1, and O pieces are represented by a 2. A continuous observation space example might be a vector of joint torques and velocities in a robot, like the one in the gif below.

![ant-mujoco-gif](https://openai.com/content/images/2017/05/image4.gif)

In this gif, the goal is to move the ant robot forward. The observations are a 28 dimensional vector of joint torques and velocities. The agent must learn to use these observations to take actions to move the ant forward.

## Action spaces

Similarly to observation spaces actions can be continuously or discretely valued. A simple example of a discrete action is one in Atari, where you move a joystick in one of four distinct directions to control where your character moves. Continuous actions can be more complex. For example, in the gif above, the actions are continuous numbers representing amounts of force to apply to the robot's joints.

## Reward functions

Reward functions yield a scalar signal at every step of the environment telling the agent how good the previous action was. The agent uses these signals to learn to take good actions in every state. To the agent, goodness is defined by how much reward was earned.

Reward functions can either be simple or complex. For a simple example, in the classic gridworld environment (see diagram below), the agent starts in one corner of a grid and must navigate an end state in the other corner of the grid. The reward at every step, no matter the action, is $$-1$$. This incentivizes the agent to navigate from start to end as quickly as possible.

![gridworld-environment](https://miro.medium.com/max/1454/1*G3q-q9gEiDc2fD8sPXHBpQ.png)

In the above image, the greyed out squares are the starting and ending points. Either one can be the start or end point. A more complicated example of a reward function would be the one for the ant robot above. The reward function for that agent is as follows:

$$r_t(s, a) = \frac{x' - x}{\Delta t} - \frac{\sum_{i=1}^{N} \vec{a}^2}{2} - \frac{1*10^{-3} * \sum_{i=1}^{N} (clip(\vec{c_E}, -1, 1))^2}{2}$$

Let's unpack this. $$r_t(s, a)$$ tells us that the reward is a function of the last state and action. $$x$$ and $$x'$$ are the $$x$$ position of the robot before and after the action, respectively. $$\Delta t$$ is the change in time before the last action and current action. $$\vec{a}$$ is the action vector, in this case the actions are a vector because we're applying forces to 8 different joints on the robot, so each entry in the vector tells us how much force to apply to the corresponding joint. $$C_E$$ is a vector of external forces on the body of the robot. $$clip$$ tells us to clip the vector $$\vec{c_E}$$ to have all values fall within the range $$[-1, 1]$$.
