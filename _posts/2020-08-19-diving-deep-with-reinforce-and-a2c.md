---
layout: post
author: Jacob Pettit
comments: true
tags: long-post reinforcement-learning code
title: Diving Deep with Policy Gradients - REINFORCE and A2C 
featured-image: /assets/imgs/Diving Deep with REINFORCE and A2C.png
---

**In this post, we'll talk about the REINFORCE policy gradient algorithm and the Advantage Actor Critic (A2C) algorithm. I'll be assuming familiarity with at least the basics of RL.**

{: class="table-of-content"}
* TOC
{:toc}
# A Tutorial on REINFORCE and A2C Policy Gradient Algorithms

During this, we'll focus on connecting theory to code. 

If you need a refresher on the basics of RL, I have a blog post on this [here](https://jfpettit.github.io/blog/2019/11/03/fundamentals-of-reinforcement-learning). For an interactive version of this blog post, see [this colab notebook](https://colab.research.google.com/drive/1F1Xv_cZk38HOikjBoviEco9piuyi9Dbq?usp=sharing)

We'll begin with some fundamental RL concepts and terminology, and then quickly move on to covering the algorithms.

A quick note as well, extra information is labelled under subsections titled **Extra Detail:**. These aren't necessary to read but are instead there for the interested reader who wants a bit more depth.

## Section 1: Some RL Background
---
### 1.1: What are Markov Decision Processes (MDPs)?
---
MDPs are the typical problem formulation for RL agents. 

![mdp](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

#### Extra Detail:

MDPs consist of an interaction loop between the agent and the environment, where on the first step, the agent receives some state from the environment, takes an action, and then receives a new state and a reward. The agent then takes in this new state and reward, and chooses the next action. The interaction continues until the episode (an episode is a period of agent-environment interaction) ends.

Some MDPs are not episodic, and instead are continuing (AKA infinite-horizon) tasks. We only look at episodic MDPS here.

### 1.2: Reward and Return:
---
In RL, the reward is used to define the goal that you want to solve. At each step, the RL agent receives a single scalar reward from the environment. The agent's goal is to maximize the reward it receives.

Return is calculated as the discounted cumulative sum of all reward earned over an episode.


### Extra Detail:

The simplest return formulation is just to take the (not discounted) sum over all reward earned during an episode. However, this doesn't work for continuing tasks, because as the horizon goes to infinity, so does the return earned. Let's write this out mathematically, denote the return as $$G_t$$, $$t$$ is the step in the environment, and $$r_t$$ is the reward given at step $$t$$. Then, the sum of the return is:

$$G_t \stackrel{.}{=} \sum_{t=0}^{t} r_t$$

We get around the return going to infinity by introducing a discount factor on the return. The discount factor is represented by $$\gamma$$ and is frequently set to a value between 0.95 and 0.99. The discounted return is written as:

$$G_t \stackrel{.}{=} \sum_{i=0}^{\infty} \gamma^i r_{i+t+1}$$

Defining it this way makes it a telescoping sum, so as time ($$t$$) goes to infinity, the reward goes to zero.

## Policies and Value functions

### Policies

A policy (denoted by $$\pi$$) is a mapping from:
- A state ($$s$$) to an action ($$a$$) (a deterministic policy): such that $$\pi \rightarrow a$$. Or it maps from 
- A state to a probability distribution over actions (a stochastic policy) such that: 

$$\pi(a|s) = \mathbb{P}_\pi [A = a | S = s]$$

In either case, our goal is to find an optimal policy so that we can be sure that whatever the state of our environment is, we are acting optimally!

### Extra Detail:

A stochastic policy tells us that the output of the policy (the action) is conditioned on the state s: 

$$\mathbb{P}_\pi [A = a | S = s]$$ 

is the way of saying that the probability of action $$a$$ being $$A$$ is dependent on $$s$$ equaling $$S$$. A deterministic policy is more straightforward and maps directly from state to action.

### Value functions

A state's value is determined by how much reward is expected to follow it. If a state is followed by a lot of reward, it must be good, and therefore should have a high value. Value functions are always defined with respect to policies. This is because how much reward should follow a state is influenced by how the policy behaves. If the policy is an optimal or near-optimal one, then the value function associated with that policy will predict different expected returns than a value function associated with a poorly performing policy.

The state-value function, $$V_\pi(s)$$, is a mapping from a state to an expectation of how much reward should follow that state. It assumes that the actor in the environment will behave in an on-policy manner (all future actions are true to the current policy) for all remaining steps into the future.

The action-value function $$Q_\pi(s, a)$$ maps from a state and the action taken in that state to an expectation of how much reward should follow that state-action pair. It assumes that the action $$a$$ may or may not have been on-policy but that all other actions taken into the future will be on-policy.

Here, our implementations will only use state-value functions. But, work like the famous Atari-playing DQN, and other, more advanced algorithms make use of the Q-value function instead of the state-value function.

**Relationship between optimal Q-function and optimal policy**:

When the optimal policy is in state $$s$$, it chooses the action $$a$$ that gives the maximum expected future return. Because of this, when we have the optimal q-function, we can get the optimal policy by:

$$a^*(s) = \underset{a}{argmax} \space q^*(s,a)$$

The * denotes optimality.

### Extra Detail:

For those who really want to see the equations for the state-value and action-value functions, I've included them here.

State-value function:

$$V_\pi(s) \stackrel{.}{=} \mathbb{E}_\pi [G_t | s_t = s] = \mathbb{E}_\pi [\sum_{i=1}^\infty \gamma^i r_{i+t+1}| s_t = s]$$

Notice that the $$G_t$$ and its expansion on the right hand side are from the Reward and Return: Extra Detail section above. Since our value function is maximizing expected future return, we can use those definitions again here.

Action-value function:

$$Q_\pi(s, a) \stackrel{.}{=} \mathbb{E}_\pi [G_t | s_t = s, a_t = a] = \mathbb{E}_\pi [\sum_{i=1}^\infty \gamma^i r_{i+t+1}| s_t = s, a_t = a]$$

Again, the definitions from reward and return are used here.

To dig deeper, read about the Bellman equations, which demonstrate a nice, recursive, self-consistent set of equations for the value of a current state and the values of following states. 

## A brief bonus: exploration/exploitation tradeoff

Many people have heard of the exploration/exploitation tradeoff in RL. Often, when we are dealing with a deterministic policy, we force it to explore by either injecting some noise into the action, or by randomly sampling actions occasionally. There are also other, more sophisticated exploration methods.

Today, we won't worry about any of that since we are dealing with policy gradient methods. These algorithms output a probability distribution over actions, and at each timestep, we sample from that distribution to get our next action. Because of this, the exploration factor is effectively built-in to these algorithms.

### Some quick terminology clarification

- Epoch: Refers to collecting one batch of experiences (batch of experiences often called steps per epoch in the environment and training on them.
    - i.e. I've set my epoch batch size to 4000, so my agent collects 4000 interactions with the environment and trains on them. Then, for the next epoch, we collect 4000 new interactions, and train on those. And so on.
- Minibatch size: Refers to sampling minibatches from the epoch batch of experiences and training on each minibatch.
    - i.e. my minibatch size is 100, so I'll sample minibatches of size 100 from the epoch batch of 4000 and train on each minibatch until they're gone.

![CartPole](https://camo.githubusercontent.com/7089af78ce27348d2a71698b6913f7656a6713cc/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f3830302f312a6f4d5367325f6d4b677541474b793143363455466c772e676966)

The environment we will work on today is the [CartPole-v1](http://gym.openai.com/envs/CartPole-v1/) environment, where the agent's goal is to balance the pole atop the cart. This is a classic RL problem.


```python
# install required packages
!pip install -q pytorch-lightning
!pip install -q gym
!pip install -q pybullet
```

```python
# import needed packages
import pytorch_lightning as pl # provides training framework pytorch-lightning.readthedocs.io
import gym # provides RL environments gym.openai.com
import pybullet_envs # extra RL environments https://pybullet.org/wordpress/
import numpy as np # linear algebra https://numpy.org/
import matplotlib.pyplot as plt # plotting https://matplotlib.org/
import torch # Neural network utilities pytorch.org/
import torch.nn as nn # NN building blocks
import torch.nn.functional as F # loss functions, activation functions
import torch.optim as optim # optimizers
from torch.utils.data import Dataset, DataLoader # dataset utilities
from typing import Union, Optional, Any, Tuple, List # type hinting https://docs.python.org/3/library/typing.html
import sys # direct printing to stdout or to file https://docs.python.org/3/library/sys.html
from argparse import Namespace, ArgumentParser # argument handling https://docs.python.org/3/library/argparse.html
from scipy.signal import lfilter # helpful in return discounting https://www.scipy.org/
```

## The Humble MLP

The MLP (Multi-Layer Perceptron) will be used to parameterize our policies and value functions. We'll set it up to take in whatever input size (observation sizes change depending on your environment) and to map to whatever output size (action space sizes change too). Make the activation function optional so that we can use the output directly for logits of a distribution over actions.


```python
class MLP(nn.Module):
    def __init__(
        self,
        in_features: int, # observation dimensions
        hidden_sizes: tuple, # hidden layer sizes
        out_features: int, # action dimensions
        activation: callable = nn.Tanh(), # activation function
        out_activation: callable = nn.Identity(), # output activation function
        out_squeeze: bool = False # whether to apply a squeeze to the output
      ):
        super().__init__()
        layer_sizes = [in_features] + list(hidden_sizes) + [out_features]
        self.activation = activation
        self.out_activation = out_activation
        self.out_squeeze = out_squeeze

        self.layers = nn.ModuleList()

        for i, l in enumerate(layer_sizes[1:]):
            self.layers.append(nn.Linear(layer_sizes[i], l))

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))

        x = self.out_activation(self.layers[-1](x))

        if self.out_squeeze:
            x = torch.squeeze(x, -1)
    
        return torch.squeeze(x, -1) if self.out_squeeze else x

```

## Actor

The Actor class provides a structure to follow when we define our policy. 

By setting the forward pass in Actor, and afterwards inheriting from Actor when we define a new policy, we only have to worry about defining the ```action_distribution``` and ```logprob_from_distribution``` functions.

Since we are mapping from states to a probability distribution over actions, $$\pi(a \mid s)$$, our ```action_distribution``` function takes input states as an argument and should return a distribution over actions. 

In policy gradient algorithms, the log-probability of a chosen action is frequently used in the loss functions. We write the ```logprob_from_distribution``` function to take in our current policy distribution $$\pi(a \mid s)$$ and the selected action and then return the log-probability of that action under the current policy distribution.

The ```forward``` function combines these steps for us. It gets the current policy, and if an action $$a$$ is input as an argument, it computes the log-probability of that action under the policy. 


```python
class Actor(nn.Module):
    def action_distribution(self, states):
        """
        Get action distribution conditioned on input states.

        Args:
          states: Input states (works with one or many) to get an action distribution with respect to.

        Returns:
          Should return a torch.distributions distribution object.
        """
        raise NotImplementedError

    def logprob_from_distribution(self, policy_distribution, action):
        """
        Calculate log-probability of actions taken under a given policy distribution.

        Args:
          policy_distribution: A torch.distributions object, the current policy distribution.
          action: The action that was taken.

        Returns:
          Log probability of the input action under input policy.
        """
        raise NotImplementedError

    def forward(self, state, a = None):
        """
        Get policy distribution on input state and, if action is input, get logprob of that action.

        Args:

        """
        policy = self.action_distribution(state)
        logp_a = None
        if a is not None:
            logp_a = self.logprob_from_distribution(policy, a)
        return policy, logp_a
```

## The Categorical Policy

In environments with discrete action spaces (i.e. the valid actions are integers (1, 2) for example), we can learn a Categorical distribution over actions, and then sample from that distribution to get an action at each timestep in the environment. This follows our math before of having a stochastic policy map from a state to a distribution over actions, $$\pi(a \mid s)$$.

We train our policy network (here, an MLP) to output logits and then paramaterize the action distribution with those logits.


```python
class CategoricalPolicy(Actor):
    """
    Define a categorical policy over a discrete action space.
    
    Does not work on any other action space.
    
    Inherits from Actor, and uses forward method from Actor.
    
    Args:
        state_dim: Input state size.
        hidden_sizes: Hidden layer sizes for MLP.
        action_dim: Action space size.
    """
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: tuple,
        action_dim: int
        ):
        super().__init__()

        self.mlp = MLP(state_dim, hidden_sizes, action_dim)

    def action_distribution(self, states):
        logits = self.mlp(states)
        return torch.distributions.Categorical(logits=logits)

    def logprob_from_distribution(self, policy, actions):
        return policy.log_prob(actions)
```

## Preparing for REINFORCE

We are going to start with implementing the REINFORCE algorithm.

REINFORCE is the simplest policy-gradient algorithm. It needs only one network, the policy, and it learns to get as much reward as possible using only the reward signal from the environment and log-probabilities of actions under the policy. As we begin implementing more algorithm-specific stuff we'll talk about the math.

After discussing REINFORCE, we'll talk about an actor-critic method (A2C), which learns a policy (the actor) and a value function (the critic).

It's helpful to define a class with a specific method for computing an action and its log-prob at each environment step, so we will do that here. The ReinforceActor has a ```step``` method that we'll call at each timestep of the environment, and it'll return to us the action and log-probability of the selected action.


```python
class ReinforceActor(nn.Module):
    """
    A cleaned-up actor for the REINFORCE algorithm.
    
    Args:
        state_space: Actual state space from the environment. Is a gym.spaces type.
        hidden_sizes: Hidden layer sizes for MLP policy.
        action_space: Action space from the environment. Is a gym.spaces type.
    """
    def __init__(
        self,
        state_space: gym.spaces,
        hidden_sizes: tuple,
        action_space: gym.spaces
        ):
        super().__init__()

        state_size = state_space.shape[0]

        """
        Check to make sure that the action space is compatible with our Categorical Policy.
        """
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
            self.policy = CategoricalPolicy(state_size, hidden_sizes, action_dim)
        else:
            raise ValueError(f"Env has action space of type {type(action_space)}. ", \
            "REINFORCE Actor currently only supports gym.spaces.Discrete action spaces!")

    def step(self, state):
        """
        Get an action and the log-probability of that action from the policy.
        
        Args:
            state: Current state of the environment.
            
        Returns:
            action: NumPy array of the chosen action.
            action_logp: Log prob of the chosen action.
        """
        with torch.no_grad():
            policy_current = self.policy.action_distribution(state) # get current policy given current state
            action = policy_current.sample() # sample an action from the policy
            # calculate the log-probability of that action under the current policy
            action_logp = self.policy.logprob_from_distribution(policy_current, action)
        return action.numpy(), action_logp.numpy()
```

## The Policy Gradient Buffer

As we train, it's necessary to be able to store the agent's experiences so that we can later use them to compute loss and perform gradient updates. We use the buffer to do this. At each timestep of the environment, we store a tuple of state, action, reward, value (only in actor-critic methods) and logprob of the action. When it is time to calculate loss and update, we need to calculate advantages, normalize returns, and then get everything from the buffer so that we can train on it.

### What is Advantage?

Advantage can be thought of as an action-value. There are many ways of estimating advantage, but they all aim to estimate the same equation. The advantage is equal to the action-value of a state, action pair minus the state-value of that state:

$$A(s,a) = Q_\pi(s,a) - V_\pi(s)$$

Intuitively, we can see why this makes sense by recognizing that the Q-value is the value of both being in state $$s$$ and taking action $$a$$, while the state-value only estimates the value of state $$s$$, and disregards any action. By subtracting the pure state-value from the Q-value, we are left with an action-value, AKA Advantage.

### But where does Q come from? We are only learning a policy!

We have an estimate of Q given to us by the environment; the reward signal! The Q function takes in states and actions and tries to predict how much reward will be earned into the future. The environment gives us this directly. When we calculate advantage, we can substitute Q with the return earned by the agent after being in state $$s$$ and taking action $$a$$. Then, we only have to learn a state-value function. Actor-critic  methods learn a value function, but REINFORCE does not.


```python
"""This class is borrowed from OpenAI's SpinningUp code, so thanks to them for this!"""

class PolicyGradientBuffer:
    """
    A buffer for storing trajectories experienced by an agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    
    Args:
        obs_dim: Size of the observations
        act_dim: Size of the action space
        size: Total size of the buffer. i.e. size=4000 stores 4000 interaction tuples before the buffer is full.
        gamma: Discount factor for return calculation
        lam: Lambda argument for GAE-lambda advantage estimation.
    """

    def __init__(
        self,
        obs_dim: Union[tuple, int],
        act_dim: Union[tuple, int],
        size: int,
        gamma: Optional[float] = 0.99,
        lam: Optional[float] = 0.95,
    ):
        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32) # not used in REINFORCE, as we are not learning a value function
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        # ptr used to store current location in the buffer, path_start_idx used to mark where episodes begin and end in the buffer
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size 

    def store(
        self,
        obs: np.array,
        act: np.array,
        rew: Union[int, float, np.array],
        val: Union[int, float, np.array],
        logp: Union[float, np.array],
    ):
        """
        Append one timestep of agent-environment interaction to the buffer.
        
        Args:
            obs: Current observations (state)
            act: Action taken in the state
            rew: Reward earned for taking the action in the current state
            val: Estimated value of the current state. NOTE: This is not used in REINFORCE, so pass in zeros for this argument when training REINFORCE.
            logp: Log probability of taking the action in the current state.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val # not used in REINFORCE, because we are not learning a value function. Pass in zeros during REINFORCE training.
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: Optional[Union[int, float, np.array]] = 0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        
        Args:
            last_val: This only needs to be passed in when the epoch ends before the episode ends.
            When training an actor-critic method, this is the estimate of future reward from the critic. 
            In REINFORCE, pass in the last obtained reward for this argument.
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val) # not used in REINFORCE, not learning a value function

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        # adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

    def _combined_shape(
        self, length: Union[int, np.array], shape: Optional[Union[int, tuple]] = None
    ):
        """
        Get combined shape of a length and another shape tuple.
        
        Args:
            length: Length to combine into shape tuple
            shape: Original shape tuple 
        """
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x: np.array, discount: float):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
            x1,
            x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
            
        Args:
            x: Vector to discount
            discount: discount factor
        """
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
```

## Setting up an RL dataset

PyTorch Lightning requires that the user feed data to their networks using PyTorch datasets. Here, we set up a simple RL dataset. When initialized, it takes the data from the buffer as input. At each index, it'll return one tuple from the buffer.


```python
class PolicyGradientRLDataset(Dataset):
    """
    A PyTorch dataset for training Policy Gradient RL algorithms.
    
    Args:
        data: The result of calling .get() from the buffer.
    """
    def __init__(
        self,
        data 
    ):
        self.data = data 

    def __len__(self):
        return len(self.data[2]) 

    def __getitem__(self, idx):
        """
        Return idx-th tuple from the buffer.
        
        Args:
            idx: index of the buffer to pull data from.
        
        Returns:
            tuple of:
                state: state at that index
                action: action taken in state
                adv: advantage estimation for taking the action in the state
                rew: observed reward for taking the action in the state
                logp: log-probability of taking the action in the state
        """
        state = self.data[0][idx]
        act = self.data[1][idx]
        adv = self.data[2][idx]
        rew = self.data[3][idx]
        logp = self.data[4][idx]

        return state, act, adv, rew, logp
```

## REINFORCE Algorithm

Now we'll cover the REINFORCE algorithm implementation. 

REINFORCE learns to update the policy distribution in such a way that it makes actions which led to high reward more likely and actions that led to low reward less likely. The gradient of our policies performance (performance is denoted by J) under REINFORCE only needs two arguments: the log-probability of a selected action and the return obtained following that action. In a more ML-friendly parlance, we can also call this the policy loss. We denote the weights of our NN policy by $$\theta$$. We can write it out:

$$\nabla J(\theta) = \log{\mathbb{P} (a_t)} * G_t$$

This equation is exactly what we need to update our policy weights according to. It's simple to write in code as well:

```
policy_loss = -(action_logps * returns).mean()
```

We have to optimize the negative of the loss function instead of the original because pre-built ML optimizers are intended to minimize your loss function. But here, we want to maximize the loss (remember, the loss here is really our policies performance!), so we trick the optimizer into doing this by using the negative loss.


```python
class REINFORCE(pl.LightningModule):

    def __init__(
        self,
        hparams
    ):

        super().__init__()

        self.hparams = hparams # register hyperparameters to PyTorch Lightning 

        hids = (int(i) for i in hparams.hidden_layers) # make sure hidden layer sizes is a tuple of ints

        torch.manual_seed(hparams.seed) # random seeding
        np.random.seed(hparams.seed)

        self.env = gym.make(hparams.env_name) # make environment for agent to interact with

        self.actor = ReinforceActor(self.env.observation_space, hids, self.env.action_space) # intialize actor

        # here we intialize our buffer
        self.buffer = PolicyGradientBuffer(
            self.env.observation_space.shape[0], # state size
            self.env.action_space.shape, # action space size
            size=hparams.steps_per_epoch, # how many interactions to collect per epoch
            gamma=hparams.gamma, # gamma disount factor for return calculation
            lam=hparams.lam # lambda factor for GAE-lambda advantage estimation
            )

        self.steps_per_epoch = hparams.steps_per_epoch # register variables to class
        self.policy_lr = hparams.policy_lr

        self.tracker_dict = {} # intialize an empty metrics tracking dictionary

        self.inner_loop() # generate our first epoch of data!!!

        self.minibatch_size = self.steps_per_epoch // 10 # set minibatch size for dataloader
        if hparams.minibatch_size is not None:
            self.minibatch_size = hparams.minibatch_size

    def forward(self, state: torch.Tensor, a: torch.Tensor = None) -> torch.Tensor:
        r"""
        Forward pass for the agent.

        Args:
            state (PyTorch Tensor): state of the environment
            a (PyTorch Tensor): action agent took. Optional. Defaults to None.
        """
        return self.actor.policy(state, a) 

    def configure_optimizers(self) -> tuple:
        r"""
        Set up optimizers for agent.

        Returns:
            policy_optimizer (torch.optim.Adam): Optimizer for policy network.
            value_optimizer (torch.optim.Adam): Optimizer for value network.
        """
        self.policy_optimizer = torch.optim.Adam(self.actor.policy.parameters(), lr=self.policy_lr)
        return self.policy_optimizer

    def inner_loop(self) -> None:
        r"""
        Run agent-env interaction loop. 

        Stores agent environment interaction tuples to the buffer. Logs reward mean/std/min/max to tracker dict. Collects data at loop end.

        """
        state, reward, episode_reward, episode_length = self.env.reset(), 0, 0, 0 # reset state, reward, etc to initial values 
        rewlst = [] # empty reward tracking list
        lenlst = [] # empty episode length tracking list

        for i in range(self.steps_per_epoch): # collect steps_per_epoch interactions between agent and environment
            action, logp = self.actor.step(torch.as_tensor(state, dtype=torch.float32)) # get action and action log probability

            next_state, reward, done, _ = self.env.step(action) # step environment
            
            # store interaction
            self.buffer.store(
                state,
                action,
                reward,
                0, # recall REINFORCE doesn't use a value function, so just pass zero to our buffer for this argument.
                logp
            )

            state = next_state # step the state to the next state
            episode_length += 1 # increment episode_length
            episode_reward += reward # increment episode reward


            timeup = episode_length == 1000 # check if we have hit our max episode length (here, 1000, but other values are valid too)
            over = done or timeup # check if episode is over (done == True) or timeup
            epoch_ended = i == self.steps_per_epoch - 1 # check if we've hit the end of our epoch
            if over or epoch_ended:
                if timeup or epoch_ended:
                    last_val = reward 
                    # if timeup or epoch_ended, the episode was cut off before it truly ended
                    # so give the current reward as an estimate of future reward
                else:
                    last_val = 0 
                    # otherwise, the episode wasn't cut off before it really ended
                    # and it is unnecessary to estimate future reward
                self.buffer.finish_path(last_val) # finish the episode in the buffer

                if over:
                    # store the episode reward and episode length
                    rewlst.append(episode_reward)
                    lenlst.append(episode_length)
                state, episode_reward, episode_length = self.env.reset(), 0, 0 # reset state, episode_reward, episode_length to intial values

        # at epoch end, store epoch metrics in local tracking dict
        trackit = {
            "MeanEpReturn": np.mean(rewlst),
            "StdEpReturn": np.std(rewlst),
            "MaxEpReturn": np.max(rewlst),
            "MinEpReturn": np.min(rewlst),
            "MeanEpLength": np.mean(lenlst),
            "Epoch": self.current_epoch
        }
        self.tracker_dict.update(trackit) # update overall tracking dictionary

        self.data = self.buffer.get() # update data with latest epoch data

    def calc_pol_loss(self, logps, returns) -> torch.Tensor:
        r"""
        Loss for REINFORCE policy gradient agent.
        """
        return -(logps * returns).mean()

  
    def training_step(self, batch: Tuple, batch_idx: int) -> dict:
        r"""
        Calculate policy loss over input batch.

        Also compute and log policy entropy and KL divergence.

        Args:
          batch (Tuple of PyTorch tensors): Batch to train on.
          batch_idx: batch index.
        """
        states, acts, _, rets, logps_old = batch

        policy, logps = self.actor.policy(states, acts) # get updated policy and logp estimates on stored states and actions 
        # (need this for PyTorch gradients)
        pol_loss = self.calc_pol_loss(logps, rets)

        ent = policy.entropy().mean() # estimate policy entropy
        kl = (logps_old - logps).mean() # calculate sample estimate of KL divergence between new and old policy
        log = {"PolicyLoss": pol_loss, "Entropy": ent, "KL": kl}
        self.tracker_dict.update(log)

        return {"loss": pol_loss, "log": log, "progress_bar": log}
      
    def training_step_end(
        self,
        step_dict: dict
    ) -> dict:
        r"""
        Method for end of training step. Makes sure that episode reward and length info get added to logger.

        Args:
            step_dict (dict): dictioanry from last training step.

        Returns:
            step_dict (dict): dictionary from last training step with episode return and length info from last epoch added to log.
        """
        step_dict['log'] = self.add_to_log_dict(step_dict['log'])
        return step_dict

    def add_to_log_dict(self, log_dict) -> dict:
        r"""
        Adds episode return and length info to logger dictionary.

        Args:
            log_dict (dict): Dictionary to log to.

        Returns:
            log_dict (dict): Modified log_dict to include episode return and length info.
        """
        add_to_dict = {
            "MeanEpReturn": self.tracker_dict["MeanEpReturn"],
            "MaxEpReturn": self.tracker_dict["MaxEpReturn"],
            "MinEpReturn": self.tracker_dict["MinEpReturn"],
            "MeanEpLength": self.tracker_dict["MeanEpLength"],
            }
        log_dict.update(add_to_dict)
        return log_dict

    def train_dataloader(self) -> DataLoader:
        r"""
        Define a PyTorch dataset with the data from the last :func:`~inner_loop` run and return a dataloader.

        Returns:
            dataloader (PyTorch Dataloader): Object for loading data collected during last epoch.
        """
        dataset = PolicyGradientRLDataset(self.data)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, sampler=None)
        return dataloader

    def printdict(self, out_file: Optional[str] = sys.stdout) -> None:
        r"""
        Print the contents of the epoch tracking dict to stdout or to a file.

        Args:
            out_file (sys.stdout or string): File for output. If writing to a file, opening it for writing should be handled in :func:`on_epoch_end`.
        """
        self.print("\n", file=out_file)
        for k, v in self.tracker_dict.items():
            self.print(f"{k}: {v}", file=out_file)
        self.print("\n", file=out_file)
  
    def on_epoch_end(self) -> None:
        r"""
        Print tracker_dict, reset tracker_dict, and generate new data with inner loop.
        """
        self.printdict()
        self.tracker_dict = {}
        self.inner_loop()

```

Above, we've implemented the REINFORCE algorithm as a PyTorch Lightning module, and below we can set up the parameters for the algorithm as a Namespace object and run the algorithm using PyTorch Lightning's Trainer class.

```python
epochs = 100 # epochs to train for
hparams = Namespace(env_name="CartPole-v1", hidden_layers=(64, 32), seed=123, \
                    gamma=0.99, lam=0.97, steps_per_epoch=4000, policy_lr=3e-4, minibatch_size=400) 
# all necessary arguments for REINFORCE. Mess with these!
trainer = pl.Trainer(
    reload_dataloaders_every_epoch = True, # need to update data every epoch with latest batch of data
    early_stop_callback = False, # don't do early stopping
    max_epochs = epochs # train for no more than whatever epochs is set to
)

agent = REINFORCE(hparams) # init agent
trainer.fit(agent) # run training
```

## The Actor-Critic

While the Actor-Critic is largely similar to the REINFORCE Actor, it is different in one key way: it also contains a value function. The value function is trained to estimate future return from the current state. We train it in a supervised fashion. 
- We store the observed returns from the environment during the interaction period.
- Then, during the training step, we update the value function using Mean Squared Error between the observed return and predicted return.  

### OK, but why use a Critic?

The critic allows us to calculate advantage, so we can use the action-value in the policy loss instead of having to use the observed returns like in REINFORCe. This reduces the variance of our policy, and leads to more stable training.

The algorithm we implement here is called Advantage Actor Critic (A2C), and it uses the same loss function as REINFORCE, except it uses the Advantage in place of the observed returns:

$$\nabla J(\theta) = \log{\mathbb{P} (a)} * A$$

Where $$A$$ is the Advantage estimate.

This is again simple to write in code:

```
policy_loss = -(action_logps * advantages).mean()
```

Our reason for optimizing the negative loss function is the same as before, ML optimizers like to minimize loss functions, but we need to maximize this one, so we trick the optimizer by backpropagating the negative of the loss.

Where before in REINFORCE we only needed a policy, now we need a policy and a value function, so we implement an actor-critic network below.

Both the policy and the value function are still MLPs (and the policy is actually the `CategoricalPolicy` we defined above), and we still use a `step` method to take in the current state and get action, action log-probabilities, and value estimates from the network.

```python
class ActorCritic(Actor):
    """
    An Actor-Critic class for A2C.
    
    Args:
        state_space: Actual state space from the environment. Is a gym.spaces type.
        hidden_sizes: Hidden layer sizes for MLP policy.
        action_space: Action space from the environment. Is a gym.spaces type.
    """
    def __init__(
        self,
        state_space: gym.spaces.Space, # environment state space
        hidden_sizes: tuple, # hidden layer sizes for MLP policy
        action_space: gym.spaces.Space # action space from the environment
    ):
        super().__init__()

        state_size = state_space.shape[0] # get state size to pass to policy 

        # Check to be sure that the environment action space is compatible with the Categorical policy.
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = action_space.n
            self.policy = CategoricalPolicy(state_size, hidden_sizes, action_dim) # make policy
        else:
            raise ValueError(f"Env has action space of type {type(action_space)}. ", 
            "A2C Actor-Critic currently only supports gym.spaces.Discrete action spaces!")

        self.value_f = MLP(state_size, list(hidden_sizes), 1) # make MLP value function. 
        # Its output size is 1 because it should output a single number for expected future return.

    def step(self, state):
        """
        Function to get action, logprob of action, and state-value estimate from the actor-critic at each environment timestep.
        
        Args:
            state: Current state of the environment.
            
        Returns:
            action: Selected action
            action_logp: Log probability of the selected action
            value: estimated state-value of the input state
        """
        with torch.no_grad():
            value = self.value_f(state)
            policy_current = self.policy.action_distribution(state)
            action = policy_current.sample()
            action_logp = self.policy.logprob_from_distribution(policy_current, action)
        return action.numpy(), action_logp.numpy(), value.numpy()
```

Now that we've implemented our actor-critic, we have all of the building blocks in place to build our Advantage Actor Critic algorithm.

```python
class A2C(pl.LightningModule):
    """
    Class for training the A2C algorithm on an env.
    
    Args:
        hparams: Namespace object containing all parameters.
    """
    def __init__(
        self,
        hparams
    ):
        super().__init__()

        self.hparams = hparams # register parameters to pytorch lightning

        hids = (int(i) for i in hparams.hidden_layers) # make sure hidden layer sizes is a tuple of ints

        torch.manual_seed(hparams.seed) # random seeding
        np.random.seed(hparams.seed)

        self.env = gym.make(hparams.env_name) # make RL environment

        self.actor_critic = ActorCritic(self.env.observation_space, hids, self.env.action_space) # create actor critic

        # initialize buffer
        self.buffer = PolicyGradientBuffer(
            self.env.observation_space.shape[0], 
            self.env.action_space.shape,
            size=hparams.steps_per_epoch,
            gamma=hparams.gamma,
            lam=hparams.lam
            )

        # register parameters to class
        self.steps_per_epoch = hparams.steps_per_epoch
        self.policy_lr = hparams.policy_lr # policy optimizer lr
        self.value_f_lr = hparams.value_f_lr # value function optimizer lr
        self.train_iters = hparams.train_iters

        self.tracker_dict = {} # init empty metric tracker dictionary

        self.inner_loop() # create first batch of data!!!

        # set minibatch size for dataloader usage
        self.minibatch_size = self.steps_per_epoch // 10
        if hparams.minibatch_size is not None:
            self.minibatch_size = hparams.minibatch_size

    def forward(self, state: torch.Tensor, a: torch.Tensor = None) -> torch.Tensor:
        r"""
        Forward pass for the agent.

        Args:
            state (PyTorch Tensor): state of the environment
            a (PyTorch Tensor): action agent took. Optional. Defaults to None.
        """
        return self.actor_critic.policy(state, a) 

    def configure_optimizers(self) -> tuple:
        r"""
        Set up optimizers for agent.

        Returns:
            policy_optimizer (torch.optim.Adam): Optimizer for policy network.
            value_optimizer (torch.optim.Adam): Optimizer for value network.
        """
        self.policy_optimizer = torch.optim.Adam(self.actor_critic.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.actor_critic.value_f.parameters(), lr=self.value_f_lr)
        return self.policy_optimizer, self.value_optimizer

    def inner_loop(self) -> None:
        r"""
        Run agent-env interaction loop. 

        Stores agent environment interaction tuples to the buffer. Logs reward mean/std/min/max to tracker dict. Collects data at loop end.

        """
        state, reward, episode_reward, episode_length = self.env.reset(), 0, 0, 0 # set state, reward, episode reward/length to initial values
        rewlst = [] # empty reward tracking list
        lenlst = [] # empty episode length tracking list

        for i in range(self.steps_per_epoch): # collect data batch of size steps_per_epoch
            action, logp, value = self.actor_critic.step(torch.as_tensor(state, dtype=torch.float32)) 
            # get action, action log prob, and state-value estimate

            next_state, reward, done, _ = self.env.step(action) # step environment with action

            # store interaction tuple
            self.buffer.store(
                state,
                action,
                reward,
                value, # this time, we are learning a value function, so we need to store our value estimates for each state. 
                logp
            )

            state = next_state # update state
            episode_length += 1 # increment episode length
            episode_reward += reward # increment episode rewad


            timeup = episode_length == 1000 # check if episode has reached maximum length of 1000
            over = done or timeup # check if episode is actually over (done == True) or if timeup
            epoch_ended = i == self.steps_per_epoch - 1 # check if we've reached the end of the epoch
            if over or epoch_ended:
                if timeup or epoch_ended:
                    # if the epoch has ended or the max episode length has been hit before the episode is over, 
                    # estimate future returns using the value function
                    last_val = self.actor_critic.value_f(torch.as_tensor(state, dtype=torch.float32)).detach().numpy()
                else:
                    # otherwise, the episode ended properly and we don't need to estimate future return
                    last_val = 0
                self.buffer.finish_path(last_val) # properly store the finished epoch in the buffer

                if over:
                    # update tracker lists
                    rewlst.append(episode_reward)
                    lenlst.append(episode_length)
                state, episode_reward, episode_length = self.env.reset(), 0, 0 # reset state and other variables to intial values

        # track epoch return and length metrics
        trackit = {
            "MeanEpReturn": np.mean(rewlst),
            "StdEpReturn": np.std(rewlst),
            "MaxEpReturn": np.max(rewlst),
            "MinEpReturn": np.min(rewlst),
            "MeanEpLength": np.mean(lenlst),
            "Epoch": self.current_epoch
        }
        # update class metric tracker dictionary
        self.tracker_dict.update(trackit)

        # update data with latest epoch data
        self.data = self.buffer.get()

    def calc_pol_loss(self, logps, advantages) -> torch.Tensor:
        r"""
        Loss for A2C policy.
        """
        return -(logps * advantages).mean()

    def calc_val_loss(self, values, returns) -> torch.Tensor:
        """
        MSE loss for value function.
        """
        return ((values - returns)**2).mean()
  
    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Depending on which optimizer is being used (optimizer_idx) update the corresponding network.
        """
        states, acts, advs, rets, logps_old = batch

        if optimizer_idx == 0:
            pol_loss_old = self.calc_pol_loss(logps_old, advs)

            policy, logps = self.actor_critic.policy(states, a=acts)
            pol_loss = self.calc_pol_loss(logps, advs)

            ent = policy.entropy().mean().item() 
            kl = (logps_old - logps).mean().item()
            delta_pol_loss = (pol_loss - pol_loss_old).item()
            log = {"PolicyLoss": pol_loss_old.item(), "DeltaPolLoss": delta_pol_loss, "Entropy": ent, "KL": kl}
            loss = pol_loss

        elif optimizer_idx == 1:
            values_old = self.actor_critic.value_f(states)
            val_loss_old = self.calc_val_loss(values_old, rets)
            # value function can take multiple passes over the input data.
            for i in range(self.train_iters):
                self.value_optimizer.zero_grad()
                values = self.actor_critic.value_f(states)
                val_loss = self.calc_val_loss(values, rets)
                val_loss.backward()
                self.value_optimizer.step()

            delta_val_loss = (val_loss - val_loss_old).item()
            log = {"ValueLoss": val_loss_old.item(), "DeltaValLoss": delta_val_loss}
            loss = val_loss

        self.tracker_dict.update(log)
        return {"loss": loss, "log": log, "progress_bar": log}
      
    def training_step_end(
        self,
        step_dict: dict
    ) -> dict:
        r"""
        Method for end of training step. Makes sure that episode reward and length info get added to logger.

        Args:
            step_dict (dict): dictioanry from last training step.

        Returns:
            step_dict (dict): dictionary from last training step with episode return and length info from last epoch added to log.
        """
        step_dict['log'] = self.add_to_log_dict(step_dict['log'])
        return step_dict

    def add_to_log_dict(self, log_dict) -> dict:
        r"""
        Adds episode return and length info to logger dictionary.

        Args:
            log_dict (dict): Dictionary to log to.

        Returns:
            log_dict (dict): Modified log_dict to include episode return and length info.
        """
        add_to_dict = {
            "MeanEpReturn": self.tracker_dict["MeanEpReturn"],
            "MaxEpReturn": self.tracker_dict["MaxEpReturn"],
            "MinEpReturn": self.tracker_dict["MinEpReturn"],
            "MeanEpLength": self.tracker_dict["MeanEpLength"]}
        log_dict.update(add_to_dict)
        return log_dict

    def train_dataloader(self) -> DataLoader:
        r"""
        Define a PyTorch dataset with the data from the last :func:`~inner_loop` run and return a dataloader.

        Returns:
            dataloader (PyTorch Dataloader): Object for loading data collected during last epoch.
        """
        dataset = PolicyGradientRLDataset(self.data)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, sampler=None, num_workers=4)
        return dataloader

    def printdict(self, out_file: Optional[str] = sys.stdout) -> None:
        r"""
        Print the contents of the epoch tracking dict to stdout or to a file.

        Args:
            out_file (sys.stdout or string): File for output. If writing to a file, opening it for writing should be handled in :func:`on_epoch_end`.
        """
        self.print("\n", file=out_file)
        for k, v in self.tracker_dict.items():
            self.print(f"{k}: {v}", file=out_file)
        self.print("\n", file=out_file)
  
    def on_epoch_end(self) -> None:
        r"""
        Print tracker_dict, reset tracker_dict, and generate new data with inner loop.
        """
        self.printdict()
        self.tracker_dict = {}
        self.inner_loop()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None
    ):
        """
        For compatibility with PyTorch Lightning, need to ignore optimizer step for value function optimizer because it is done in training step.
        """
        if optimizer_idx == 0:
            if self.trainer.use_tpu and XLA_AVAILABLE:
                xm.optimizer_step(optimizer)
            elif isinstance(optimizer, torch.optim.LBFGS):
                optimizer.step(second_order_closure)
            else:
                optimizer.step()

                # clear gradients
                optimizer.zero_grad()

        elif optimizer_idx == 1:
            pass

    def backward(
        self,
        trainer,
        loss,
        optimizer,
        optimizer_idx
    ):
        """
        For compatibility with PyTorch Lightning, need to ignore backward pass for value function because it is done in training step.D
        """
        if optimizer_idx == 0:
            if trainer.precision == 16:
                # .backward is not special on 16-bit with TPUs
                if not trainer.on_tpu:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
            else:
                loss.backward()

        elif optimizer_idx == 1:
            pass

```

The below block runs the code, the Namespace object contains all of the necessary arguments to set up A2C. 

```python
epochs = 100
hparams = Namespace(env_name="CartPole-v1", hidden_layers=(64, 32), seed=123, \ 
                    gamma=0.99, lam=0.97, steps_per_epoch=4000, policy_lr=3e-4, \ 
                    value_f_lr=1e-3, minibatch_size=400, train_iters=80)
trainer = pl.Trainer(
    reload_dataloaders_every_epoch = True,
    early_stop_callback = False,
    max_epochs = epochs
)

agent = A2C(hparams)
trainer.fit(agent)
```

# Bonus section:

## Challenge Problems!!!

If you've enjoyed this stuff and want to try to learn to do something on your own, I'll list a couple of recommended next steps ("challenges") here.

1. Implement and train [PPO](https://arxiv.org/abs/1707.06347). The code from A2C doesn't require much modification to be converted into PPO.
2. Write policy networks for continuous action spaces. These are commonly called Gaussian policies, and they learn to output the mean of a (Normal) action distribution. Some implementations also learn the log standard devaiation of the distribution, but this isn't necessary here. You can fix the log standard deviation to -0.5 and achieve decent scores on most things.
  - Further reading to help with this: 
    - [SpinningUp](https://spinningup.openai.com/en/latest/)
    - [My open-source implementations (very similar to the code we've written here)](https://github.com/jfpettit/flare)

## Some Extra Reading!

For the curious person who wants to go deeper:

- [Sutton and Barto's Introduction to RL](http://incompleteideas.net/book/the-book-2nd.html)
  - This is really the holy grail of classical RL.
- [OpenAI's SpinningUp](https://spinningup.openai.com/en/latest/)
  - Another good introduction, doesn't require nearly as much time as the Sutton and Barto book but still provides a good overview of RL.
- [Lilian Weng's "A (Long) Peek Into Reinforcement Learning"](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
  - Lilian Weng's blog is in general a great resource, and she has written many blog posts on other RL topics as well.
