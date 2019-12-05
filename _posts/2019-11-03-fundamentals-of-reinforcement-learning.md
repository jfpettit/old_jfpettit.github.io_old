---
layout: post
author: Jacob Pettit
comments: true
title: Looking at the Fundamentals of Reinforcement Learning
tags: long-post reinforcement-learning
---

In this post, we'll get into the weeds with some of the fundamentals of reinforcement learning. Hopefully, this will serve as a thorough overview of the basics for someone who is curious and doesn't want to invest a significant amount of time into learning all of the math and theory behind the basics of reinforcement learning.

{: class="table-of-content"}
* TOC
{:toc}

# Markov Decision Processes

In reinforcement learning (RL), we want to solve a Markov Decision Process (MDP) by figuring out how to take actions that maximize the reward received. The actor in an MDP is called an agent. In an MDP, actions taken influence both current and future rewards received and actions influence future states the agent finds itself in. Because of this, solving an MDP requires that an agent is able to handle delayed reward and must be able to balance a trade-off between obtaining reward immediately and delaying reward collection. Below we have a diagram of the classic MDP formulation of agent-environment interaction.

![mdp-rl-interaction-loop](https://www.kdnuggets.com/images/reinforcement-learning-fig1-700.jpg)

At initialization, the environment outputs some state $$s_t$$. The agent observes this state and in response takes an action, $$a_t$$. This action is then applied to the environment, the environment is stepped forward in response to the action taken, and it yields a new state, $$s_{t+1}$$ and reward signal $$r_{t}$$ to the agent. This loop continues until the episode (period of agent-environment interaction) terminates.

Some examples of a state might be a frame of an Atari game or the current layout of pieces in a board game such as chess. The reward is a scalar signal calculated following a reward function, and the action can be something like which piece to move where on a chess board, or which direction to go in an Atari game.

## Model of the environment

The model of the environment consists of a state transition function and a reward function. Here, the transition function will be discussed and later we'll talk about reward functions. In finite MDPs (the kind of MDP we are concerned with), the number of states, actions, and rewards are all finite values. Because of this property, they also all have well-defined probability distributions. The distribution over the next state and next reward depends only on the previous state and action. This is called the Markov property. For the random variables $$s' \in S$$ and $$r \in \mathbb{R}$$, there is a probability at time $$t$$ that they'll have particular values. This probability is conditional on the previous state and action, and can be written out like this:

$$p(s', r | s, a) \stackrel{.}{=} P[s_t = s', r_t = r | s_{t-1} = s, a_{t-1} = a]$$

This 4-argument function, $$p$$, fully captures the dynamics of the MDP and tells us, formally, that our new state ($$s'$$) and reward are random variables whose distribution is conditioned on the previous state and action. A complete model of an MDP can be used to calculate anything we want about the environment. State-transition probabilities can be found, expected rewards for state-action pairs, and even expected rewards for state-action-next state triplets.

## Episodic and Continuing Tasks

An episodic MDP (we can also refer to an MDP as a task) is one with a clear stopping, or terminating, state. An example of this might be something like a game of chess, where the natural stopping state is when one player wins and the other loses.

A continuing task is one that doesn't have a clear stopping state, and can be allowed to continue indefinitely. Consider trying to train a robot to walk; there is not necessarily a clear stopping point. In practice, when we are training a simulated robot to walk, we terminate the episode when it falls over. Over time, however, the agent learns to walk successfully. Once it is successfully walking, there isn't an easily defined stopping point. Since we don't want to let a simulation run forever, we typically enforce some maximum number of interactions allowed per episode. Once this number is reached, the episode terminates.

## Reward & Return

We use reward to define the goal in the problem we'd like the RL agent to solve. At every step $$t$$, the agent receives a single, scalar reward signal $$r_t$$ from the environment. The agent's aim is to maximize the reward it receives over all future actions. This is called the return.

Return is the cumulative sum of all reward earned in an episode. Let's denote return with $$G_t$$. Then, the return of an episode is defined with:

$$ G_t \stackrel{.}{=} r_t + r_{t+1} + r_{t+2} + r_{t+3} + \dotsb + r_T $$

$$r_T$$ is the reward earned at the final step of an episode. $$\stackrel{.}{=}$$ means "defined by". We can rewrite the above expression more concisely:

$$ G_t \stackrel{.}{=} \sum_{t=0}^T r_t $$

The reward formulation above doesn't work for continuing tasks, because as the number of time steps goes to infinity, so does the reward. We need a formulation of reward that will converge to a finite value as the number of time steps goes to infinity. To do this, we can assign a discount factor to future rewards. We define a  discount rate parameter, $$\gamma$$ (gamma) and set $$0 \leq \gamma \leq 1$$. We incorporate $$\gamma$$ into our reward formulation like so:

$$G_t \stackrel{.}{=} r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \dots$$

Rewrite for brevity:

$$G_t \stackrel{.}{=} \sum_{i=0}^\infty \gamma^{i} r_{i+t+1}$$

Since $$0 \leq \gamma \leq 1$$, this is a telescoping sum. As $$t \to \infty$$, the reward approaches zero. This is what we want, because under this formulation, the expected future reward converges to a finite value, instead of going to infinity.

In practice, we normally discount future rewards with $$\gamma$$ roughly between 0.95 and 0.99, even in episodic tasks. Intuitively, this is because reward now is normally better than reward later. There are cases where a $$\gamma$$ value outside of that range will yield the best performance, but most papers and libraries seem to use a $$\gamma$$ in the above range.

## Agent-Environment Interaction

The MDP formulation is a clear way to frame the problem of learning from interaction to achieve a certain goal. Interaction between an agent and its environment occur in discrete time steps, i.e. $$t = 0, 1, 2, 3, \dots$$. As we know, at every time step our agent takes an action $$a_t$$ and receives a new observation $$s_{t+1}$$ and reward $$r_{t+1}$$. Writing this out directly, the interaction between agent and MDP produces a trajectory that progresses like this:

$$s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, r_3, s_3, a_3, \dots$$

We can denote interaction trajectories by $$\tau$$, the Greek letter Tau.

**Interesting note:** Theoretically, the current state and reward in an MDP should only depend on the previous state and action. In practice, however, this condition (called the [Markov property](https://en.wikipedia.org/wiki/Markov_property)) is violated very regularly. When an RL agent is learning how to control a robot to move forward, the current position of the robot is not only dependent on the previous state and action, but all of the states and actions before it. Somewhat surprisingly, deep RL algorithms are able to achieve excellent performance on these domains despite some problems technically being non-Markovian.

# Policies and Value functions

A policy is a function that maps from states to actions (or to a probability distribution over actions) and is the decision-making part of the agent. A value function estimates how good a particular state, or state-action pair, is. "Good" is defined in terms of expected future reward following that state or state-action pair.

## Policies

The policy is a mapping from states to actions (or to a probability distribution over actions), and can be written like so:

$$\pi : s_t \mapsto a_t$$

Where $$\pi$$ represents the policy, $$s_t$$ represents the current state, and $$a_t$$ is the chosen action to take while in state $$s_t$$.

Policies can be stochastic or deterministic. In the case of a stochastic policy, the output would be a probability distribution over actions. In a deterministic policy, the output is directly what action to take. These can be written mathematically:
- Stochastic policy: $$\pi(a \vert s) = P_{\pi} [A = a \vert S = s]$$
- Deterministic policy: $$\pi(s) = a$$

Let's break this down. In the stochastic policy, $$\pi(a \vert s)$$ is telling us that the output of the policy $$\pi$$ is conditioned on the state $$s$$. $$P_{\pi} [A = a \vert S = s]$$ says that the probability of the action $$a$$ being equal to $$A$$ depends on $$s$$ equaling $$S$$. The deterministic policy simply tells us that the policy $$\pi$$ takes in state $$s$$ and maps to an action $$a$$.

## Value functions

The value of a state is determined by how much reward is expected to follow it. If there's lots of reward expected after state $$s_t$$, then that must be a good state. But, if there is very little reward expected to follow state $$s_{t+5}$$, then that's a bad state to be in. The state-value function is a function that, given a state, outputs an estimate of the expected future return following that state. An action-value function will take in a state and an action and will output an estimate of the expected future return following that state-action pair. Value functions are defined with respect to policies. The value of a state under policy $$\pi$$ is written with $$v_\pi (s)$$; this is the state-value function. Mathematically:

$$v_\pi(s) \stackrel{.}{=} \mathbb{E}_\pi[G_t \vert s_t = s] = \mathbb{E}_\pi[\sum_{i=1}^\infty \gamma^i r_{i+t+1} \vert s_t = s]$$

Recall that $$G_t$$ is the return and $$\sum_{i=1}^\infty \gamma^i r_{i+t+1}$$ is the formula for discounted return. The action-value function is written a bit differently and makes a slightly different assumption than the state-value function. Whereas the state-value function assumes that the policy starts in state $$s$$ and afterward takes all actions according to $$\pi$$, the action-value function assumes that the policy starts in state $$s$$, takes an action $$a$$ (which may or may not be on policy), and thereafter acts following $$\pi$$.

$$q_\pi(s, a) \stackrel{.}{=} \mathbb{E}_\pi [G_t \vert s_t = s, a_t = a] = \mathbb{E}_\pi[\sum_{i=1}^\infty \gamma^i r_{i+t+1} \vert s_t = s, a_t = a]$$

The main distinction to note is that $$v_\pi(s)$$ is dependent only on the current state, while $$q_\pi (s,a)$$ is dependent on both the current state and action.

## Advantage function

The advantage function is found by subtracting the state-value from the state-action value, to get the action value (advantage function). This is often used in practice when training deep RL algorithms.

$$A(s,a) = q_\pi(s,a) - v_\pi(s)$$

See [GAE-lambda advantage estimation](https://arxiv.org/abs/1506.02438) for state-of-the-art with regards to advantage functions.

## Optimal Policies and Value functions

Now that we know about policies and value functions, it's time to see what optimal policies and value functions are.

### Optimal policies

An optimal policy always takes the action in a state $$s$$ that will yield the maximum expected future reward. This can be written like so:

$$\pi^* = \underset{\pi}{argmax} \space G_t$$

where $$\pi^*$$ is the optimal policy. The $$argmax_\pi G_t$$ means "take the action on policy that yields the highest future return". Finding the optimal policy is the central problem in RL, because once the optimal policy is found, the agent can then always take the best action in any state. See [here for more](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#the-rl-problem).

### Optimal value functions

The optimal state-value function (written with $$v^* (s)$$) is the function that always gives you the expected return when starting in state $$s$$ and afterwards acting according to the optimal policy.

$$v^*(s) = \underset{\pi}{max} \space \mathbb{E}_\pi [G_t | s_t = s]$$

The optimal action-value function ($$q^* (s,a)$$) always gives the expected future return if you start in state $$s$$, take action $$a$$ (that may or may not be on-policy) and then afterwards act following the optimal policy.

$$q^*(s,a) = \underset{\pi}{max} \space \mathbb{E}_\pi [G_t | s_t = s, a_t = a]$$

When the optimal policy is in state $$s$$, it chooses the action which maximizes the expected return when starting from the current state. Therefore, when we have the optimal $$q$$ function, the optimal policy is easily found by:

$$a^*(s) = \underset{a}{argmax}\space q^*(s, a)$$

Where $$a^*$$ is the optimal action, and $$\underset{a}{argmax}\space q^*(s, a)$$ means "take the action that maximizes $$q^*$$"

## Bonus: $$\varepsilon$$-greedy algorithms

In RL, we need to trade off between *exploiting* what an agent has already learned and *exploring* the environment and actions to find potentially better actions. When we deal with a greedy policy (one that always chooses the action with the highest expected return), we often must sometimes force such a policy to explore non-greedy actions during training. This is done by picking a random action instead of the on-policy action with some probability $$\varepsilon$$.

# Bellman equations

Bellman equations demonstrate a relationship between the value of a current state and the values of following states. It looks from a current state into the future, averages all future states and the possible actions in those states, and weights each state-action pair by the probability that it will occur. The Bellman equations are a set of equations that break the value function down into the reward in the current state plus the discounted future values.

## State-value Bellman equation

The set of equations for state-value is this:

$$v(s) = \mathbb{E}[G_t \vert s_t = s]$$

$$v(s) = \mathbb{E}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots \vert s_t = s]$$

$$v(s) = \mathbb{E}[r_t + \gamma(r_{t+1} + \gamma r_{t+2} + \dots) \vert s_t = s]$$

$$v(s) = \mathbb{E}[r_t + \gamma G_{t+1} \vert s_t = s]$$

$$v(s) = \mathbb{E}[r_t + \gamma v(s)_{t+1} \vert s_{t} = s]$$

## Action-value Bellman equation

And for action-value:

$$q(s,a) = \mathbb{E}[G_t \vert s_t = s, a_t = a]$$

$$q(s,a) = \mathbb{E}[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots \vert s_t = s, a_t = a]$$

$$q(s,a) = \mathbb{E}[r_t + \gamma(r_{t+1} + \gamma r_{t+2} + \dots) \vert s_t = s, a_t = a]$$

$$q(s,a) = \mathbb{E}[r_t + \gamma G_{t+1} \vert s_t = s, a_t = a]$$

$$q(s,a) = \mathbb{E}[r_t + \gamma \underset{a \sim \pi}{\mathbb{E}}[q(s_{t+1},a)] \vert s_{t} = s, a_t = a]$$

The $$a \sim \pi$$ means "action is sampled from policy $$\pi$$".

## Optimal Bellman Equations

If we don't care about computing the expected future reward when following a policy, then we can use the optimal Bellman equations:

$$v^*(s) = \underset{a}{max} \mathbb{E}[r + \gamma v^*(s_{t+1})]$$

$$q^*(s, a) = \mathbb{E}[r + \gamma \space \underset{a_{t+1}}{max} \space q^*(s_{t+1}, a_{t+1})]$$

The difference between the on-policy and optimal Bellman equations is whether or not the $$max$$ operation is present. When the $$max$$ is there, it means that for the agent to act optimally, it has to take the action that has the maximum expected return (aka highest value).

# Observations and actions

## Observation spaces

An observation is the state, or portion of the state, as it is observed by the agent. In fully-observable MDPs, the state and the observation can be identical, but sometimes are not (for example, in the Atari [DQN paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), the observations to the agent are the most recent few frames of the game, concatenated together), and in partially-observable MDPs, the agent cannot see the entirety of the environment's state. Therefore, the distinction between the *state* of an environment and the *observations* an agent receives is an important one.

Observation spaces can be discretely or continuously valued. An example of a discrete observation space is the layout of a tic-tac-toe board, where empty space is represented by a zero, X pieces are represented by a 1, and O pieces are represented by a 2. A continuous observation space example might be a vector of joint torques and velocities in a robot, like the one in the gif below.

![ant-mujoco-gif](https://openai.com/content/images/2017/05/image4.gif)

In this gif, the goal is to move the ant robot forward. The observations are a 28 dimensional vector of joint torques and velocities. The agent must learn to use these observations to take actions to achieve the goal.

## Action spaces

Similarly to observation spaces, actions can be continuously or discretely valued. A simple example of a discrete action is one in Atari, where you move a joystick in one of four distinct directions to control where your character moves. Continuous actions can be more complex. For example, in the gif above, the actions are a real valued vector of continuous numbers representing amounts of force to apply to the robot's joints. Continuous actions are always a vector of values.

## Reward functions

Reward functions yield a scalar signal at every step of the environment telling the agent how good the previous action was. The agent uses these signals to learn to take good actions in every state. To the agent, goodness is defined by how much reward was earned.

Reward functions can either be simple or complex. For a simple example, in the classic gridworld environment (see diagram below), the agent starts in one corner of a grid and must navigate an end state in the other corner of the grid. The reward at every step, no matter the action, is $$-1$$. This incentivizes the agent to navigate from start to end as quickly as possible.

![gridworld-environment](https://miro.medium.com/max/1454/1*G3q-q9gEiDc2fD8sPXHBpQ.png)

In the above image, the greyed out squares are the starting and ending points. Either one can be the start or end point. A more complicated example of a reward function would be the one for the ant robot above. The reward function for that agent is as follows:

$$r_t(s, a) = \frac{x' - x}{\Delta t} - \frac{\sum_{i=1}^{N} \vec{a}^2}{2} - \frac{1*10^{-3} * \sum_{i=1}^{N} (clip(\vec{c_E}, -1, 1))^2}{2}$$

Let's unpack this. $$r_t(s, a)$$ tells us that the reward is a function of the last state and action. $$x$$ and $$x'$$ are the $$x$$ position of the robot before and after the action, respectively. $$\Delta t$$ is the change in time before the last action and current action. $$\vec{a}$$ is the action vector, in this case the actions are a vector because we're applying forces to 8 different joints on the robot, so each entry in the vector tells us how much force to apply to the corresponding joint. $$C_E$$ is a vector of external forces on the body of the robot. $$clip$$ tells us to clip the vector $$\vec{c_E}$$ to have all values fall within the range $$[-1, 1]$$.

# Some classical methods

Here, we'll briefly touch on some classical methods in reinforcement learning. These methods aren't in use on cutting-edge problems today, but do lay an important theoretical foundation for modern algorithms.

## Dynamic Programming

Dynamic programming (DP) algorithms require a perfect model of the environment. Practically, this is often unrealistic to expect and therefore DP algorithms often are not actually used in RL. However, they are still theoretically important. We can use DP to find optimal value functions, and from there, optimal policies.

### Policy Evaluation

We can compute the state-value function for a policy $$\pi$$ by using policy evaluation. Mathematically:

$$v_{t+1}(s) \stackrel{.}{=} \mathbb{E}_\pi [r_t + \gamma \space v_t(s_{t+1}) \vert s_t = s] = \sum_a \pi(a \vert s) \space \sum_{s_{t+1}, r} p(s_{t+1}, r \vert s, a)[r + \gamma \space v_k(s_{t+1})]$$

where $$\underset{a}{\sum}$$ means sum over actions, and other symbols should be known from earlier sections.

### Policy Improvement

Policy improvement finds a new and improved policy $$\pi' \geq \pi$$ by greedily selecting the action with highest value in each state.

$$q_\pi(s,a) \stackrel {.}{=} \mathbb{E}_\pi [r_t + \gamma \space v_\pi(s_{t+1}) \vert s_t = s, a_t = a] = \sum_a \pi(a \vert s) \sum_{s_{t+1}, r} p(s_{t+1}, r \vert s, a)[r + \gamma \space v_k(s_{t+1})]$$

### Policy Iteration

Once we've used $$v_\pi(s)$$ to improve $$\pi$$ and get $$\pi'$$, we can again perform a policy evaluation step (compute the state value function for $$\pi'$$) and a policy improvement step (compute the improved poicy $$\pi'' \geq \pi'$$). By continuing this cycle we can get consistently improving policies and value functions. The trajectory of policy iteration looks like this ($$e$$ is for policy evaluation and $$i$$ is for policy improvement):

$$\pi_0 \xrightarrow[]{\text{e}} v_{\pi_0} \xrightarrow[]{\text{i}} \pi_1 \xrightarrow[]{\text{e}} v_{\pi_1} \xrightarrow[]{\text{i}} \pi_2 \xrightarrow[]{\text{e}} \dotsb \xrightarrow[]{\text{i}} \pi^* \xrightarrow[]{\text{e}} \space v^*$$

## Monte Carlo Methods

In the previous section, we discussed policy iteration for deterministic policies. The math and theory described there extends to stochastic policies too. Monte Carlo (MC) methods do not require a model of the environment and instead can learn entirely from experience. The core idea of MC methods is to use average *observed* return as an approximation for the value of a state. To get an empirical return, MC methods need complete episodes and the episodes have to end. We can estimate $$v_\pi(s)$$ under first visit or every visit MC. First visit MC estimates the value of $$s$$ as an average of the return after the first visit to $$s$$, while every visit MC estimates the value by averaging the return of $$s$$ every time the state is visited. Both first and every visit MC converge to the true value of the state as the number of visits goes to infinity.

![mc-policy-iteration](https://lilianweng.github.io/lil-log/assets/images/MC_control.png)

*Image from Lilian Weng's blog. Showing that learning an optimal policy via MC is done by following a similar idea to policy iteration.*

Similarly to policy iteration, we improve the policy greedily with respect to the current value function.

$$\pi(s) = \underset{a}{argmax} \space q_\pi(s,a)$$

Then, we use the updated policy to generate a new episode to train on. Finally, we estimate the $$q$$ function using information gathered from the episode.

## Temporal Difference Learning

Temporal Difference (TD) learning combines ideas from Monte Carlo and Dynamic Programming methods. Like MC methods, TD doesn't require a model of the environment and instead learns only from experience. TD methods update value estimates based partially on other estimates that have already been learned and they can learn from incomplete episodes.

### Bootstrapping

TD methods use existing estimates to update values instead of only relying on empirical and complete returns like MC methods do. This is known as bootstrapping.

### TD value estimation

Similarly to MC methods, TD methods use experience to estimate values. Both follow a policy $$\pi$$ and collect experience over episodes. Both TD and MC methods update their value estimates for every nonterminal state in the set of gathered experience. A difference is that MC methods wait until the end of an episode, when empirical return is known, to update their value estimates, whereas TD methods can update their estimates with respect to other estimates (they don't rely on empirical return) during an episode. We can write a simple version of an MC method that works well in nonstationary environments:

$$v(s_t) \leftarrow v(s_t) + \alpha [G_t - v(s_t)]$$

It is simple to turn this MC update into a TD update by switching out the empirical return $$G_t$$ for the current value estimate of the next state $$v(s_{t+1})$$:

$$v(s_t) \leftarrow v(s_t) + \alpha [r_t + \gamma v(s_{t+1}) - v(s_t)]$$

$$\alpha$$ is a step-size parameter where $$\alpha \in [0, 1]$$.

This update can also be written for the action-value function:

$$q(s_t, a_t) \leftarrow  q(s_t, a_t) + \alpha [r_t +\gamma q(s_{t+1}, a_{t+1}) - q(s_t, a_t)]$$

Learning an optimal policy using TD learning is called TD control. We'll next look at two algorithms for TD control.

### SARSA: On-policy TD control

We again follow the pattern of policy iteration, except now we use TD methods for the evaluation (value estimation) steps. In SARSA, we need to learn a $$q$$ function and then define a greedy (or $$\varepsilon$$-greedy) policy with respect to that $$q$$ function. This can be done using the $$q$$ update rule from above:

$$q(s_t, a_t) \leftarrow  q(s_t, a_t) + \alpha [r_t +\gamma q(s_{t+1}, a_{t+1}) - q(s_t, a_t)]$$

This update is performed after every nonterminal state. If $$s_t$$ is terminal, then $$q(s_{t+1}, a_{t+1})$$ is 0. This rule uses each element in the tuple: $$(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$$. This tuple also gives the SARSA algorithm its name. SARSA algorithm has these steps:

1. From $$s_t$$, pick an action according to the current $$q$$ function, often $$\varepsilon$$-greedily: $$a_t = \underset{a}{argmax} q(s_t, a)$$
2. Our selected action, $$a_t$$ is applied to the environment, the agent gets reward $$r_t$$, and the environment steps to a new state $$s_{t+1}$$
3. Pick next action same way as in step one
4. Do the action-value function update: $$q(s_t, a_t) \leftarrow  q(s_t, a_t) + \alpha [r_t +\gamma q(s_{t+1}, a_{t+1}) - q(s_t, a_t)]$$
5. Time steps forward and the algorithm repeats from the first step

### Q-learning: Off-policy TD control

Q-learning was an early breakthrough in reinforcement learning by [Watkins and Dyan in 1989](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf). The algorithms update rule is:

$$q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha[r_t + \gamma \space \underset{a}{max} \space q(s_{t+1}, a_{t+1}) - q(s_t, a_t)]$$

Under this rule, $$q$$ directly approximates the optimal action-value function $$q^*$$, independent of the current policy.

The Q-learning algorithm has these steps:

1. Start in $$s_t$$ and pick an action according to the policy defined by the $$q$$ function. Could be a $$\varepsilon$$-greedy policy.
2. Take action $$a$$, gather $$r_t$$ and step the environment to the next state $$s_{t+1}$$.
3. Apply the update rule: $$q(s_t, a_t) \leftarrow q(s_t, a_t) + \alpha[r_t + \gamma \space \underset{a}{max} \space q(s_{t+1}, a_{t+1}) - q(s_t, a_t)]$$
4. Time steps forward to the new state and algorithm repeats from the first step.

Thank you for sticking with me through this long blog post. I really hope it was worth your while and should you find any errors, please [email me](mailto:jfpettit@gmail.com). Please feel free to have a discussion or raise questions in the comments. In my next post, I'm planning to write about policy gradient methods and dive into the theory behind them. See you then!

# References
- [Lilian Weng's blog post on reinforcement learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
- [Sutton and Barto's RL Book](http://incompleteideas.net/book/the-book.html)
- [My undergraduate thesis on RL fundamentals](https://github.com/jfpettit/senior-practicum/blob/master/PracticumPaper.pdf)
- [Markov property](https://en.wikipedia.org/wiki/Markov_property)
- [High-Dimensional Continuous Control using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [OpenAI's SpinningUp course](https://spinningup.openai.com/en/latest/)
- [Mnih et al. paper on Human Level Control through Deep Reinforcement Learning (Atari paper)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Q-learning: Watkins and Dyan](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)
