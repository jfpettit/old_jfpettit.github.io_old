# About Me

## Current work

I recently finished reading Sutton and Barto's "Reinforcement Learning: An Introduction", and I wrote from-scratch implementations of algorithms I found most interesting. Read the book [here.](http://incompleteideas.net/book/the-book.html) Now, I'm working on Applied Machine Learning at Lawrence Livermore National Laboratory.

## Hopeful future work

I'm curious about improving generalization in reinforcement learning(RL), working on improving sample complexity in RL, single-shot and few-shot learning, knowledge representation, and learning theory in general.

## Contact Me

I can be reached via [email](mailto: jfpettit@gmail.com) or you can find me on [twitter](https://twitter.com/jacobpettit18).

## Projects

### gym-snake-rl

After reading OpenAI's [Requests for Research 2.0](https://openai.com/blog/requests-for-research-2/), I got inspired and wanted to contribute however I could. I was already interested in working on generalization and knowledge transfer in RL, so naturally I was curious about doing experiments in this area, and I wanted to be able to do some on my laptop. I quickly realized that there weren't really any environments with randomly generated maps that were also computationally inexpensive, so I thought I could try to help fill this gap. Read my blog post about this project [here](https://jfpettit.svbtle.com/introducing-gym-snake-rl).

### Senior Practicum in Reinforcement Learning

To complete the Bachelor of Science degree in Computational Science at FSU, each senior has to complete a semester-long senior project. I spent my project investigating the fundamentals of Reinforcement Learning and familiarizing myself with them. I wrote implementations of a few different algorithms from scratch as part of this work, and my final product for the semester was a paper, linked [here](https://github.com/jfpettit/senior-practicum/blob/master/PracticumPaper.pdf). You can view all of my (unedited and messy) code from the project at [this repository](https://github.com/jfpettit/senior-practicum).

#### Dynamic Programming Policy Iteration in Gridworld

This is an implementation of the Gridworld exercise from Chapter 4 of Reinforcement Learning: An Introduction. Credit for the gridworld environment goes to [Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py). Gridworld is a simple N by N grid environment where the agent is randomly initialized on a square and must navigate to a terminal square. The reward for every step the agent takes in trying to reach the terminal state is -1, so the agent is incentivized to terminate as quickly as possible. The code I wrote performs policy iteration and outputs estimates of an optimal policy and value function. It also has a code cell that'll render an agent moving through gridworld. 

#### Temporal Difference Learning tic-tac-toe agent

This code was written based on section 5 in Chapter 1 of Reinforcement Learning: An Introduction, where they discuss creating a temporal difference agent for tic-tac-toe playing. My code is based largely off of that example.  

#### Blackjack Monte Carlo

The code here employs Monte Carlo to learn both an action-value function and a policy for playing Blackjack against a deterministic dealer. The dealer follows a stationary policy, so it is very much a toy version of the full Blackjack problem. 

#### Q-learning

In this code I implement a tabular Q-learning algorithm for the Gridworld-CliffWalking environment described in Section 6.5 of the Reinforcement Learning: An Introduction textbook. 

### Course Projects

#### Hidden Markov Model

In the Applied Machine Learning (course code STA 5635) class at FSU, one assignment was to implement a Hidden Markov Model and use it to solve a given problem. When developing this code, one of my goals was for it to be flexible, so that it could be used for many problems. [Nathaniel Leonard](https://github.com/NateAnthonyLeonard) also contributed. Our code is [here.](https://github.com/jfpettit/machine-learning/tree/master/hidden-markov-model)

#### Multiclass Logistic Regression with Feature Selection with Annealing (FSA)

[FSA paper](https://arxiv.org/pdf/1310.2880.pdf)

This was also an assignment in the Applied Machine Learning course. Again, I strove for enough flexibility that the code can be used generally. This is a multiclass logistic regression that also makes use of Feature Selection with Annealing to reduce the number of model parameters throughout training. I've linked to the Feature Selection with Annealing paper above.  [Nathaniel Leonard](https://github.com/NateAnthonyLeonard) and I worked together. The code for this project is [here.](https://github.com/jfpettit/machine-learning/tree/master/multiclass-logreg)
