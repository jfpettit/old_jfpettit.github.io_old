---
layout: project_page
title: Projects
---

<h1>Projects</h1>


<h2 class='post-title'><a class='post-title' href="https://github.com/jfpettit/flare">flare</a></h2>

I developed my own reinforcement learning framework with goals of simplicity, ease of understanding, beginner friendliness, and performance. So far, I've implemented a couple of algorithms and in the future will extend the framework to include more. My old blog post on ```flare``` is [here](https://jfpettit.svbtle.com/rlpack) and the GitHub repository is [here](https://github.com/jfpettit/flare). I will write an updated blog post on ```flare``` in the near future.

<h2 class='post-title'><a class='post-title' href="https://github.com/jfpettit/gym-snake-rl">gym-snake-rl</a></h2>

When reading OpenAI's [Requests for Research 2.0](https://openai.com/blog/requests-for-research-2/), I got inspired by the point suggesting that people work on a multi-agent version of the classic [Snake game](https://www.coolmathgames.com/0-snake) and had already been interested in working on generalization in reinforcement learning. I created a custom OpenAI [Gym](https://gym.openai.com/) environment to try and tackle both of those things. An in-depth blog post about the environment is [here](https://jfpettit.svbtle.com/introducing-gym-snake-rl) and the code is on [GitHub](https://github.com/jfpettit/gym-snake-rl).

<h2 class='post-title'><a class='post-title' href="https://github.com/jfpettit/senior-practicum">Senior Practicum</a></h2>

To complete the Bachelor of Science degree in Computational Science at FSU, each senior has to complete a semester-long senior project. I spent my project investigating the fundamentals of Reinforcement Learning and familiarizing myself with them. I wrote implementations of a few different algorithms from scratch as part of this work, and my final product for the semester was a paper, linked [here](https://github.com/jfpettit/senior-practicum/blob/master/PracticumPaper.pdf).

<h2 class='post-title'><a class='post-title' href="https://github.com/jfpettit/reinforcement-learning#policy-iteration-with-dynamic-programming-to-solve-gridworld">Dynamic Programming Policy Iteration in Gridworld</a></h2>

I wrote this code as a part of my senior research project in reinforcement learning. Chapter 4 of "Reinforcement Learning: An Introduction" discusses dynamic programming and using it to perform policy iteration on gridworld. Credit for the gridworld environment goes to [Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py). Gridworld is a simple N by N grid environment where the agent is randomly initialized on a square and must navigate to a terminal square. The reward for every step the agent takes in trying to reach the terminal state is -1, so the agent is incentivized to terminate as quickly as possible. The code I wrote performs policy iteration and outputs estimates of an optimal policy and value function. It also has a code cell that'll render an agent moving through gridworld. My code is [here](https://github.com/jfpettit/reinforcement-learning#policy-iteration-with-dynamic-programming-to-solve-gridworld).

<h2 class='post-title'><a class='post-title' href="https://github.com/jfpettit/senior-practicum">tic-tac-toe agent</a></h2>

This code was written as part of my senior research project on reinforcement learning. In section 5 in Chapter 1 of Sutton and Barto's ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/the-book.html), they discuss creating a temporal difference agent for tic-tac-toe playing. My code is based largely off of that example. You can find my code [here](https://github.com/jfpettit/senior-practicum). See how to easily play against it yourself [here](https://jfpettit.svbtle.com/making-it-easier-to-play-my-tic-tac-toe-agent).

<h2 class='post-title'><a class='post-title' href="https://github.com/jfpettit/machine-learning/tree/master/hidden-markov-model">Hidden Markov Model</a></h2>

In the Applied Machine Learning class at FSU, one assignment was to implement a Hidden Markov Model and use it to solve a given problem. When developing this code, one of my goals was for it to be flexible, so that it could be used for many problems. [Nathaniel Leonard](https://github.com/NateAnthonyLeonard) also contributed. Our code is [here](https://github.com/jfpettit/machine-learning/tree/master/hidden-markov-model).

<h2 class='post-title'><a class='post-title' href="https://github.com/jfpettit/machine-learning/tree/master/multiclass-logreg">Multiclass Logistic Regression with FSA</a></h2>

[FSA (Feature Selection with Annealing) paper.](https://arxiv.org/abs/1310.2880)

This was also an assignment in the Applied Machine Learning course. Again, I strove for enough flexibility that the code can be used generally. This is a multiclass logistic regression that also makes use of Feature Selection with Annealing to reduce the number of model parameters throughout training. I've linked to the Feature Selection with Annealing paper above.  [Nathaniel Leonard](https://github.com/NateAnthonyLeonard) and I worked together. The code for this project is [here](https://github.com/jfpettit/machine-learning/tree/master/multiclass-logreg).
