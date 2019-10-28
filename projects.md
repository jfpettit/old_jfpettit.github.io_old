# Projects

## [rlpack](https://github.com/jfpettit/rl-pack)

I developed my own reinforcement learning framework with goals of sipmlicity, ease of understanding, and beginner friendliness. So far, I've implemented a couple of algorithms and in the future will extend the framework to include more. My blog post on rlpack is [here](https://jfpettit.svbtle.com/rlpack) and the GitHub repository for rlpack is [here](https://github.com/jfpettit/rl-pack).

## [gym-snake-rl](https://github.com/jfpettit/gym-snake-rl)

When reading OpenAI's [Requests for Research 2.0](https://openai.com/blog/requests-for-research-2/), I got inspired by the point suggesting that people work on a multi-agent version of the classic [Snake game](https://www.coolmathgames.com/0-snake) and had already been interested in working on generalization in reinforcement learning. I created a custom OpenAI [Gym](https://gym.openai.com/) environment to try and tackle both of those things. An in-depth blog post about the environment is [here](https://jfpettit.svbtle.com/introducing-gym-snake-rl) and the code is on [GitHub](https://github.com/jfpettit/gym-snake-rl).

## [Senior Practicum](https://github.com/jfpettit/senior-practicum)

To complete the Bachelor of Science degree in Computational Science at FSU, each senior has to complete a semester-long senior project. I spent my project investigating the fundamentals of Reinforcement Learning and familiarizing myself with them. I wrote implementations of a few different algorithms from scratch as part of this work, and my final product for the semester was a paper, linked [here](https://github.com/jfpettit/jfpettit.github.io/blob/master/PracticumPaper.pdf).

## [Dynamic Programming Policy Iteration in Gridworld](https://github.com/jfpettit/reinforcement-learning#policy-iteration-with-dynamic-programming-to-solve-gridworld)

I wrote this code as a part of my senior research project in reinforcement learning. Chapter 4 of "Reinforcement Learning: An Introduction" discusses dynamic programming and using it to perform policy iteration on gridworld. Credit for the gridworld environment goes to [Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/gridworld.py). Gridworld is a simple N by N grid environment where the agent is randomly initialized on a square and must navigate to a terminal square. The reward for every step the agent takes in trying to reach the terminal state is -1, so the agent is incentivized to terminate as quickly as possible. The code I wrote performs policy iteration and outputs estimates of an optimal policy and value function. It also has a code cell that'll render an agent moving through gridworld. My code is [here](https://github.com/jfpettit/reinforcement-learning#policy-iteration-with-dynamic-programming-to-solve-gridworld).

## [tic-tac-toe agent](https://github.com/jfpettit/senior-practicum)

This code was written as part of my senior research project on reinforcement learning. In section 5 in Chapter 1 of Sutton and Barto's ["Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/the-book.html), they discuss creating a temporal difference agent for tic-tac-toe playing. My code is based largely off of that example. You can find my code [here](https://github.com/jfpettit/senior-practicum). See how to easily play against it yourself [here](https://jfpettit.svbtle.com/making-it-easier-to-play-my-tic-tac-toe-agent).

## [Hidden Markov Model](https://github.com/jfpettit/machine-learning/tree/master/hidden-markov-model)

In the Applied Machine Learning class at FSU, one assignment was to implement a Hidden Markov Model and use it to solve a given problem. When developing this code, one of my goals was for it to be flexible, so that it could be used for many problems. [Nathaniel Leonard](https://github.com/NateAnthonyLeonard) also contributed. Our code is [here](https://github.com/jfpettit/machine-learning/tree/master/hidden-markov-model).

## [Multiclass Logistic Regression with FSA](https://github.com/jfpettit/machine-learning/tree/master/multiclass-logreg)

[FSA (Feature Selection with Annealing) paper.](https://arxiv.org/abs/1310.2880)

This was also an assignment in the Applied Machine Learning course. Again, I strove for enough flexibility that the code can be used generally. This is a multiclass logistic regression that also makes use of Feature Selection with Annealing to reduce the number of model parameters throughout training. I've linked to the Feature Selection with Annealing paper above.  [Nathaniel Leonard](https://github.com/NateAnthonyLeonard) and I worked together. The code for this project is [here](https://github.com/jfpettit/machine-learning/tree/master/multiclass-logreg).
