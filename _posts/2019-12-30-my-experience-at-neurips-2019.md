---
layout: post
author: Jacob Pettit
comments: true
tags: conferences
---

**Last year I was fortunate enough to attend NeurIPS 2019. It was an amazing experience, I was able to meet lots of smart people and learned a ton. This post discusses my time at NeurIPS 2019**

This December, I was lucky enough to be able to go to my first [NeurIPS](https://neurips.cc/) and present my [work](https://arxiv.org/abs/1912.03408) at the workshop [Tackling Climate Change with AI](https://www.climatechange.ai/NeurIPS2019_workshop.html). While it was exciting to be able to present my first paper at such a big workshop (and to give a [spotlight talk](https://slideslive.com/38922109/tackling-climate-change-with-ml-4), my talk starts at about the 33:30 mark), the real highlights were getting to hear about the amazing work being done by others in the field and networking with people. And I got to meet [Richard Sutton](http://www.incompleteideas.net/), the godfather of Reinforcement Learning :open_mouth:! (Richard is the guy in the blue button down shirt in the photo)


<img src='/assets/imgs/IMG_1174.JPG' style='width:800px; height:600px'>

{: class="table-of-content"}
* TOC
{:toc}

# Conference Talks!

In no particular order, the talks I especially enjoyed are: 
- [Josh Tobin's](http://josh-tobin.com/) talk on [Geometry-Aware Neural Rendering](josh-tobin.com/assets/pdf/geometry_aware_neural_rendering_vNeurIPS.pdf)
- [Pieter Abbeel's](https://people.eecs.berkeley.edu/~pabbeel/) talk on [Better Model-Based RL through Meta-RL](https://slideslive.com/38922013/metalearning-2)
- [Igor Mordatch's](https://openai.com/blog/authors/igor/) [discussion](https://slideslive.com/38921888/biological-and-artificial-reinforcement-learning-3) of [Multi-agent interaction](https://openai.com/blog/emergent-tool-use/) and online optimization in RL
- [Richard Sutton's](http://incompleteideas.net/) overview of next steps [Towards a General AI-Agent Architecture](https://slideslive.com/38921889/biological-and-artificial-reinforcement-learning-4) (talk starts around the 15 minute mark)
- [David Ha's](http://otoro.net/ml/) talk about [Innate Bodies, Innate Brains, and Innate World Models](https://slideslive.com/38922072/learning-transferable-skills-3)

Please send me an email or leave a comment if you find some error or find a link relevant to the talks that you feel should be included.

I took some notes during each of the above talks. I'll try to make sense of them below. My notes for some of the talks are sparse or end early. In this case, I tried to fill in the blanks but if you have the time and are interested enough, I strongly recommend watching the full talks, or at least looking through the slides.

## Geometry Aware Neural Rendering

The problem being addressed here is that robots need to be able to understand the world in a robust way in order to be able to act reliably. Learning a policy to directly map from states to actions works for simple tasks, but starts to break down when tasks become complex. We want to use neural rendering to model the state of the world implicitly.

The work done here extends the [Generative Query Network (GQN)](https://deepmind.com/blog/article/neural-scene-representation-and-rendering) to:
- higher dimensions in the images
- objects with greater degrees of freedom
- more realistic objects

They created an attention mechanism, called Epipolar Attention, to improve upon the original GQN. The talk concluded with the statement that: "Geometrically inspired neural network primitives improve implicit 3D understanding."

## Interaction of Model-Based RL and Meta RL

A big theme in this talk was how to make RL bridge the gap between RL's horrible sample inefficiency and human's comparably sample-efficient learning. Very quickly Pieter pointed out that Humans after 15 minutes of experience on an Atari typically outperform [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) after 115 hours of training. Humans are able to use their past experience to quickly pick up new tasks. I.e., if I am an athlete who plays hockey, maybe I can use many of the skills I learned from hockey to quickly perform well at lacrosse. We'd like to make machine learning algorithms also use their past experience to quickly pick up new skills. Enter meta-RL:

### Meta RL (Super Briefly)

Typically the agent is an RNN, so that there is some memory of performing past tasks, which will be leveraged to quickly pick up new tasks. Different activations in the RNN means that the current policy is different from the last policy. And the meta training goal can be optimized using an existing reinforcement learning algorithm.

### Domain Randomization

A major fallacy of training in simulation is that we cannot create a single simulation that will reliably and exactly recreate the physics and interactions that a robot will encounter in the real world. One way to get around this is via domain randomization. This technique randomizes many aspects of the simulation, such as the coefficient of friction, colors and sizes of objects in the simulation, and so on, and trains a policy across these randomized simulations. A slide on Pieter Abbeel's presentation read: "If the RL model sees enough simulated variations, the real world may look like just the next simulator." In this case, the model would be able to perform the task in the real world, since it would have learned how to perform the task in a robust way that worked across all simulations. They used domain randomization for robot grasping by randomizing the structure of the objects in simulation and were able to show that a policy trained in simulation also worked in the real world.

### Better model-based RL through meta-RL

In model based RL (MBRL), a poliy interacts with the real world and then is updated. Using these collected interactions, a learned simulator is trained to model the environment. Then, the policy is improved by interacting with the learned simulator (not with the real environment).

Overfitting happens in MBRL because policy optimization wants to exploit the regions of the learned simulator where there hasn't been enough data collected for the learned simulator to accurately model the environment. This leads to massive failures.

In an alternative approach, [Model based Meta Policy Optimization (MBMPO)](https://arxiv.org/abs/1809.05214), interaction data is collected under an adaptive policy. Then, an ensemble of $$k$$ simulators is learned from the collected interaction data. Following this, meta-policy optimization is done over the ensemble and a new meta policy and new adaptive policies are collected from the optimization update. In this way, the authors could meta-learn a policy that could adapt quickly.

A couple of points from the talk: 
- **Asynchronous training for MBRL is more effective than synchronous training. Asynchronous learns faster.**
- **Meta-training has higher sample complexity than regular RL training.**

The talk went on and covered [Guided Meta-Policy Search](https://arxiv.org/abs/1904.00956) but I do not have any good notes on that.

## Multi-Agent Interaction and Online Optimization in RL

Igor prefaced this talk with saying that we've seen good AI progress in specific and well-defined tasks, but that we still don't have the ability to move to complex and varied tasks. He proposes that multi-agent interaction can be a tool to move towards this because:
- Applications of AI are often social (multi-agent). i.e. language, learning and teaching are all social things.
- We'd like to achieve beneficial and robust mult-robot and human-robot interaction.
- A social AI should be able to explain itself and understand how to act in a safe way.

Such a system could be deployed in a self-supervised way for continuous learning.

### Multi-agent interaction

If we mix:
- A rudimentary physical simulation environment
- A game such as hide-and-seek
- PPO

This happens!
<iframe width="560" height="315" src="https://www.youtube.com/embed/kopoLzvh5jY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In the hide-and-seek environment agents are given a team based reward so that collaboration is encouraged. The hiders are also given a preparatory phase where the seekers are frozen in place and cannot move. This preparation phase allows the hiders to construct their shelter. 

They saw many unexpected behaviors from the agents during experiments in this work. 
- Seekers box surfing to break into shelters
- Hiders endlessly running away from the map
- Hiders using a bug in the physics to push the ramp out of the map
- Seekers using the ramp and a bug in the physics to jump out over the map and land into the hiders shelter

In [the blog post](https://openai.com/blog/emergent-tool-use/) they discuss testing intrinsic motivation compared to multi-agent competition and show that their method outperforms an intrinsic motivation scheme. It looks like they only compare to count-based motivation, and while I'm not an expert in intrinsic motivation, it appears that there [are](https://arxiv.org/abs/1802.07442) [other](https://arxiv.org/abs/1907.03116) [methods](http://www.cs.cornell.edu/~helou/IMRL.pdf) that might perform better.

Measuring progress in the hide-and-seek environment was a big challenge for them in this work, and they used "intelligence tests" to measure progress of the agents as they learned! 

### Online Optimization

The second part of Igor's talk focused on this topic and seemed to discuss some work relevant to the [POLO paper](https://arxiv.org/abs/1811.01848). 

So far in RL we've focused on learning habitual, reflexive behaviors. These are hard to generalize or to improvise with. We'd like to move more towards learning through some feedback-guided exploration. 

#### Online model-predictive control

Under this paradigm, we can:
- Act from little experience (better sample complexity)
- Learn in a continual and reset-free setting
    - Therefore, adapt to changes in the world without predicting them
- In nonstationary or changing environments anything you don't model is nonstationary

#### Energy-based generative models

1. Implicit generation
    - Only one object to learn, an energy network
    - Feedback-guided processing
2. Contrastive loss
    - Adversarially probes the model, beneficial for robustness

Some surprising benefits of energy-based models are that they are robust, generalize well, and learn continuously.

This is as far as I got with my notes. Please see [Igor's talk](https://slideslive.com/38921888/biological-and-artificial-reinforcement-learning-3) for more!

## Towards a General AI-Agent Architecture

Richard Sutton laid out the premises for his talk:
- Seek general principles for learning to predict and control experiences
- Interested in the domain-dependent part of intelligence
- The reward goal of RL is adequate
- Need function approximation, partial observability, temporal abstraction, and non-stationarity
    - Even if the world were stationary, approximations of the world should not be stationary
- All learning is based on online TD prediction
- Need to build specific general mental components:
    - State features
    - Subproblems
    - Solutions to subproblems (options)
    - World model in terms of these things (for planning)

These are basic needs but obtaining all of them is unprecedented.

We need algorithms for constructing state features by learning non-linear recurrent state update functions.

### Planning

- [Dyna](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=711FEF6BA26BBF98C28BC111B26F8761?doi=10.1.1.48.6005&rep=rep1&type=pdf)
    - Planning and learning are very similar
- Dynamic programming style planning
    - Planning happens by using a model to imagine something about the future
    - Each 'imagining' from a state-action pair forward is called a lookahead
- What should the output of a lookahead be?
    - A distribution over states?
        - What if our states are real-valued vectors?
    - A sample state from a distribution?
    - An expected state feature vector?
        - This can't be rolled out

**Theorem**: If the approximation function is linear, value iteration with a distribution model learns nothing if an expectation model is used instead.

The value function should be linear in state features!

### Subproblems and Play

Currently, core RL learns:
- Value functions
- Policies

We *need* to learn:
- State features
- Skills
- World models

Subproblems can help solve the main problem by:
- Shaping the state representation
- Shaping behavior
- Enabling planning at a higher level

My notes at this point end. Sorry this section was kind of sparse, but please watch [Sutton's talk](https://slideslive.com/38921889/biological-and-artificial-reinforcement-learning-4) for more detail! :smiley:

## Innate Bodies, Innate Brains, and Innate World Models

In this talk, David discussed his work on [Weight Agnostic Neural Networks](https://weightagnostic.github.io/) and on [Learning to Predict without Looking Ahead](https://learningtopredict.github.io/).

### Weight Agnostic Neural Networks

- Much animal behavior is innate (not learned)
- Animal brains aren't blank slates at birth (as imagined by many AI researchers)
    - Animals are born with lots of skills built in, like avoiding predators and finding food
- Hand-designed components have strong inductive priors
- Random search on networks can already get near state of the art
- Model predictive controllers have widespread applications
- Despite appearances, it seems that much of human intelligence is built on environment-specific tricks and strategies

### Learning to Predict Without Looking Ahead

- This work was sort of like training a blindfolded agent to drive.
    - They only let an agent see observations every once in a while, when there weren't observations, a predictive model had to do the work of imaginging what was happen in the environment and provide those imaginings as observations. 
    - They did train this model on the car racing task... So the model did basically drive blindfolded.
    - The world model (predictive part) would realign with reality whenever it was given another observation.
    - The controller (part steering the car) was only given real observations about 5% of the time.
        - 95% of the time it is driving blindfolded! By "blindfolded" I mean that it sees the outputs of the world model instead of observations.
- The policy learned inside of the dream world transfers well to the actual environment.
    - Perhaps the dream world is harder than the real environment is?

That's the end of my notes here. Sorry the first half is sparse, and that it's all short, but again, check out [David's talk](https://slideslive.com/38922072/learning-transferable-skills-3) for more.

# Things I learned from this NeurIPS

I had an amazing time at the conference and saw some fascinating work. Luckily, I had a friend going with me who had been twice before and gave me a little bit of advice before we went. Mainly:
- Don't try to see everything - you'll tire yourself out on the first day
    - Plan what you want to see
- Go to some social events
- Go to the Expo, talk to the sponsors, get free stuff!
- Email someone you *really* want to meet with and talk to and pick their brain - maybe they'll say yes!

I'm happy to say that I took pretty much all of those pieces of advice. I was very careful to pace myself and see only what I really wanted to see, and I'm really glad I did. Even after doing this, I was still really tired every night. Also, being selective about time spent at the conference gave me slightly more opportunity to experience Vancouver (it's a really cool city with good food and coffee).

I went to a couple of social events. This is one part of my NeurIPS I'd do differently next time. The only events I really went to were a New in Machine Learning meetup (which was lots of fun) and the Reinforcement Learning Social, which is where I met Richard Sutton. Next time, I'll try to go to one or two of the company parties as well. The sponsor parties at NeurIPS are famous, and it seems for good reason. Some people that went to these parties posted photos of them on Twitter, and they looked very fun. 

The Expo was a really cool part of the conference but it was also very overwhelming. There is opportunity to engage with brilliant researchers at many prestigious companies, which is very exciting, but there is also just a lot of people crammed into the Expo room almost all the time. I enjoyed going and talking to folks at different companies (and getting free stuff!) and will do it again next time. One thing I did notice was that by the end of the Expo (Wednesday) all of the people manning the sponsor booths are tired and less congenial than earlier in the week. I can't blame them - the Expo was crazy the whole time it was happening - but it is worth remembering to make sure to go to the Expo early in the week in the future.

Fortunately for me, I emailed two people I really wanted to meet and one of them agreed to have breakfast with me! The other didn't, but I also waited until the very last minute (the day before the conference began) to reach out. Having breakfast with this person was an excellent experience, and they were friendly, expressed interest in my work, and let me pick their brain. I would definitely recommend doing this to anyone attending NeurIPS or any other large machine learning conference. The worst that can happen is they can say no to meeting you, but if they say yes then you could make a new friend and have a positive interaction with someone. :blush:

Trying to keep up with all of the new work is already like trying to drink from a fire hose. Trying to do it at NeurIPS is like trying to drink from 100 fire hoses. So, I've learned to be very selective about the work that I put effort into *really* understanding during the conference. I saw a [tweet](https://twitter.com/colinraffel/status/1203756410426626049) from [Colin Raffel](https://twitter.com/colinraffel) saying that you shouldn't try to deeply understand more than one new paper a day during NeurIPS, and I think that's great advice.

I've concluded that I agree with all of the people who say that the most important thing at NeurIPS isn't all of the new work being presented (even though that is important), but rather the opportunity to interact with so many brilliant ML researchers in one place. One can go back and read papers on arXiv at any time, but only a couple of times a year do large conferences like this happen, and the networking opportunity here is just too massive to spend all of your time catching up on the new work. I made some new connections, but not as many as I could have, and next time I'll work to more effectively split my time between seeing new work and connecting with all of the interesting people. :+1:

# Conclusion

NeurIPS 2019 was an overwhelmingly positive experience for me and I'm extremely excited to go back. Everyone that I met was incredibly kind, smart, and friendly and I'm happy that I connected with some people.

One final tip: before you go to NeurIPS (or another big conference) take some Emergen-C or Airborne or something. I got sick towards the end of the conference and I probably could have avoided it by taking some vitamin C in advance. :wink:

Thank you for reading this post! I'm happy to answer any questions in the comments and if you have your own NeurIPS experience to share, it would be great to hear about it!

:tada: **Happy New Year!** :tada: