---
layout: post
author: Jacob Pettit
comments: true
tags: reinforcement-learning
title: Why I think RL Tooling Matters
featured-image: /assets/imgs/default-banner.png
---

**The tools we use influence the research we do, and while there are many good RL tools out there, there are still areas where tools need to be built.**

{: class="table-of-content"}
* TOC
{:toc}

# The RL Tools Everyone Has 

Maybe I'm wrong, but I think every RL researcher has some tools they've built and that they use across their projects. Since they've built them, these tools are the perfect fit for them. But their tools might also be useful for someone else. However, we rarely see code for RL tools get packaged up and open-sourced. I'm guilty of it too. I've got the same NN code that I copy across projects, and my method of reuse of some core utility functions is criminal. Suffice to say, any software engineer would cringe at my process. But it works for me. I run experiments, try new things, and get results.  

The thing that sucks, though, is when I want to let someone else run my code. Suddenly, they're looking at this mess that I've been hacking at for months (and it's really a mess). I doubt this is a problem that only I have. So lately, I've been trying to be better about writing clear, maintainable code. I've made an effort to make my few NNs and utilities easier to reuse across projects (as in, I've stopped just cp-ing files into whatever project directory I'm currently in). I think this has made a big difference in my work. Coworkers say my code is "pretty" and "readable". Imagine that! I'm basking in my own glow a bit over here, but my point is that a little work up front goes a long way towards enabling future you (and your coworkers) to understand and reuse your code.

I'm not talking about building some massive, engineered, impressive library just for you and whatever coworkers happen to find it useful (I mean, if you open-source it and get lucky, maybe *everyone* will love it) but rather I'm talking about taking a bit of extra time at the start of a new project to properly set it up as a repository and as an [editable Python package](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) to make it easier for future you to reuse your code. If you like Jupyter Notebooks, I think [nbdev](https://nbdev.fast.ai) is an awesome resource for this kind of stuff. 

# The RL Tools We Need

Again, I'm going out on a limb here and extrapolating from a few experiences (mine and coworkers), but it seems like there are some RL tools that don't really exist yet and might be generally useful. 

I think a widely used and trusted collection of environment wrappers would be excellent. Things like frame stacking, state and reward normalization, and meta-env wrappers would be extremely helpful. Some libraries include some of these wrappers, but I've yet to see anyone put it front-and-center in their package. I think that there is too much focus on implementing algorithms for people to run, and not enough focus on providing distinct tools that researchers can use when building new ideas. Like there really just needs to be a few high quality RL algorithm implementations (hopefully in general a set of scalable implementations for each DL framework) for benchmarking purposes and production work (if you can productionize RL, serious props) and such. Besides that, I suspect it's more useful to provide a framework that does something enabling researchers to only implement the novel parts of their work, and use pre-built components where they can.  

This has been a bit of a rant, but this problem has been bugging me a bunch lately. Unfortunately, it seems like the modular RL frameworks that I think look easy and pleasant to use are all made by DeepMind, which means that they work with JAX and TensorFlow, but not PyTorch. As a PyTorch user and lover, this makes me sad. I mean, JAX seems awesome too but on top of working full time, it's just hard to set aside the time to learn a new framework. 

Recently, [PFRL](https://pfrl.readthedocs.io/en/latest/) came out, and I think it helps with this problem some, but of course there is still progress to be made. And I still haven't found a package offering a set of high-quality, modular environment wrappers! 

If someone wrote a set of wrappers that worked with NumPy arrays (because that's what Gym takes as arguments and what it returns) then it is easy for people using frameworks to convert those NumPy arrays to the array format of their chosen framework. Maybe I'll work on this myself, we'll see. 

# The Tools I Love 

If I were to talk only about the RL tools I truly love, this would be a short list. We'd talk about OpenAI's Gym and about the PyBullet environments and that would be it. So instead I'll also talk about a couple of ML tools that I use a ton in RL that I love.

## [PyTorch](https://pytorch.org)

I use PyTorch for all of the general NN and math stuff it's built for.

## [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)

[PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) is awesome. It can feel kind of weird and hacky to use it for RL, but it imposes a common structure across code that is just fantastic. Plus, it comes with a ton of free stuff, like GPU/multi-node training, automatic experiment logging, and so on. I won't do a full sales pitch for PyTorch-Lightning, but check out their documentation if you're a PyTorch user.

## [OpenAI's Gym](https://gym.openai.com)

The Gym package provides a standard set of RL environmnents, and gives an API for implementing new environments. It's been flexible and helpful in RL research for a while. 

## [PyBullet](https://pybullet.org/wordpress/)

I have yet to build a custom environment with PyBullet, but I love that it's open-source and that they provide alternatives to the MuJoCo environments. It works well, is easy to install, and runs quickly on my laptop (a MacBook Pro). Love it.

## [Weights and Biases](https://www.wandb.com)

This is the best experiment logger I've used. If you're doing RL, and you wrap your environment in a Monitor, then Weights and Biases can automatically log videos of your agent in the environment to their dashboard.

## [nbdev](https://nbdev.fast.ai)

If you like Jupyter Notebooks then I think nbdev is a great thing to look at for building your code up as a package. All the stuff that you don't export into your package becomes tests for your code, so the tests are pretty much built-in from your notebook! However, I also like to experiment with Python files a bunch, so I've also found it useful in the case where you're cleaning up and packaging some code you've pre-written.

# Wrapping it up 

Sorry the structure of this post doesn't really make much sense. It's basically been a big rant about how I think RL tools aren't good enough yet. I'm experimenting with writing less-polished, more-frequent posts instead of agonizing over trying to make a post as perfect as possible. So this is much more stream-of-consciousness than earlier posts.  

But otherwise, yeah, this is a pretty big problem. A library or set of libraries that provide common utilities to RL folks will probably help a ton with the reproducibility crisis in deep RL right now. If we could design algorithms using trusted component implementations and then only custom-write what's new to our algorithm, that would likely help eliminate bugs in research code. It would even help people re-implementing your algorithm later on. Plus, I (and I bet other researchers) would be able to iterate a lot quicker if I could use and swap out or custom-write whatever algorithm components I need during the experimentation process. 

I suppose this is all just food for thought.