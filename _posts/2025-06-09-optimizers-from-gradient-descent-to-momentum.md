---
title: "Optimizers Part 1: From Gradient Descent to Momentum"
layout: single
author_profile: true
math: true
toc: true
toc_label: "On this page"
toc_sticky: true
description: "Part 1 of common optimizers used in Machine Learning"
categories: [blog]
tags: ["Machine Learning", "Deep Learning", "Optimizers", "Mathematics"]
---
> **Disclaimer:** *None of the content in this post was written or edited by LLMs — everything here was written by me.*

## Introduction

Every machine learning model, whether it's simple linear regression or a sophisticated neural network, shares one common goal: minimizing some kind of error or loss function. But how does a model actually figure out what changes to make to improve its predictions?

That's where **optimizers** come in. An optimizer is an algorithm that adjusts the parameters (weights and biases) of a machine learning model during training to minimize the loss function as efficiently as possible. Concretely, if the learning problem is parameterized by the parameter vector \\(w \in \R^n\\), we want to efficiently find

$$
\min_w \mathcal{L}(w)
$$

where \\( \mathcal{L} : \R^n \rightarrow \R \\) is the loss function. Let's denote the optimal value of \\(w\\) as \\( w^\*\\). The optimizers we will cover fall into the category of *search direction methods*, a term taken from the field of numerical optimization, where the core idea is that given a starting point \\(w_0\\), we construct a sequence of iterates \\(w_1, w_2, \dots\\) with the goal that, under certain conditions, $w_k \rightarrow w^\*$ as $k \rightarrow \infty$. In a search direction method, we think of constructing $w_{k + 1}$ from $w_k$  by writing $w_{k + 1} \leftarrow w_k + s$ for some "step" $s$. All of the optimizers we discuss will have the following generic format:

$$
\begin{array}{l}
\textbf{Algorithm: Search Direction Method} \\
\textbf{Input: } w_0 \text{ (initial weights)} \\
k \leftarrow 0 \\
\textbf{While} \text{ not converged:} \\
1. \quad \text{Pick a step } s \\
2. \quad w_{k+1} \leftarrow w_k + s \text{ and } k \leftarrow k + 1 \\
3. \quad \text{If converged, set } \hat{w} = w_k \text{ and break} \\
\textbf{Return } \hat{w}
\end{array}
$$


The optimizer is responsible for choosing the step $s$ at every iteration of the above algorithm.

In this post, we will cover a range of optimization algorithms, starting from basic Gradient Descent, moving through Stochastic Gradient Descent (SGD), and then covering SGD with Momentum and its variant, Nesterov Accelerated Gradient. Many of these ideas will lay the foundation for Part 2, where we’ll do a deep dive into more advanced optimizers like Adam and AdamW.

## Gradient Descent

The core idea of gradient descent can be summed up as: given that we are currently at $w_k$, determine the direction in which the function decreases most rapidly and take a step in that direction. Using the linear approximation to $\mathcal{L}$ at $w_k$ given by the Taylor Series, we find that this direction is given by the *negative gradient* of the loss at $w_k$, i.e $s \propto -\nabla \mathcal{L}(w_k)$. However, note that the linear approximation is only good locally around $w_k$, so the step needs to be multiplied by some small constant to keep our steps small. Concretely, $s = -\alpha \nabla \mathcal{L}(w_k)$, where $\alpha > 0$ is known as the **learning rate**. 

In practice, if the loss function is computed over a dataset of $N$ examples, say $\mathcal{L}(w_k) =  \frac{1}{N} \sum_{i=1}^N \ell_i(w_k)$, then gradient descent computes the gradient of the total loss by calculating the individual gradients on each data point and averaging them:

\\[
\begin{equation}
\nabla \mathcal{L}(w_k) = \frac{1}{N} \sum_{i=1}^N \nabla \ell_i(w_k)
\end{equation}
\\]

To determine whether the algorithm has converged, we check if the norm of the gradient is below some small threshold $\varepsilon > 0$, i.e

$$
\| \nabla \mathcal{L}(w_k) \| < \varepsilon
$$

This is because, for a differentiable loss function, any local minimum should have gradient 0. Therefore, if the gradient is very small in norm, it suggests that we're very close to a stationary point -- ideally a local minimum. Also, if the gradient is small, we have reached a point where the function isn't changing much in any direction, so further updates will have negligible effect.

Putting this together, the gradient descent algorithm is:

$$
\begin{array}{l}
\textbf{Algorithm: Gradient Descent} \\
\textbf{Input: } w_0, \alpha, \varepsilon \\
k \leftarrow 0 \\
\textbf{While } \text{True:} \\
1. \quad \text{Set } g_k = \frac{1}{N} \sum_{i=1}^N \nabla \ell_i(w_k) \\
2. \quad \text{If } \|g_k\| < \varepsilon \text{, set } \hat{w} = w_k \text{ and break} \\
3. \quad w_{k+1} \leftarrow w_k - \alpha \cdot g_k \text{ and } k \leftarrow k + 1 \\
\textbf{Return } \hat{w}
\end{array}
$$

Despite its appealing simplicity, gradient descent does have a few issues that make it a bad choice for modern machine learning methods:
1. Gradient descent is only guaranteed to converge to the optimal $w^\*$, i.e the global minimum, if the loss function is convex. However, most machine learning problems are nonconvex, and have very complicated loss surfaces, with many local minima. It's highly likely that if gradient descent finds a minimum, it is merely a local one, and not the global minimum.
2. Computing the gradient on *every single point* in the dataset at every iteration is very computationally expensive, given that today's machine learning problems use datasets that number in the billions or even trillions of datapoints.
3. **Saddle points** are critical points where the function is neither at a global maximum or a global minimum. The function may be curving upwards in one direction (like a hill) and downwards in another (like a valley). Training can slow down, or even stop, near a saddle point. What's worse, gradient descent has no way to know that it's at a saddle point.

To address issue 1, we turn to **Stochastic Gradient Descent** (SGD), which introduces randomness in exchange for greater computational efficiency.

## Stochastic Gradient Descent

As we mentioned above, computing the average gradient for every point in the dataset at every iteration is extremely expensive. If the $N$ points in our dataset are numnbered $1, \dots, N$, what if, instead, we sample a random point $i$ from the distribution, calculate the gradient of the loss function $\nabla \ell_i(w_k)$ for that particular point, treat that as an unbiased estimate of the actual gradient at $w_k$, and use that as the direction of the step at a given iteration?

Well, that's precisely what Stochastic Gradient Descent does. In practice, we generate a random permutation $\sigma \in S_N$, where $S_N$ is the set of all permutations of the set \\( \\{1, \dots, N\\} \\), instead of sampling a new random point at every iteration. Here's the algorithm:

$$
\begin{array}{l}
\textbf{Algorithm: Stochastic Gradient Descent (One Epoch)} \\
\textbf{Input: } w \text{ (current parameter vector)}, \alpha, \sigma \\
\textbf{For } j = 1 \text{ to } N: \\
\quad \text{Let } i \leftarrow \sigma(j) \\
\quad \text{Compute } g = \nabla \ell_i(w) \\
\quad w \leftarrow w - \alpha \cdot g \\
\textbf{Return } w
\end{array}
$$

This process defines one pass over the dataset, which is commonly referred to as an **epoch**. At the beginning of every epoch, we generally sample a fresh permutation $\sigma$

### Convergence in SGD

One epoch is not good enough to converge to a local minimum, and therefore, we typically multiple epochs until we've converged. But, what's our convergence condition in this case? For vanilla gradient descent, we checked if the total gradient's norm was below some stopping point. SGD does not calculate the full gradient, and so we cannot use the same convergence condition. Instead, we typically use a proxy, like
1. Running the algorithm for some fixed number of epochs.
2. Tracking the changes in validation loss across epochs, and stopping when the improvement plateaus.
3. Monitoring the magnitude of gradient updates (using an exponentiallly moving average, for example) stopping when they become sufficiently small.

### Pros and Cons

SGD is a popular optimizer, and has its own strengths and tradeoffs:

#### Pros

1. **Computationally efficient:**  Only one data point is used per step, saving time and memory.
2. **Fast initial progress:** Frequent updates help reach a good region of the solution space quickly.
3. **Works in online settings:** Naturally suited for streaming data or online learning.

#### Cons

1. **Noisy:** The variance inherent in choosing one random point at a time makes convergence less stable.
2. **Slower convergence near minima:** The algorithm can oscillate around a minimum instead of settling.
3. **Learning rate sensitivity:** The two points above mean that SGD is highly sensitive to the choice of learning rate. It also needs a learning rate scheduler that decays the rate over time (e.g. $\alpha_t \propto \frac{1}{t}$) to have theoretical convergence guarantees.
4. **Susceptible to getting stuck in local minima:** Even though the noisiness of using one point at a time gives some implicit regularization and can help improve generalization, SGD -- just like vanilla GD -- is still prone to converging to local minima and not the global minimum.
5. **Saddle Points:** Despite its noise, SGD can still get trapped near saddle points

### Mini-batch Gradient Descent

In practice, we interpolate between the extreme cases of vanilla Gradient Descent (where every datapoint is used in every iteration) and SGD (where only one datapoint is used in each iteration) by using "mini-batches". At every iteration, the gradient terms are computed for a small number (like 64, or 128) of randomly chosen points, with the average of those gradients used to compute the step. Mini-batches are typically chosen to be powers of 2 (e.g., 32, 64, 128) because memory access patterns and buffer allocations on GPUs are optimized for powers-of-2 sizes, enabling more efficient use of hardware resources. These sizes are also large enough to leverage data parallelism on modern training hardware. When properly implemented, this decreases the variance in each iteration, improving convergence stability and the speed of training, at the cost of almost zero computational overhead.

## SGD with Momentum

Although SGD offers significant efficiency gains, it has a major shortcoming: it tends to oscillate heavily, especially in regions where the loss surface is steep in one direction and flat in another ( a so-called *ravine* in ML literature). This behavior slows convergence and can prevent the optimizer from escaping narrow valleys or saddle points efficiently.

The idea behind momentum is simple yet powerful: instead of using just the current gradient to update our parameters, why not keep a running average of past gradients? This allows the optimizer to "remember" the direction it's been heading and maintain speed in those directions — much like how a ball rolling downhill gains speed and keeps going, even across small bumps.

<figure style="display: flex; flex-direction: row; justify-content: center; gap: 20px;">
  <img src="/assets/images/optim-1/valley1.gif" alt="Without momentum" style="width:45%;">
  <img src="/assets/images/optim-1/valley2.gif" alt="With momentum" style="width:45%;">
  <figcaption style="text-align: center; width: 100%; margin-top: 10px;">
    <em>Two optimization trajectories: left shows standard SGD, right shows SGD with Momentum. Notice how momentum helps smooth out oscillations and accelerates in the consistent direction. Source: <a href="https://people.willamette.edu/~gorr/classes/cs449/momrate.html" target="_blank" rel="noopener noreferrer">Genevieve Orr</a> </em>
  </figcaption>
</figure>

Concretely, this is implemented using an exponential moving average (EMA) of past gradients. By smoothing the gradients across time, momentum helps dampen oscillations and accelerates convergence, especially in the directions of consistent descent.

We introduce a velocity term $v_k \in \R^n$, initialized as zero, that acts as the EMA with gradient . At each step, we update this velocity using the current gradient and a momentum parameter $\beta \in [0,1]$. The parameter determines how much of the previous velocity to retain:

\\[
\begin{align} 
v_{k+1} &= \beta v_k - \alpha \nabla \ell_i(w_k) \\newline
w_{k+1} &= w_k + v_{k+1} 
\end{align} 
\\]

Do note that this presentation is only one way to write momentum updates. PyTorch's `torch.optim.sgd` function, for example, uses a slightly different equation.

The momentum parameter $\beta$ generally takes values like 0.9 or 0.99. Momentum-based methods are especially helpful in deep learning scenarios where loss surfaces are high-dimensional and full of ravines and saddle points. By accumulating gradient information, momentum helps escape shallow local minima and converges faster to useful solutions.

### Nesterov Accelerated Gradient

A variation on the Momentum Update that has recently gained a lot more popularity over vanilla Momentum is the Nesterov Accelerated Gradient (NAG). The core idea behind NAG is that when the current parameter vector is at the point $w_k$, then looking at the momentum update above, we know that the momentum term alone will nudge the parameter vector by $\beta v_k$. So, to compute the gradient, we can use the "lookahead point" $w_k + \beta v_k$, instead of the "stale" point $w_k$. The update equation now becomes

\\[
\begin{align} 
v_{k+1} &= \beta v_k - \alpha \nabla \ell_i(w_k + \beta v_k) \\newline
w_{k+1} &= w_k + v_{k+1} 
\end{align}
\\]

NAG has been observed in various deep learning applications to converge faster and with fewer oscillations compared to regular momentum. This is especially true in ravines, which are common in the loss surfaces seen in Deep Learning. NAG is backed by stronger convergence guarantees in convex settings (Nesterov's original analysis showed an optimal convergence rate for smooth convex functions). NAG only requires evaluating the gradient at a different point (the lookahead), which is a small extra cost in practice. However, this small cost yields disproportionate benefits.

> **Note:** *This entire blog exists as a way for me to create an organised set of notes of what I've learned, and is not meant to be as complete as a textbook. Many topics are only mentioned briefly, since I'm already fairly comfortable with them.*
