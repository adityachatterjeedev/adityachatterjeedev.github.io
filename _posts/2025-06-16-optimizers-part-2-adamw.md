---
title: "Optimizers Part 2: From SGD with Weight Decay to AdamW"
layout: single
author_profile: true
math: true
toc: true
toc_label: "On this page"
toc_sticky: true
description: "Part 2 of common optimizers used in Machine Learning"
categories: [blog]
tags: ["Machine Learning", "Deep Learning", "Optimizers", "Mathematics"]
---

## Introduction

In [Part 1](../optimizers-part-1-from-gradient-descent-to-momentum){:target="_blank"}, we explored foundational optimization methods used in training machine learning models, from vanilla **Gradient Descent** to **Stochastic Gradient Descent (SGD)**, and extensions like **Momentum** and **Nesterov Accelerated Gradient**. These algorithms form the backbone of modern machine learning, offering efficient methods for navigating complex loss surfaces and updating model parameters.

While these methods focus primarily on the mechanics of optimization — how to move through the parameter space efficiently — they don’t directly address another critical concern in machine learning: **generalization**. (Author’s note: I like em-dashes. No AI here!)

A model that minimizes training loss isn’t necessarily a good model. Without proper safeguards, it can overfit the training data, capturing noise rather than signal. This is where **regularization** enters the picture, helping to steer optimization toward simpler solutions that generalize better to unseen data at the cost of some training loss.

In this second part, we begin with a look at **SGD with Weight Decay**, often motivated through **L2 regularization**, and explore how it modifies the update rule to penalize large weights and encourage smoother models.

This leads naturally into a deep dive on **Adam**, one of the most widely used adaptive optimizers in deep learning. Adam combines ideas from Momentum and RMSProp to maintain separate learning rates for each parameter, adapting them based on historical gradient information. However, despite its popularity, Adam has a subtle but important flaw that can hurt generalization, specifically when used with naïvely implemented weight decay. That brings us to **AdamW**, a seemingly small but profoundly important fix to Adam's update rule. We'll unpack the math behind AdamW, show how it improves generalization in practice, and explain why this subtle shift leads to better-behaved training dynamics and more robust models.

## SGD and Weight Decay

In many models, we don't just want to minimize the loss on the training data, we also want to *discourage large weights* to promote model generalization and prevent overfitting. This leads to a regularized objective (aka loss) function:

\\[
\mathcal{L}_{reg}(w) = \mathcal{L}(w) + \frac{\lambda}{2} \\| w \\|^2
\\]
where
- $\mathcal{L}(w)$ is our original loss function (eg. MSE or cross-entropy)
- $\\| w \\|^2$ is the squared L2 norm of the weights (we don't usually regularize biases)
- $\lambda \in [0,1]$ is the regularization strength, a tunable hyperparameter

The gradient of this regularized loss function is

\\[
\nabla \mathcal{L}_{reg}(w) = \nabla \mathcal{L}(w) + \lambda w
\\]

The SGD update rule then becomes

\\[
\begin{align}
w_{t + 1} &= w_t - \alpha \left( \nabla \mathcal{L}(w_t) + \lambda w_t \right) \\newline
 &= (1 - \alpha \lambda) w_t - \alpha \nabla \mathcal{L}(w_t)
\end{align}
\\]

So, each weight gets shrunk slightly at each update step, before taking the gradient step, and this is where the idea of weight decay comes from.

Notice how $(1 - \alpha \lambda)w_t$ applies shrinkage **independent of the loss gradient**, acting as a constant force gently pulling weights towards zero. Importantly, we've just shown that L2 regularization and weight decay regularization are mathematically equivalent for vanilla SGD. However, as we shall see soon, this does not hold in the case of adaptive optimizers like Adam.

## From Adam to AdamW

AdamW is arguably the most popular optimizer in modern deep learning applications. Its predecessor, Adam, was introduced by Kingma and Ba in their now-famous [2014 paper](https://arxiv.org/abs/1412.6980), and quickly gained traction in the deep learning community due to its adaptive learning rate mechanism and ease of use. However, in its early years, Adam didn’t always outperform classical momentum-based methods like Nesterov Accelerated Gradient, which often showed better convergence properties in practice.

This changed in 2017 when [Loshchilov and Hutter](https://arxiv.org/abs/1711.05101) proposed a simple yet critical modification to Adam, resulting in AdamW. Their paper showed that the commonly used implementation of L2 regularization in Adam was *not equivalent* to traditional weight decay, and that the two should be treated separately. By decoupling weight decay from the gradient-based parameter updates, they introduced an optimizer that not only preserved Adam’s adaptive behavior but also improved generalization and training stability across a wide range of tasks.

To understand why this is the case, let's do a step-by-step breakdown of Adam, to see how AdamW improves on its predecessor. At a high level, Adam blends Momentum and RMSProp to maintain separate learning rates for each parameter, adapting them based on historical gradient information. Let's look at what happens to a single parameter $w$:

1. **Get the gradient:** Compute the gradient of the loss function w.r.t $w$ at timestep $t$: 
    \\[
    g_t = \frac{\partial \mathcal{L}}{\partial w_t}
    \\]

2. **Find the first moment (aka momentum):** 
    \\[
    m_t = \beta_1 \cdot m_{t - 1} + (1 - \beta_1) \cdot g_t    
    \\]

    where $\beta_1 \in [0,1]$ is a tunable hyperparameter. This is an exponentially moving average of the gradients, similar to what we've seen with Momentum in part 1. 

3. **Find the second moment (aka variance):**
    \\[
    v_t = \beta_2 \cdot v_{t - 1} + (1 - \beta_2) g_t^2
    \\]

    where $\beta_2 \in [0,1]$ is a tunable hyperparameter. This is an EMA of the squared gradients, which acts as a running estimate of how large or volatile the gradient is for each parameter over time. This will help with the adaptive rescaling in the update step, as we'll see shortly. $\beta_2$ is generally close to 1 (eg 0.999)

4. **Bias Correction:** Since $m_t$ and $v_t$ start at 0, they are biased during the early steps. To fix that, we divide them by a scaling factor:

    \\[
        \hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
    \\]

    Without correction, both $m_t$ and $v_t$ would systematically underestimate the true first and second moments in the early steps, especially when $t$ is small. By dividing by $1 - \beta_1^t$ and $1 - \beta_2^t$ respectively, we "de-bias" the estimates and ensure they reflect the true moving averages rather than being skewed toward zero due to their initial values. Note that as $t$ grows larger, the denominators tend to 1.

5. **Apply the update:**
    \\[
        w_{t + 1} = w_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
    \\]

    where $\varepsilon$ is a small smoothing constant to prevent division by zero. Here is where we start to see the importance of the variance. When $v_t$ is large, it means that the gradients for the parameter may be unstable, or noisy, or even changing directions rapidly. In such cases, a large and unstable gradient could *overshoot the minimum*, leading to oscillations instead of convergence, or even cause exploding updates. To prevent this, Adam/AdamW scales the learning rate down for the parameter.

    On the other hand, if $v_t$ is small, it indicates that gradient updates are small and stable, in which case Adam is certain about the direction in which the parameter should be moved. In this case, the update step is scaled up for the parameter.

This is the full update loop for one update step in Adam. This update is applied to every single parameter in the model.

### Weight decay, The Crucial Difference

In the original Adam algorithm, if you wanted to apply L2 regularization, you would modify the loss function:

\\[
\mathcal{L}_{reg}(w) = \mathcal{L}(w) + \frac{\lambda}{2} \\| w \\|^2
\\]
and then compute gradients as usual, resulting in 

\\[
    g_t = \frac{\partial \mathcal{L}}{\partial w_t} + \lambda w_t
\\]

This approach *adds the regularization term directly into the gradient*, meaning it's passed through all of Adam’s internal machinery (momentum averaging, variance scaling, bias correction) before being applied to the weights. But here's the problem: **this is _not_ true weight decay**.

Loschilov and Hutter showed that this breaks the intended behavior of regularization, namely that the weight decay should be independent of the gradient-based update, as we saw above in equation (2). The penalty on large weights becomes entangled with the gradient’s magnitude and direction, leading to unintuitive — and sometimes harmful — update dynamics.

AdamW's fix was simple yet quite brilliant. Instead of modifying the original loss function, AdamW simply shrinks the weights directly at each step:

\\[
w_{t + 1} = w_t - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda w_t \right)
\\]
which can be rewritten, just like we saw earlier in the SGD with weight decay equations, to give us the **AdamW update step**:

\\[
w_{t + 1} = \underbrace{(1 - \alpha \lambda) \cdot w_t}_{\text{Weight decay}} - \underbrace{\alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}} _{\text{Adam update step}}
\\]

This decoupling, while subtle, is powerful. It restores weight decay to its original interpretation: a constant force pulling the weights toward zero, regardless of the gradient. This leads to more predictable and often better generalization performance, especially in large-scale models.