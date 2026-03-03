---
layout: post
title: "So why do we scale dot product attention?"
date: 2026-02-23
tags: [deep-learning, transformers, attention]
---

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

When I first learned about attention it was the kind used in transformers - the equation above. I didn't really understand the equation beyond the high level "multiply this and that and then you get some vectors in the output" and I didn't think much about it. 

A few years ago, when I was working with LLMs and implementing some components for custom models I started diving deeper into the math and code implementations behind different pieces of the transformer architecture. It was then that I came back to the equation you see above. 

<!--more-->

What kept nagging at me was why divide by sqrt(dim)? I had a hunch for the answer since it had the tell-tale signs of scaling variance, but I never bothered to prove it to myself and see it with my own eyes. That's when I decided to run a little experiment.

## What I Did
The experiment I did was pretty simple.

First of all I implemented scaled dot product attention (SDPA) in JAX because I wanted to have access to all the intermediate arrays.

I made two matrices - Q and K - filled with random values drawn from $\mathcal{N}(0, 1)$. Mean of zero, standard deviation of 1. Then I multiplied them, like attention does.

```python
from jax import numpy as jnp

q = random.normal(key1, (1, 100, 1000))  # std ~1
k = random.normal(key2, (1, 100, 1000))  # std ~1

logits = jnp.matmul(q, k.transpose((0, 2, 1)))
```

The standard deviation went from ~1 to ~31. The values exploded. This is the part I had a hunch about. But why did this matter? I kept exploring and examining outputs at each stage of the SDPA.

The next step was to pass these through softmax. This is where it got interesting, because every row collapsed — one entry near 1, everything else near 0. The softmax was fully saturated. Weird, and my intuition told me this was a problem. I had an inkling that this would somehow bode poorly for the gradients. I started looking at other resources online to try and figure this out. 

Turned out that, yes, saturation in the softmax meant that the softmax derivative ($S_i(1 - S_j)$) becomes tiny and gradients vanish during training. Which can be problematic.

The really cool thing is that I could see it — actually see it! The saturation of the softmax that the online resources talked about. It felt so exciting, re-discovering it on my own.

## What I Figured Out

The natural next step was to go back to scaling. All resources pointed to that little sqrt(dim) scaling as the way to combat this issue. But why? What caused the need to scale and why scale by sqrt(dim)?

My intuition told me it had something to do with variances (the sqrt was what was tripping that hunch). So I went back to definitions and pen and paper.

Each entry in $QK^T$ is a dot product — you're summing $d_k$ products of random values. If each product has variance 1, then summing $d_k$ of them gives you variance $d_k$:

$$\text{Var}\left[\sum_{d=1}^{d_k} X_d \cdot Y_d\right] = d_k$$

I checked: $31.28^2 \approx 961$. My $d_k$ was 1000. It all lined up.

So the variance grows with the dimension. The bigger your model, the worse this gets. And now the fix was starting to reveal itself naturally — if you divide by $\sqrt{d_k}$, you undo exactly that scaling:

$$\text{Var}\left[\frac{QK^T}{\sqrt{d_k}}\right] = \frac{1}{d_k} \cdot d_k = 1$$

```python
scaled = logits / jnp.sqrt(1000)
# std: 0.989
```

Back to ~1. Softmax behaves and gradients can flow better. Such a small thing but it was so cool to walk through it, see the reason behind the scaling emerge naturally to the point where I was sort of re-discovering this for myself!

## A bit of math
To wrap up the exercise I wanted to work out how this all made sense mathematically. This meant proving, at least roughly, to myself, with equations, that all of the above was true. Let me re-create my paper scribbles here in a more legible manner.

Let $X \sim \mathcal{N}(0, 1)$ and $Y \sim \mathcal{N}(0, 1)$ be two independent random variables sampled from the standard normal distribution.

Then the variance of the product of those two variables can be written as
$$\text{Var}(X \cdot Y)$$

Since $X$ and $Y$ are independent, we can expand this using the definition of variance:

$$\text{Var}(X \cdot Y) = E[X^2 \cdot Y^2] - E[X \cdot Y]^2 = E[X^2] \cdot E[Y^2] - E[X]^2 \cdot E[Y]^2$$

Now, we recall that $X$ and $Y$ have zero mean and variance 1 in our case. Since $E[X] = E[Y] = 0$, the second term vanishes:

$$= E[X^2] \cdot E[Y^2] - 0$$

And since $E[X^2] = \text{Var}(X) + E[X]^2 = 1 + 0 = 1$ (and likewise for $Y$):

$$= 1 \cdot 1 = 1$$

So each pairwise product $X_d \cdot Y_d$ has variance 1. The dot product in attention is a sum of $d_k$ such independent products, so:

$$\text{Var}\left[\sum_{d=1}^{d_k} X_d \cdot Y_d\right] = \sum_{d=1}^{d_k} \text{Var}(X_d \cdot Y_d) = d_k \cdot 1 = d_k$$

Which is exactly what we observed — and the case for our transformer attention setup!


## Conclusion
This little exercise was really exciting for me and getting to that "aha" moment felt so rewarding. I walked away feeling like I actually understood what was going on and I wanted to share this with others, perhaps if you are like me a few years back and curious about some details, this post will help make it click for you too!
