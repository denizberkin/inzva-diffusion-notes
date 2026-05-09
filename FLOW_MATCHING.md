# **Flow Matching Notes**

---

# **Table of Contents**

- [0. Idea](#0-idea)
- [1. Main Objects](#1-main-objects)
  - [1.1 Trajectory](#11-trajectory)
  - [1.2 Flow](#12-flow)
  - [1.3 Probability Path](#13-probability-path)
  - [1.4 Vector Field / Velocity](#14-vector-field--velocity)
- [2. From Diffusion to Flow Matching](#2-from-diffusion-to-flow-matching)
- [3. Conditional Probability Path](#3-conditional-probability-path)
- [4. Conditional Flow Matching Objective](#4-conditional-flow-matching-objective)
- [5. Training](#5-training)
- [6. Inference](#6-inference)
- [7. Rectified Flow](#7-rectified-flow)
- [8. Summary](#8-summary)
- [References](#references)

---

# **0. Idea**

In generative modeling, we want to sample from the data distribution:

$$
p_{\text{data}}(x)
$$

Flow matching does this by learning a **velocity field** that transports samples from a simple initial distribution to the data distribution.

Usually:

$$
p_0(x) = \mathcal{N}(0, I)
$$

$$
p_1(x) = p_{\text{data}}(x)
$$

So the generation problem becomes:

> Start from Gaussian noise and learn how to move it continuously toward data.

Unlike DDPM-style diffusion, where the common convention is:

$$
\text{data} \rightarrow \text{noise}
$$

flow matching often uses:

$$
\text{noise at } t=0 \rightarrow \text{data at } t=1
$$

---

# **1. Main Objects**

## **1.1 Trajectory**

A **trajectory** is the path followed by a single sample over time:

$$
x_t, \quad t \in [0, 1]
$$

It tells us where one point is located at each time step.

---

## **1.2 Flow**

A **flow** is the collection of trajectories for many starting points.

If different initial samples start from $p_0$, the flow moves all of them toward the target distribution $p_1$.

---

## **1.3 Probability Path**

The **probability path** describes the distribution of samples at time $t$:

$$
p_t(x)
$$

At the endpoints:

$$
p_0(x) = \mathcal{N}(0, I)
$$

$$
p_1(x) = p_{\text{data}}(x)
$$

So $p_t(x)$ is the intermediate distribution between noise and data.

---

## **1.4 Vector Field / Velocity**

The **vector field** tells each sample where to move:

$$
u_t(x)
$$

It depends on:

- current time $t$
- current location $x$

The sample evolves according to the ODE:

$$
\frac{dx_t}{dt} = u_t(x_t)
$$

This is the core object learned in flow matching.

---

# **2. From Diffusion to Flow Matching**

Diffusion models and score-based models learn how to reverse a noising process.

Flow matching instead directly learns a velocity field.

| Method | Learns | Generation |
|---|---|---|
| DDPM | noise $\epsilon_\theta(x_t, t)$ | reverse denoising chain |
| Score-based model | score $s_\theta(x_t, t) \approx \nabla_x \log p_t(x)$ | reverse SDE / probability-flow ODE |
| Flow matching | velocity $u^\theta_t(x)$ | deterministic ODE |

The key distinction:

- **Score** tells the direction of increasing density.
- **Velocity** tells how the sample should move through time.

So, score matching is closer to learning a **compass**, while flow matching learns the **motion itself**.

<div style="text-align: center;">
  <img src="assets/fm.gif" alt="fm" />
</div>


---

# **3. Conditional Probability Path**

To make the objective tractable, flow matching uses a simple conditional path between:

- noise sample $x_0 \sim \mathcal{N}(0, I)$
- clean data sample $x_1 \sim p_{\text{data}}$

A common interpolation is:

$$
x_t = (1-t)x_0 + tx_1
$$

This gives:

$$
x_{t=0} = x_0
$$

$$
x_{t=1} = x_1
$$

For a fixed target data point $x_1$, the conditional probability path can be written as:

$$
p_t(x \mid x_1) = \mathcal{N}(tx_1, (1-t)^2I)
$$

The corresponding conditional vector field is:

$$
u_t(x \mid x_1) = \frac{x_1 - x}{1-t}
$$

For the sampled interpolation $x_t = (1-t)x_0 + tx_1$, this is equivalent to:

$$
u_t(x_t \mid x_1) = x_1 - x_0
$$

So the target velocity is simple:

$$
\text{target velocity} = \text{data} - \text{noise}
$$

---

# **4. Conditional Flow Matching Objective**

The ideal flow matching objective would compare the model velocity to the true marginal vector field:

$$
\mathcal{L}_{\text{FM}}=
\mathbb{E}_{t, x}
\left\|
u^\theta_t(x) - u_t(x)
\right\|^2
$$

But $u_t(x)$ is generally not directly available.

Conditional flow matching uses the tractable conditional target instead:

$$
\boxed{
\mathcal{L}_{\text{CFM}}=
\mathbb{E}_{t, x_0, x_1}
\left\|
u^\theta_t(x_t) - (x_1 - x_0)
\right\|^2
}
$$

where:

$$
x_0 \sim \mathcal{N}(0, I)
$$

$$
x_1 \sim p_{\text{data}}
$$

$$
t \sim \mathcal{U}(0, 1)
$$

$$
x_t = (1-t)x_0 + tx_1
$$

This is one of the main advantages of flow matching: the final training loss is very simple.

---

# **5. Training**

Training procedure:

1. Sample noise:

$$
x_0 \sim \mathcal{N}(0, I)
$$

2. Sample clean data:

$$
x_1 \sim p_{\text{data}}
$$

3. Sample time:

$$
t \sim \mathcal{U}(0, 1)
$$

4. Interpolate:

$$
x_t = (1-t)x_0 + tx_1
$$

5. Predict velocity:

$$
u^\theta_t(x_t)
$$

6. Compare with the target velocity:

$$
\text{target} = x_1 - x_0
$$

Final loss:

$$
\boxed{
\mathcal{L} =
\left\|
u^\theta_t(x_t) - (x_1 - x_0)
\right\|^2
}
$$

**Intuition**:

The model sees a partially transformed sample $x_t$ and learns which direction and speed would move it toward the clean data sample.

---

# **6. Inference**

During inference, we no longer have paired clean data.

We only sample noise:

$$
x_0 \sim \mathcal{N}(0, I)
$$

Then solve the ODE:

$$
\frac{dx_t}{dt} = u^\theta_t(x_t)
$$

from $t=0$ to $t=1$.

Using Euler integration:

$$
x_{t+\Delta t} = x_t + \Delta t \cdot u^\theta_t(x_t)
$$

After several steps, the final output is:

$$
x_1 \approx \text{generated sample}
$$

Flow matching is already deterministic at inference time because generation is defined through an ODE.

---

# **7. Rectified Flow**

A practical issue with basic flow matching is that learned paths can be curved or inefficient.

This matters because:

- curved paths need more ODE steps
- crossing paths can make the learned velocity field harder to model
- inefficient paths slow down inference

**Rectified flow** tries to make the paths straighter.

Basic reflow idea:

1. Train an initial flow model.
2. Use it to generate pairs:

$$
(z_0, z_1)
$$

where $z_0$ is noise and $z_1$ is the model-generated output.

3. Retrain the model on these paired samples.
4. Repeat if needed.

The goal is to obtain straighter trajectories, which allows faster sampling with fewer Euler steps.

In practice, reflow is often done only a small number of times because repeated reflow can accumulate errors.

---

# **8. Summary**

## **8.1 What Flow Matching Learns**

Flow matching learns a velocity field:

$$
u^\theta_t(x)
$$

This tells a sample how to move from noise to data.

---

## **8.2 Training**

Training is based on simple interpolation:

$$
x_t = (1-t)x_0 + tx_1
$$

and the target velocity:

$$
x_1 - x_0
$$

So the model learns:

$$
u^\theta_t(x_t) \approx x_1 - x_0
$$

---

## **8.3 Inference**

Inference starts from Gaussian noise and solves:

$$
\frac{dx_t}{dt} = u^\theta_t(x_t)
$$

from $t=0$ to $t=1$.

The final point is the generated sample.

---

## **8.4 Main Difference from Diffusion**

DDPM:

$$
\text{learn noise}
$$

Score matching:

$$
\text{learn score}
$$

Flow matching:

$$
\text{learn velocity}
$$

---

# **References**

- <a id="flow-matching"></a> **[1] Flow Matching for Generative Modeling**  
  Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le  
  https://arxiv.org/abs/2210.02747

- <a id="rectified-flow"></a> **[2] Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**  
  Xingchao Liu, Chengyue Gong, Qiang Liu  
  https://arxiv.org/abs/2209.03003

- <a id="stochastic-interpolants"></a> **[3] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions**  
  Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden  
  https://arxiv.org/abs/2303.08797
