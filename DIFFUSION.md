# **Diffusion Notes**


# **Table of Contents**


---

## **0.1 Outline**

- Initial focus will be on the unconditioned generation for simplicity.

<div style="text-align: center;">
  <img src="assets/outline.svg" alt="outline" />
</div>

---

## **0.2 Idea**

- Understand a way to sample an observation from a distribution that is **new**.

<div style="text-align: center;">
  <img src="assets/new_sample.svg" alt="new_sample" />
</div>

---

## **0.3 Generate from Noise**

Why start from noise?
- Inherent randomness
- Noise $\sim$ Gaussian and we love Gaussian.
- Surprisingly simple.

---

## **0.4 Intuition**

Starting from $x_0$, input image:
- Add noise $q(x_t \mid x_{t-1})$ for $T$ steps: $x_1, x_2, ..., x_T$.
- Learn the reverse process $p_\theta(x_{t-1} \mid x_t)$ for $T$ steps: $x_T, x_{T-1}, ..., x_0$.

<div style="text-align: center;">
  <img src="assets/fwd_bwd_combined.svg" alt="forward_backward" />
</div>

---

## **0.5 Dimensionality & Probability Distribution**

There are multiple ways to represent your images. In the end, when we talk about images, we are talking about **vectors**.
So for a 3 channel RGB image: $n = 3 \times H \times W$

$$
x_0 = \begin{pmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{pmatrix}
$$

What this means for the distributions we will be working with:
**Single value $\approx$ vector of values.**

$$
x \sim \mathcal{N}(\mu, \Sigma)
$$

$$
\underbrace{
\begin{pmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{pmatrix}}_{x}
\sim
\mathcal{N}
\underbrace{
\begin{pmatrix}
\mu_{1} \\
\mu_{2} \\
\vdots \\
\mu_{n}
\end{pmatrix}}_{\mu}
,
\underbrace{
\begin{pmatrix}
\Sigma_{1,1} & \Sigma_{1,2} & \cdots & \Sigma_{1,n} \\
\Sigma_{2,1} & \Sigma_{2,2} & \cdots & \Sigma_{2,n} \\
\vdots & \vdots & \ddots & \vdots \\
\Sigma_{n,1} & \Sigma_{n,2} & \cdots & \Sigma_{n,n}
\end{pmatrix}
}_{\Sigma}
$$

**Note**: The covariance matrix $\Sigma$ is symmetric and we will only focus on **isotropic** Gaussians where:

$$
\Sigma = \sigma^2 I = \begin{pmatrix}
\sigma^2 & 0 & \cdots & 0 \\
0 & \sigma^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma^2
\end{pmatrix}
$$

$\sigma^2$ is the variance and $I$ is the identity matrix.

---

## **0.6 Covariance?**

**It is just a quantifier of how two random variables vary in a linear way.**
You may also call this the **joint variability** of two random variables.

<div style="text-align: center;">
  <img src="assets/covariance.png" alt="covariance" width="70%" />
</div>

---

# **1. Forward Process**

- Starting from $x_0$

<div style="text-align: center;">
  <img src="assets/forward_process.gif" alt="forward" />
</div>

---

## **1.1 Representing Noise**

$$
\epsilon = \begin{pmatrix}
\epsilon_{1} \\
\epsilon_{2} \\
\vdots \\
\epsilon_{n}
\end{pmatrix}
\sim \mathcal{N}(0, I)
$$

$\epsilon_i \sim \mathcal{N}(0, 1)$ is **independent** and **identically distributed**.

---

## **1.2 Single Forward Step**

<div style="text-align: center;">
  <img src="assets/forward_step.svg" alt="forward_step" />
</div>

---

## **1.3 Deriving the Forward Process**

At each step, we sample:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I\right)
$$

- $\beta_t$ controls how much noise is added at step $t$
- small $\beta_t$ → little corruption
- large $\beta_t$ → stronger corruption
- $I$ means independent Gaussian noise per pixel/channel

$$x_t = \sqrt{1-\beta_t}\cdot x_{t-1} + \
\sqrt{\beta_{t}}\cdot\epsilon_t, \quad \epsilon \sim \mathcal{N}(0, I)
$$

After reparameterization, $x_t$ can be expressed in terms of $x_0$ and $\epsilon$;

$$
x_t = \sqrt{\bar\alpha_t}\cdot x_{0} + \
\sqrt{1-\bar\alpha_{t}}\cdot\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**But how did we get here?!**

---

## **1.4 Diving Deeper**

Let's define $\alpha_t = 1 - \beta_t$, updated process will become:

$$x_t = \sqrt{\alpha_t}\cdot\underbrace{x_{t-1}}_{\downarrow} + \sqrt{1-\alpha_{t}}\cdot\epsilon_t$$

$$x_{t-1} = \sqrt{\alpha_{t-1}}\cdot{x_{t-2}} + \sqrt{1-\alpha_{t-1}}\cdot\epsilon_{t-1}$$

Replacing $x_{t-1}$ in the first equation:

$$
x_{t} = 
\sqrt{\alpha_t\cdot\alpha_{t-1}}\cdot x_{t-2} + 
\sqrt{\alpha_t\cdot(1-\alpha_{t-1})}\cdot\epsilon_{t-1} + 
\sqrt{1-\alpha_{t}}\cdot\epsilon_t
$$

It actually turns out for any number of independent variables that are drawn from a Gaussian, you can combine the terms together to get a single $\mu$ and $\sigma^2$.

$$x_t = \sqrt{\alpha_t\cdot\alpha_{t-1}}\cdot x_{t-2} + \sqrt\sigma_{combined}\cdot\epsilon$$


**How do we add two independent Gaussian variables together?**

$$\text{Term 1}: \underbrace{\sqrt{\alpha_t\cdot(1-\alpha_{t-1})}}_{\sigma_1}\cdot\epsilon_{t-1}$$

$$\text{Term 2}: \underbrace{\sqrt{1-\alpha_{t}}}_{\sigma_2}\cdot\epsilon_t$$

$$\epsilon_{t-1} \sim \mathcal{N}(0, \sigma_1^2), \quad \epsilon_{t} \sim \mathcal{N}(0, \sigma_2^2) \quad \epsilon_{t-1} \perp \epsilon_t$$

**BIG NOTE: Sum of independent Gaussians variance is the sum of their variances**

$$\sigma^2_{combined} = \sigma_1^2 + \sigma_2^2$$

$$=\alpha_t\cdot(1-\alpha_{t-1}) + (1-\alpha_{t})$$

$$=1-\alpha_t\cdot\alpha_{t-1}$$

Replacing the equation for combined $\sigma^2$ back into the equation for $x_t$:

$$x_t = \sqrt{1-\alpha_t\cdot\alpha_{t-1}}\cdot x_{t-2} + \sqrt{1-\alpha_t\cdot\alpha_{t-1}}\cdot\epsilon$$

If we keep expanding this process, we can generalize that:

$$x_t = \sqrt{\bar\alpha_t}\cdot x_{0} + \sqrt{1-\bar\alpha_{t}}\cdot\epsilon$$

Where:
- $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
- $\epsilon \sim \mathcal{N}(0, I)$

**Key Takeaways**:
- Combining multiple independent Gaussians helped us derive a closed-form expression for $x_t$ in terms of $x_0$ and $\epsilon$.
- We can directly sample $x_t$ at any time step $t$ without any intermediate steps.

---

# **2. Reverse Process**

Going back to the forward-backward diagram:

<div style="text-align: center;">
  <img src="assets/fwd_bwd_combined.svg" alt="forward_backward" />
</div>

---

## **2.0 Objective**

**We want to learn the $p_\theta$**

$$\max_{\theta} \quad p_\theta(x_0)\rightarrow \text{maximise } p_\theta(x_0)$$

$$\max_{\theta} \quad \log p_\theta(x_0)  
\rightarrow \text{maximise the log-likelihood of } x_0 \text{ under } p_\theta$$

$$\max_{\theta} \quad \mathbb{E}_{q(x_{1:T})}\left[\log p_\theta(x_0)\right]
\rightarrow \text{maximise the expectation of } \log p_\theta(x_0) \text{ over the distribution of } x_{1:T}$$


**Rephrasing**: We want to find the model parameters $\theta$ that maximise the likelihood of the observed data under the model $p$

Note: We are using $\log$ because it has nice properties and provides computational stability.

---

## **2.1 Joint Probability Distributions**

For two probability distributions $x_1$ and $x_2$, we can express their relationship with conditional probabilities:

$p(x_1, x_2)$ is either:

$=\underbrace{p(x_2 \mid x_1)}_{x_2 \text{ given } x_1}\times p(x_1)$

$=\underbrace{p(x_1 \mid x_2)}_{x_1 \text{ given } x_2}\times p(x_2)$

## **2.1.1 Joint Probability Distribution - Marginalization**

<div style="text-align: center;">
  <img src="assets/joint_marginalization.gif" alt="joint_marginalization" />
</div>

So, if we want to find the probability distribution of $x_1$ alone, we can marginalize out $x_2$ or vice versa:

$$p(x_1) = \int p(x_1, x_2) dx_2$$

Applying for all the time steps:

<div style="text-align: center;">
  <img src="assets/full_marginalization.gif" alt="full_marginalization" />
</div>

# **TODO**: ADD 3D VISUAL FOR JOINT PROBABILITY DISTRIBUTION

---

## **2.1.2 Joint Probability Distribution - Summarize**

**Joint Probability Distribution**

$$p(x_1, x_2, \ldots, x_T) = p(x_1) \times p(x_2 \mid x_1) \times \cdots \times p(x_T \mid x_{T-1})$$

**Marginalization**

$$p(x_1) = \int p(x_1, x_2, \ldots, x_T) dx_2 \cdots dx_T $$

**Another notation**: $p(x_1, x_2, \ldots, x_T) = p(x_{1:T})$

---

## **2.2 Back to the Objective**

<p align="center"><b>How to compute?</b></p>

$$\log p_\theta(x_0)$$

One idea, *marginalization*: $\log p_\theta(x_0) = \log \int p_\theta(x_0, x_{1:T}) dx_{1:T}$

The problem with this is that marginalizing from noise to clean image for all trajectories is not viable to compute.

---

## **2.3 Possible Strategy on how to compute $\log p_\theta(x_0)$**

If you have already heard, this will be about **ELBO** (Evidence Lower Bound).

1. Derive a **lower bound** for the maximum likelihood objective $\log p_\theta(x_0)$.
2. Expand the lower bound terms
3. Show lower bound is **tractable**, meaning solvable in polynomial time at most.
4. Deduce loss function $\Leftrightarrow$ **training objective**

---

## **2.4 Deriving ELBO**

1. Derive a **lower bound**

$$\mathbb{E}_{x_0\sim q (x_0)} \left(\log p_\theta (x_0)\right) \underbrace{\geq \mathbb{E}_{x_0\sim q(x_{0:T})} \left(\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_{0})} \right)}_{\text{\textbf{ELBO}= \textbf{E}vidence \textbf{Lower} \textbf{BO}und }} $$

A bit scary.

How do we get here?

# TODO: DERIVATION UNTIL JENSEN INEQUALITY

<div style="text-align: center;">
  <img src="assets/jensen_inequality.gif" alt="jensen_inequality" />
</div>

---

# **3. Variationality**


---

# **4. Training**


---

# **5. Inference**


---

# **6. Sampling ---**


# **References**

- <a id="ddpm"></a> **\[1] DDPM**  
  Denoising Diffusion Probabilistic Models  
  https://arxiv.org/abs/2006.11239

- <a id="stanford-diffusion"></a> **\[2] Stanford CME296**  
  Diffusion & Large Vision Models Course  
  https://cme296.stanford.edu/syllabus/

- <a id="intelligent-systems-lab"></a> **\[3] Intelligent Systems Lab**
  Video lectures on probability, distributions, and related topics.
  https://www.youtube.com/@intelligentsystemslab907/videos

 <!-- Actual referred list using [text][referral_number]  -->

[1]: https://arxiv.org/abs/2006.11239
[2]: https://cme296.stanford.edu/syllabus/
[3]: https://www.youtube.com/@intelligentsystemslab907/videos
