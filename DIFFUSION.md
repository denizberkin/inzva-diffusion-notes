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

- Term 1 $x: \underbrace{\sqrt{\alpha_t\cdot(1-\alpha_{t-1})}}_{\sigma_1}\cdot\epsilon_{t-1}$

- Term 2 $y: \underbrace{\sqrt{1-\alpha_{t}}}_{\sigma_2}\cdot\epsilon_t$

$$\epsilon_{t-1} \sim \mathcal{N}(0, \sigma_1^2), \quad \epsilon_{t} \sim \mathcal{N}(0, \sigma_2^2) \quad \epsilon_{t-1} \perp \epsilon_t$$

**BIG NOTE: Sum of independent Gaussians variance is the sum of their variances**

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

- $\max\limits_{\theta} \quad p_\theta(x_0)\rightarrow \text{ maximise } p_\theta$ 

- $\max\limits_{\theta} \quad \log p_\theta(x_0)\rightarrow \text{ maximise log-likelihood of } p_\theta$

- $\max\limits_{\theta} \quad \mathbb{E}_{q(x_{1:T})}[\log p_\theta(x_0)]$ $\rightarrow \text{ maximise the expectation of } \log p_\theta(x_0) \text{ over the distribution of } x_{1:T}$


**Rephrasing**: We want to find the model parameters $\theta$ that maximise the likelihood of the observed data under the model $p$

Note: We are using $log$ because it has nice properties and provides computational stability.

---

## **2.1 Joint Probability Distributions**

For two probability distributions $x_1$ and $x_2$, we can express their relationship with conditional probabilities:

$p(x_1, x_2)$ is either:
- $=\underbrace{p(x_2 \mid x_1)}_{x_2 \text{ given } x_1}\times p(x_1)$
- $=\underbrace{p(x_1 \mid x_2)}_{x_1 \text{ given } x_2}\times p(x_2)$

---

# **3. Variationality**


---

# **4. Training**


---

# **5. Inference**


---

# **6. Sampling --- **


# **References**

- <a id="ddpm"></a> **\[1] DDPM**  
  Denoising Diffusion Probabilistic Models  
  https://arxiv.org/abs/2006.11239

- <a id="stanford-diffusion"></a> **\[2] Stanford CME296**  
  Diffusion & Large Vision Models Course  
  https://cme296.stanford.edu/syllabus/



 <!-- Actual referred list using [text][referral_number]  -->

[1]: https://arxiv.org/abs/2006.11239
[2]: https://cme296.stanford.edu/syllabus/
