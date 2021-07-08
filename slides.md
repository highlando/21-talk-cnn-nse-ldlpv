---
author: 
 - Jan Heiland & Peter Benner (MPI Magdeburg)
title: Convolutional AEs for low-dimensional parameterizations of Navier-Stokes flow
subtitle: IFAC Seminar -- Data-driven Methods in Control -- 2021
title-slide-attributes:
    data-background-image: pics/mpi-bridge.gif
parallaxBackgroundImage: pics/csc-en.svg
parallaxBackgroundSize: 1000px 1200px
bibliography: nn-nse-ldlpv-talk.bib
---

# Introduction 

$$\dot x = f(x) + Bu$$

---

## {data-background-video="pics/triple_swingup_slomo.MP4"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Control of an inverted pendulum

 * 9 degrees of freedom
 * but nonlinear controller.

:::

## {data-background-image="pics/dbrc-v_Re50_stst_cm-bbw.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Stabilization of a laminar flow

 * 50'000 degrees of freedom
 * but linear regulator.

:::

## Control of Nonlinear & Large-Scale Systems

A general approach would include

 * powerful backends (linear algebra / optimization)
 * exploitation of general structures
 * data-driven surrogate models
 * all of it?!

# SDC Representation
$$
\dot x = [A(x)]\,x + Bu
$$

---

 * Under mild conditions, the flow $f(x)$ can be factorized
$$
\dot x = [A(x)]\,x + Bu
$$ 
-- a *state dependent coefficient* system --
with some $$A\colon \mathbb R^{n} \to \mathbb R^{n\times n}.$$ 

 * Control through a *state-dependent state-feedback law*
 $$
 u=-[B^*P(x)]\,x.
 $$

## Nonlinear SDRE Feedback

 * Set 
 $$
 u=-[B^TP(x)]\,x.
 $$
 * with $P(x)$ as the solution to the state-dependent Riccati equation
$$
A(x)^TP + PA(x) - PBB^TP + C^TC=0
$$
 
 * the system $$\dot x = f(x) + Bu \;=[A(x)- BB^TP(x)]\,x$$ can be controlled towards an equilibrium; see, e.g., @BanLT07.

## Linear Updates as an Alternative

**Theorem** @BenH18 

* ...

* If $P_0$ is the Riccati solution for $x=x_0$

* and if $E$ solves the **linear** equation
$$A(x)E + E(A(x_0)-BB^TP_0)=A(x_0)-A(x)$$

* with $\|E\| \leq \epsilon < 1$,

* then $u=-B^TP_0(I+E)^{-1}$ stabilizes the system.

$$
\DeclareMathOperator{\spann}{span}
\DeclareMathOperator{\Re}{Re}
$$

# LPV Representation

$$
\dot x \approx [A_0+\Sigma \,\rho_k(x)A_k]\, x + Bu
$$

---

The *linear parameter varying* (LPV) representation/approximation
$$
\dot x = f(x) + Bu = [\tilde A(\rho(x))]\,x + Bu \approx  [A_0+\Sigma \,\rho_k(x)A_k]\, x + Bu
$$
with **affine parameter dependency** can be exploited for designing nonlinear controller through scheduling.

---

## Scheduling of $H_\infty$ Controllers

* If $\rho(x)\in \mathbb R^{k}$ can be confined to a bounded polygon,

* there is globally stabilizing $H_\infty$ controller

* that can be computed

* through solving $k$ **coupled LMI** in the size of the state dimension;

see @ApkGB95 .


## Series Expansion of SDRE Solution

For $A(x)=\sum_{k=1}^r\rho_k(x)A_k$, the solution $P$ to the SDRE
$$
A(x)^TP + PA(x) - PBB^TP + C^TC=0
$$
can be expanded in a series
$$
P(x) = P_0 + \sum_{|\alpha| > 0}\rho(x)^{(\alpha)}P_{\alpha}
$$
where $P_0$ solves a Riccati equation and $P_\alpha$ solve Lyapunov (linear!) equations;

see @BeeTB00.

## We see

Manifold opportunities if only $k$ was small.

# Low-dimensional LPV

**Approximation** of *Navier-Stokes Equations* by *Convolutional Neural Networks*

---


## {data-background-image="pics/cw-Re60-t161-cm-bbw.png" data-background-size="cover"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}
The *Navier-Stokes* equations

$$
\dot v + (v\cdot \nabla) v- \frac{1}{\Re}\Delta v + \nabla p= f, 
$$

$$
\nabla \cdot v = 0.
$$
:::

---

* Let $v$ be the velocity solution and let
$$
V =
\begin{bmatrix}
V_1 & V_2 & \dotsm & V_r
\end{bmatrix}
$$
be a, say, *POD* basis with $$v(t) \approx VV^Tv(t)=:\tilde v(t),$$

* then $$\rho(v(t)) = V^Tv(t)$$ is a parametrization.

---

* And with $$\tilde v = VV^Tv = V\rho = \sum_{k=1}^rV_k\rho_k,$$

* the NSE has the low-dimensional LPV representation via
$$
(v\cdot \nabla) v \approx (\tilde v \cdot \nabla) v = [\sum_{k=1}^r\rho_k(V_k\cdot \nabla)]\,v.
$$

## Question

Can we do better than POD?

## {data-background-image="pics/scrsho-lee-cb.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Lee/Carlberg (2019): *MOR of dynamical systems on nonlinear manifolds using deep convolutional autoencoders*
:::

## {data-background-image="pics/scrsho-choi.png"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Kim/Choi/Widemann/Zodi (2020): *Efficient nonlinear manifold reduced order model*
:::

## Convolution Autoencoders for NSE

1. Consider solution snapshots $v(t_k)$ as pictures.

2. Learn convolutional kernels to extract relevant features.

3. While extracting the features, we reduce the dimensions.

4. Encode $v(t_k)$ in a low-dimensional $\rho_k$.

## Our Example Architecture Implementation


## {data-background-image="pics/nse-cnn.jpg"}

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

 * A number of convolutional layers for feature extraction and reduction

 * A full linear layer with nonlinear activation for the final encoding $\rho\in \mathbb R^{r}$

 * A linear layer (w/o activation) that expands $\rho \to \tilde \rho\in \mathbb R^{k}$.

:::

## Input:

 * Velocity snapshots $v_i$ of an FEM simulation with $$n=50'000$$ degrees of freedom

 * interpolated to two pictures with `63x95` pixels each

 * makes a `2x63x69` tensor. 

## Training for minimizing:
$$
\| v_i - VW\rho(v_i)\|^2_M
$$
which includes

 1. the POD modes $V\in \mathbb R^{n\times k}$,

 2. a learned weight matrix $W\in \mathbb R^{k\times r}\colon \rho \mapsto \tilde \rho$,

 3. the mass matrix $M$ of the FEM discretization.

## Going PINN

Outlook: 
the induced low-dimensional affine-linear LPV representation of the convection
$$\| (v_i\cdot \nabla)v_i - (VW\rho_i \cdot \nabla )v_i\|^2_{M^{-1}}$$
as the target of the optimization.

Implementation issues:

 * Include FEM operators while
 * maintaining the *backward* mode of the training.

## Results

## {data-background-image="pics/plot-one.png" data-background-size="70%"}

<!--
Averaged (nonlinear) projection error:

|| CNN | POD |
|---:|:----:|:----:|
|`r=3`  | `0.0416` | `0.0645` | 
|`r=5`  | `0.0248` | `0.0296` |
|`r=8`  | `0.0176` | `0.0179` |
|`r=12` | `0.0092` | `0.0086` |
-->

. . .

::: {style="position: absolute; width: 60%; right: 0; box-shadow: 0 1px 4px rgba(0,0,0,0.5), 0 5px 25px rgba(0,0,0,0.2); background-color: rgba(0, 0, 0, 0.9); color: #fff; padding: 20px; font-size: 40px; text-align: left;"}

Simulation parameters:

 * Cylinder wake at $\Re=40$, time in $[0, 8]$
 * `1000` snapshots/data points 
 * 2D-CNN with 4 layers
 * `kernelsize, stride = 5, 2`.
 * `batch_size = 40`

:::

# Conclusion

## ... and Outlook

 * LPV with affine-linear dependencies are attractive if only $k$ is small.

 * Proof of concept that CNN can *improve* POD at very low dimensions.

 * Next: Include the parametrized convection in the training.

 * Outlook: Use for nonlinear controller design.

. . .

Thank You!

---
