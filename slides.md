---
author: 
 - Jan Heiland (MPI Magdeburg)
 - Peter Benner (MPI Magdeburg)
title: Tensor-Galerkin POD for UQ in PDEs with Multivariate Random Parameters
subtitle: virtuelle GAMM 202021 an der Uni Kassel
title-slide-attributes:
    data-background-image: pics/mpi-bridge.gif
parallaxBackgroundImage: pics/csc-en.svg
parallaxBackgroundSize: 1000px 1200px
bibliography: nn-nse-ldlpv-talk.bib
---

# Introduction 

$$\dot x = f(x) + Bu$$

---

## Control of Nonlinear Large-Scale Systems

A general approach would include

 * powerful backends (linear algebra / optimization)
 * model reduction
 * data-driven surrogate models
 * all of it?!

## This talk:

???

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


## Setup
$$
\DeclareMathOperator{\spann}{span}
\DeclareMathOperator{\Re}{Re}
$$

Vice versa:

> Theorem: The $\hat s$-dimensional subspace $\hat S\subset S$ that optimally parametrizes $y\in S\otimes X \otimes W$ in $\hat S \otimes X \otimes W$ is defined by the $\hat s$ leading mode-(1) singular vectors of $\mathbf Y \in \mathbb R^{s \times r \times p}$.


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

* Let $v$ be the velocity solution and 
$$
V =
\begin{bmatrix}
V_1 & V_2 & \dotsm & V_r
\end{bmatrix},
$$
say, a *POD* basis with $v(t) \approx VV^Tv(t)=:\tilde v(t)$.

* Then $\rho(v(t)) = V^Tv(t)$ is a parametrization

* and with $\tilde v = VV^Tv = V\rho = \sum_{k=1}^rV_k\rho_k$

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

## Convolution Neural Networks for NSE

1. Consider solution snapshots $v(t_k)$ as pictures
2. Learn convolutional kernels to extract relevant features
3. While extracting features, reduce the dimensions
4. Encode $v(t_k)$ in a low-dimensional $\rho_k$.

## Example Architecture Implementation

Comes here

---

Input: 

 * velocity snapshots $v_i$ of an FEM simulation with $n=50'000$ degrees of freedom
 * interpolated to two `H x W` pictures -- makes a `2 x H x W` tensor

## Train for minimizing:
$$
\| v_i - VW\rho(v_i)\|^2_M
$$
which includes

 * the POD modes $V\in \mathbb R^{n\times k}$
 * a learned weight matrix $W\in \mathbb R^{k\times r}$
 * the mass matrix $M$ of the FEM discretization.

## Going PINN

## Results

Averaged (nonlinear) projection error:

|| CNN | POD |
|---:|:----:|:----:|
|`k=5`| `1e-2` | `1e-3` |


# Conclusion

## ... and Outlook

 * Multidimensional Galerkin POD applies naturally for FEM/PCE discretizations.

 * Significant savings of computation time and memory.

 * Outlook: Optimal control and time dependent problems.

. . .

Thank You!

## Resources

* Submitted to *Int. J. Numerical Methods in Engineering*

* Preprint: [arxiv:2009.01055](https://arxiv.org/abs/2009.01055)

* Code: [doi:10.5281/zenodo.4005724](https://doi.org/10.5281/zenodo.4005724)

---

---
nocite: |
  @SoiG04, @BabTZ04
...
