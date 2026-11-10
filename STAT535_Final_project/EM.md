# EM Algorithm

Below we will implement the EM algorithm, an iterative algorithm used to estimate the parameters in the likelihood of $y$. Recall that our model is given by

$$
\mathbf{Y} = X\boldsymbol{\beta} + \mathbf{w} + \boldsymbol{\epsilon},
$$

where

- $X \in \mathbb{R}^{n \times p}$ is the design matrix of covariates.
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the vector of regression coefficients.
- $\mathbf{w} \sim N_n(0,\Sigma_\theta)$ is a stationary isotropic Gaussian process with Matérn covariogram, where $\theta = (\sigma^2,\phi)^\top$.
- $\boldsymbol{\epsilon} \sim N_n(0,\tau^2 I_n)$ represents the nugget effect.

The parameter of interest is hence $\vec\eta=(\vec\beta, \sigma^2,\phi,\tau^2)^\top$, where $\sigma^2$ is the spatial variance (sill), $\phi$ is the range-scale parameter (range), and $\tau^2$ is the nugget effect. Hence we have

$$
\mathbf{Y} | \vec\eta \sim N_n(X\boldsymbol{\beta},\Sigma_\theta+\tau^2I_n).
$$

In EM algorithm, we introduce the spatial effect $\mathbf{w}$ as a latent variable (missing data), then

$$
\mathbf{Y} | \mathbf{w},\vec\eta \sim N_n(X\boldsymbol{\beta}+\mathbf{w}, \tau^2I_n),
$$

and we can write the joint likelihood as

$$
f(\mathbf{Y}, \mathbf{w}; \vec\eta) = f(\mathbf{Y} | \mathbf{w};\vec\eta)\cdot f(\mathbf{w};\vec\eta),
$$

therefore the log-likelihood of the parameters $\vec\eta$ can be written as

$$
\ell(\vec\eta) = \log f(\mathbf{Y} | \mathbf{w}; \vec\eta) + \log f(\mathbf{w};\vec\eta).
$$

Since $\mathbf{Y}|\mathbf{w};\vec\eta$ and $\mathbf{w}$ are both distributed according to multivariate Gaussian, their log-likelihood are given by

$$
\log f(\mathbf{Y} | \mathbf{w};\vec\theta) = -\frac{n}{2}\log(2\pi)-\frac{n}{2}\log\tau^2 - \frac{1}{2\tau^2}\Vert \mathbf{Y} - X\boldsymbol{\beta}-\mathbf{w}\Vert^2;
$$

$$
\log f(\mathbf{w};\vec\eta) = -\frac{n}{2}\log(2\pi) - \frac{1}{2} \log| {\Sigma}_\theta| - \frac{1}{2} \mathbf{w}^\top \Sigma_\theta^{-1} \mathbf{w}.
$$

The EM algorithm is 