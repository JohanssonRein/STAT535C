# EM Algorithm

## Basic set up

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

The EM algorithm contains an E (Expectation) step and a M (Maximization) step. 

### E Step:

At iteration $k$, suppose the current estimates of the parameters are given by

$$
 \vec\eta^{(k)} = (\boldsymbol{\beta}^{(k)}, (\sigma^2)^{(k)}, \phi^{(k)}, (\tau^2)^{(k)}),
$$

The E step computes the following:

$$
Q(\vec\eta | \vec\eta^{(k)}) = E_{\mathbf{w}|\mathbf{Y}} [ \ell(\vec\eta)|\mathbf{Y}]\Big|_{\vec\eta^{(k)}},
$$

which is the expectation taken with respect to the conditional distribution of $\mathbf{w}|\mathbf{Y}$, evaluated at the current parameter estimate $\vec\eta^{(k)}$.

### M Step:

In this step we update the parameters $\vec\eta^{(k+1)}$ via

$$
\vec\eta^{(k+1)} = \argmax_{\vec\eta} Q(\vec\eta | \vec\eta^{(k)}).
$$


## Modified EM


The standard EM algorithm for our model requires repeated operations involving the covariance matrix

$$
V_\theta=\Sigma_\theta+\tau^2I_n.
$$

For large spatial datasets, the covariance matrix $\Sigma_\theta$ is generally dense. Direct matrix factorization requires $O(n^3)$ computational time and $O(n^2)$ memory, making standard EM computationally expensive.

To address this problem, we combine **covariance tapering**, **sparse Cholesky factorization**, and **randomized trace estimation** to construct a computationally scalable EM algorithm. Furthermore, to get a closed-form update formula, we require the range parameter $\phi$ to be known. This can be done by fixing range parameter using its empirical estimate. Or we may extend the M-step to estimate $\phi$ jointly alongside with other parameters, but then we will not get the closed-form formula derived below.


### Covariance Tapering

Covariance tapering approximates the original spatial covariance by multiplying it with a compactly supported correlation function.

Let $T_r(h)$ denote a taper function with taper radius $r$. We consider the Wendland taper

$$
T_r(h)=
\begin{cases}
(1-h/r)^4(1+4h/r), & 0\leq h<r,\\
0, & h\geq r.
\end{cases}
$$

The tapered covariance matrix is

$$
\widetilde{\Sigma}_\theta=\Sigma_\theta\circ T_r,

$$

where $\circ$ denotes the Hadamard product. Since $T_r(h)=0$ for $h\geq r$, observations separated by more than the taper radius have zero covariance under the approximated model. Consequently, $\widetilde{\Sigma}_\theta$ is sparse. We write

$$
\widetilde{\Sigma}_\theta=\sigma^2R,
$$

where $R$ is the tapered Matérn correlation matrix.

### Sparse Cholesky factorization

In the E-step, the conditional distribution of the latent spatial effects is

$$
\mathbf w\mid\mathbf Y,\Theta^{(k)}
\sim N_n(\mathbf m_k,C_k).
$$

Define

$$
V_k=\sigma_k^2R+\tau_k^2I_n,
\qquad
\mathbf r_k=\mathbf Y-X\boldsymbol\beta_k.
$$

Then conditional mean and covariance can be expressed as

$$
\mathbf m_k=\mathbf r_k-\tau_k^2V_k^{-1}\mathbf r_k, \quad C_k=\tau_k^2I_n-\tau_k^4V_k^{-1}.
$$


Although $V_k$ is sparse, its inverse is generally dense. Therefore, we avoid explicitly calculating $V_k^{-1}$, and use Cholesky decomposition instead. We solve linear systems of the form

$$
V_k\mathbf a=\mathbf r_k.
$$


### Hutchinson's Randomized Trace Estimator

The M-step requires quantities involving the trace of the conditional covariance, particularly

$$
\operatorname{tr}(C_k)
=
n\tau_k^2-\tau_k^4\operatorname{tr}(V_k^{-1}).
$$

Computing $\operatorname{tr}(V_k^{-1})$ exactly can be expensive.
We use the Hutchinson's randomized trace estimator, the idea is as follow:

Let $\mathbf z_1,\ldots,\mathbf z_K$ be independent Rademacher random vectors (i.e, whose entries independently take values $-1$ and $1$ with equal probability).

The Monte-Carlo trace estimator is given by

$$
\widehat{\operatorname{tr}(V_k^{-1})}
=
\frac1K\sum_{j=1}^K
\mathbf z_j^\top V_k^{-1}\mathbf z_j,
$$

where for each $\mathbf z_j$, we solve
$
V_k\mathbf a_j=\mathbf z_j
$
using Cholesky factorization.

This estimator is unbiased for a fixed matrix, and its Monte Carlo variance decreases as $K$ increases.

### Modified EM Updates

Using the tapered covariance and randomized trace approximation, the EM algorithm proceeds as follows.

#### E-step

Compute the conditional mean 
$
\mathbf m_k=\mathbf r_k-\tau_k^2V_k^{-1}\mathbf r_k
$
and estimate
$
t_k=\operatorname{tr}(V_k^{-1})
$
using Hutchinson's estimator.

#### M-step

The regression coefficients are updated according to

$$
\boldsymbol\beta_{k+1}
=
(X^\top X)^{-1}X^\top(\mathbf Y-\mathbf m_k).
$$

The spatial variance (sill) $\sigma^2$ is updated according to

$$
\sigma_{k+1}^2
=
\frac{\sigma_k^2}{n}
\left[
\mathbf m_k^\top V_k^{-1}\mathbf r_k
+
\tau_k^2\widehat t_k
\right].
$$

The nugget $\tau^2$ is updated according to

$$
\tau_{k+1}^2
=
\frac1n
\left[
\|\mathbf Y-X\boldsymbol\beta_{k+1}-\mathbf m_k\|^2
+n\tau_k^2-\tau_k^4\widehat t_k
\right],
$$

where $\widehat t_k$ is the randomized estimate of $\operatorname{tr}(V_k^{-1})$.

The algorithm repeats these updates until the parameter estimates stabilize or the number of maximum iteration is achieved.

### Remarks

In our initial implementation, the Matérn range parameter $\phi$ and smoothness parameter $\nu$ are fixed at their true simulation values, while $\boldsymbol\beta$, $\sigma^2$, and $\tau^2$ are estimated.

The proposed approach offers two computational advantages:

1. **Covariance tapering:** Produces a sparse covariance matrix, enabling sparse Cholesky factorization and efficient linear solves.
2. **Randomized trace estimation:** Avoids constructing dense inverse and conditional covariance matrices.

However, covariance tapering changes the original covariance model and can introduce approximation bias. Additionally, randomized trace estimation introduces Monte Carlo error into the EM updates.

Consequently, the resulting procedure is an approximate EM algorithm for the tapered Gaussian process model. Unlike exact EM, the approximate iterations are not guaranteed to monotonically increase the observed-data likelihood.

In our simulation study, we evaluate the method by comparing parameter estimates with their known true values, examining convergence, and recording computational runtime. We also investigate the influence of the taper radius and number of random trace probes on estimation accuracy and computational efficiency.

## Implement our algorithm in R

Below is the R code to implement the EM algorithm for our simulated spatial data. We assume all the code in [Simulation.md](Simulation.md) was already implemented. Additionally, we need $2$ more packages:

```r
install.packages(c("Matrix", "dbscan"))

library(Matrix)
library(dbscan)
```

The following code is used to construct the tapered covariance matrix:

```r
# Prepare the simulated data
Y <- sim_data$Y

X <- model.matrix(
  ~ X1 + X2 + X3,
  data = sim_data
)

coords <- as.matrix(sim_data[, c("x", "y")])
n <- length(Y)

# Fixed covariance parameters
phi <- 0.2
nu <- 1

# Taper radius, measured in coordinate units
taper_range <- 0.06

# Matérn correlation
matern_corr <- function(d, phi, nu) {
  z <- d / phi
  out <- numeric(length(z))

  out[z == 0] <- 1

  positive <- z > 0
  out[positive] <-
    (2^(1 - nu) / gamma(nu)) *
    z[positive]^nu *
    besselK(z[positive], nu)

  out
}

# Wendland taper
wendland <- function(d, taper_range) {
  u <- d / taper_range

  ifelse(
    u < 1,
    (1 - u)^4 * (1 + 4 * u),
    0
  )
}

# Find all neighbors within the taper radius
nn <- dbscan::frNN(
  coords,
  eps = taper_range,
  search = "kdtree"
)

# Convert neighbor lists into index vectors
i_all <- rep(seq_len(n), lengths(nn$id))
j_all <- unlist(nn$id, use.names = FALSE)
d_all <- unlist(nn$dist, use.names = FALSE)

# Keep each undirected pair only once
keep <- j_all > i_all

i <- i_all[keep]
j <- j_all[keep]
d <- d_all[keep]

# Tapered Matérn correlations
corr <- matern_corr(d, phi, nu) *
  wendland(d, taper_range)

# Build the sparse symmetric correlation matrix
R_taper <- Matrix::sparseMatrix(
  i = c(seq_len(n), i, j),
  j = c(seq_len(n), j, i),
  x = c(rep(1, n), corr, corr),
  dims = c(n, n)
)

R_taper <- Matrix::forceSymmetric(R_taper)

# Check the sparsity
cat("Dimensions:", dim(R_taper), "\n")
cat("Nonzero entries:", Matrix::nnzero(R_taper), "\n")
cat(
  "Sparsity:",
  1 - Matrix::nnzero(R_taper) / n^2,
  "\n"
)

# Verify positive definiteness
chol_R <- Matrix::Cholesky(
  R_taper,
  LDL = FALSE
)
```

Next, we implement the tapered EM:

```r

EM_taper <- function(Y, X, R,
                     sigma2_init = 1,
                     tau2_init = 0.2,
                     max_iter = 1000,
                     tol = 1e-5,
                     n_probes = 32,
                     seed = 123) {

  n <- length(Y)

  stopifnot(
    nrow(X) == n,
    nrow(R) == n,
    ncol(R) == n,
    sigma2_init > 0,
    tau2_init > 0,
    n_probes >= 1
  )

  # OLS initialization
  XtX <- crossprod(X)
  beta <- as.vector(qr.solve(X, Y))

  sigma2 <- sigma2_init
  tau2 <- tau2_init

  # Fixed Rademacher vectors for trace estimation
  set.seed(seed)

  Z <- matrix(
    sample(c(-1, 1), n * n_probes, replace = TRUE),
    nrow = n,
    ncol = n_probes
  )

  history <- data.frame(
    iteration = integer(),
    sigma2 = numeric(),
    tau2 = numeric(),
    change = numeric()
  )

  converged <- FALSE

  for (iter in seq_len(max_iter)) {

    old_par <- c(beta, sigma2, tau2)

    # -----------------------------------
    # E-step
    # -----------------------------------

    # Sparse marginal covariance
    V <- sigma2 * R + Matrix::Diagonal(n, tau2)

    # Sparse Cholesky factorization
    V_chol <- Matrix::Cholesky(
      V,
      LDL = FALSE,
      super = TRUE
    )

    # a = V^(-1)(Y - X beta)
    residual <- Y - as.vector(X %*% beta)

    a <- as.vector(
      solve(V_chol, residual)
    )

    # Posterior mean of latent spatial field
    m_post <- residual - tau2 * a

    # Hutchinson estimate of tr(V^(-1))
    V_inv_Z <- solve(V_chol, Z)

    trace_V_inv <- sum(Z * V_inv_Z) / n_probes

    # -----------------------------------
    # M-step
    # -----------------------------------

    # Update regression coefficients
    beta_new <- as.vector(
      solve(
        XtX,
        crossprod(X, Y - m_post)
      )
    )

    # Update spatial variance
    sigma2_new <- (
      sigma2 * sum(m_post * a) +
      tau2 * sigma2 * trace_V_inv
    ) / n

    # Update nugget variance
    error <- Y -
      as.vector(X %*% beta_new) -
      m_post

    trace_C <- tau2 * n -
      tau2^2 * trace_V_inv

    tau2_new <- (
      sum(error^2) + trace_C
    ) / n

    if (!is.finite(sigma2_new) ||
        !is.finite(tau2_new) ||
        sigma2_new <= 0 ||
        tau2_new <= 0) {
      stop("Invalid variance update. Increase n_probes or check R.")
    }

    # -----------------------------------
    # Update and monitor convergence
    # -----------------------------------

    beta <- beta_new
    sigma2 <- sigma2_new
    tau2 <- tau2_new

    new_par <- c(beta, sigma2, tau2)

    change <- max(
      abs(new_par - old_par) /
        pmax(1, abs(old_par))
    )

    history <- rbind(
      history,
      data.frame(
        iteration = iter,
        sigma2 = sigma2,
        tau2 = tau2,
        change = change
      )
    )

    if (iter %% 10 == 0 || iter == 1) {
      cat(
        "Iteration:", iter,
        "| sigma2:", round(sigma2, 4),
        "| tau2:", round(tau2, 4),
        "| change:", signif(change, 3),
        "\n"
      )
    }

    if (change < tol) {
      converged <- TRUE
      break
    }
  }

  # Recompute posterior mean using final estimates
  V_final <- sigma2 * R + Matrix::Diagonal(n, tau2)

  V_chol_final <- Matrix::Cholesky(
    V_final,
    LDL = FALSE,
    super = TRUE
  )

  residual_final <- Y - as.vector(X %*% beta)

  a_final <- as.vector(
    solve(V_chol_final, residual_final)
  )

  w_hat <- residual_final - tau2 * a_final

  list(
    beta = beta,
    sigma2 = sigma2,
    tau2 = tau2,
    w_hat = w_hat,
    history = history,
    iterations = iter,
    converged = converged
  )
}
```

Now, run the EM algorithm with $1000$ max-iterations:

```r
system.time({

  fit_EM_taper <- EM_taper(
    Y = Y,
    X = X,
    R = R_taper,
    sigma2_init = 1,
    tau2_init = 0.2,
    max_iter = 1000,
    tol = 1e-5,
    n_probes = 32
  )

})
```

Check our results:
```r
# Regression coefficients
data.frame(
  Parameter = colnames(X),
  True = c(2, 0.5, -1.2, 0.8),
  Estimated = fit_EM_taper$beta
)

# Covariance parameters
data.frame(
  Parameter = c("sigma2", "tau2"),
  True = c(1, 0.1),
  Estimated = c(
    fit_EM_taper$sigma2,
    fit_EM_taper$tau2
  )
)

fit_EM_taper$converged
fit_EM_taper$iterations
```

## Improved Covariance Tapering: Double Tapering

In our previous version, we only applied a taper to the covariance matrix. Recall our model

$$
\mathbf{Y} =X\boldsymbol\beta+\mathbf{w}+\epsilon,
$$

where $w\sim N(0,\sigma^2R_\phi)$ and $\epsilon\sim N(0,\tau^2I_n)$. Let $V_\theta=\sigma^2R_\phi+\tau^2I_n$ denote the covariance matrix, and let $T$ be a compactly supported taper matrix.

The double-taper negative log-likelihood criterion, up to an additive constant, is

$$
\frac12\log|V_\theta\circ T|
+
\frac12\operatorname{tr}\left[
\big(rr^\top\circ T\big)
(V_\theta\circ T)^{-1}
\right]
$$

where $r=Y-X\beta$, $\theta=(\sigma^2,\tau^2)$, and $\circ$ denotes the Hadamard (elementwise) product.

The first taper makes the covariance matrix sparse, allowing efficient matrix factorizations. The second taper corrects the expected-score distortion introduced by single tapering, improving covariance-parameter estimation. We estimate $\theta=(\sigma^2,\tau^2)$ while fixing the Matérn range $\phi$ and smoothness $\nu$.

Our implementation combines three numerical techniques:

- Sparse Cholesky factorization: Computes the log determinant and solves linear systems without explicitly inverting the covariance matrix.

- Hutchinson's trace estimator: Approximates the trace term using $K$ independent Rademacher random vectors:

$$
\operatorname{tr}(BA^{-1})
\approx
\frac1K\sum_{k=1}^{K}z_k^\top BA^{-1}z_k,
$$

where $A=V_\theta\circ T$ and $B=rr^\top\circ T$.

- L-BFGS-B optimization: Minimizes the approximate criterion over $(\log\sigma^2,\log\tau^2)$ subject to parameter bounds. Since analytical gradients are not supplied, R's optim() computes finite-difference gradients.

The regression coefficients $\beta$ are initially estimated using ordinary least squares (OLS). After estimating the covariance parameters, feasible generalized least squares (GLS) is used to obtain updated regression coefficient estimates.

Below we provide the implementation of double tapering for our simulated data. For model diagnotics, we randomly split our simulated data into an $80\%$ training, and a $20\%$ testing set.

```r
#Creating the training and testing set
set.seed(123)

n <- nrow(sim_data)

train_idx <- sample(
  seq_len(n),
  size = floor(0.8 * n)
)

sim_train <- sim_data[train_idx, ]
sim_test  <- sim_data[-train_idx, ]

Y_train <- sim_train$Y
Y_test  <- sim_test$Y

X_train <- model.matrix(
  ~ X1 + X2 + X3,
  data = sim_train
)

X_test <- model.matrix(
  ~ X1 + X2 + X3,
  data = sim_test
)

coords_train <- as.matrix(
  sim_train[, c("x", "y")]
)

coords_test <- as.matrix(
  sim_test[, c("x", "y")]
)

n_train <- length(Y_train)
n_test <- length(Y_test)
```

Then the following code will estimate the paramaters on our training set.
```r
# Matérn parameters
phi <- 0.2
nu <- 1
taper_range <- 0.3

# Find training-location neighbors
nn <- dbscan::frNN(
  coords_train,
  eps = taper_range,
  search = "kdtree",
  sort = FALSE
)

i_all <- rep(seq_len(n_train), lengths(nn$id))
j_all <- unlist(nn$id, use.names = FALSE)
d_all <- unlist(nn$dist, use.names = FALSE)

keep <- j_all > i_all

i <- i_all[keep]
j <- j_all[keep]
d <- d_all[keep]

taper_values <- wendland(d, taper_range)

corr <- matern_corr(d, phi, nu) * taper_values

R_train <- Matrix::sparseMatrix(
  i = c(seq_len(n_train), i, j),
  j = c(seq_len(n_train), j, i),
  x = c(rep(1, n_train), corr, corr),
  dims = c(n_train, n_train)
)

T_train <- Matrix::sparseMatrix(
  i = c(seq_len(n_train), i, j),
  j = c(seq_len(n_train), j, i),
  x = c(rep(1, n_train), taper_values, taper_values),
  dims = c(n_train, n_train)
)

R_train <- Matrix::forceSymmetric(R_train)
T_train <- Matrix::forceSymmetric(T_train)

#refit
logdet <- as.numeric(
  determinant(
    V_chol,
    logarithm = TRUE,
    sqrt = FALSE
  )$modulus
)

# Initial regression
beta_ols <- as.vector(
  qr.solve(X_train, Y_train)
)

r_train <- Y_train -
  as.vector(X_train %*% beta_ols)

# Hutchinson probes
set.seed(456)

K <- 64

Z <- matrix(
  sample(c(-1, 1),
         n_train * K,
         replace = TRUE),
  nrow = n_train,
  ncol = K
)

# Start total runtime clock
start_time <- proc.time()[["elapsed"]]

fit_two_taper_train <- optim(
  par = log(c(1, 0.1)),
  fn = two_taper_objective,
  r = r_train,
  R_taper = R_train,
  T_mat = T_train,
  Z = Z,
  method = "L-BFGS-B",
  lower = log(c(1e-4, 1e-4)),
  upper = log(c(10, 10)),
  control = list(maxit = 100)
)

theta_hat <- exp(fit_two_taper_train$par)

names(theta_hat) <- c("sigma2", "tau2")

theta_hat
fit_two_taper_train$convergence
```

For prediction, run the following code:
```r
#Prediction
# Estimated covariance parameters
sigma2_hat <- unname(theta_hat["sigma2"])
tau2_hat <- unname(theta_hat["tau2"])

# Training covariance
V_train <- sigma2_hat * R_train +
  Matrix::Diagonal(n_train, tau2_hat)

V_chol <- Matrix::Cholesky(
  V_train,
  LDL = FALSE,
  super = TRUE
)

# Build cross-covariance between test and train locations
nn_test <- dbscan::frNN(
  coords_train,
  eps = taper_range,
  query = coords_test,
  search = "kdtree",
  sort = FALSE
)

# Row: training point; column: test point
train_id <- unlist(nn_test$id, use.names = FALSE)
test_id <- rep(seq_len(n_test), lengths(nn_test$id))
dist_cross <- unlist(nn_test$dist, use.names = FALSE)

cross_value <- sigma2_hat *
  matern_corr(dist_cross, phi, nu) *
  wendland(dist_cross, taper_range)

C_cross <- Matrix::sparseMatrix(
  i = train_id,
  j = test_id,
  x = cross_value,
  dims = c(n_train, n_test)
)

# Compute V^{-1}(Y - X beta)
residual_train <- Y_train -
  as.vector(X_train %*% beta_ols)

a <- as.vector(
  solve(V_chol, residual_train)
)

# Predictive means
mu_pred <- as.vector(
  X_test %*% beta_ols +
    crossprod(C_cross, a)
)

# Predictive variances, computed in batches
var_pred <- numeric(n_test)

batch_size <- 100

for (start in seq(1, n_test, by = batch_size)) {
  
  end <- min(start + batch_size - 1, n_test)
  
  idx <- start:end
  
  C_batch <- as.matrix(
    C_cross[, idx, drop = FALSE]
  )
  
  # Solve V A = C_batch
  A <- as.matrix(
    solve(V_chol, C_batch)
  )
  
  # Diagonal of C_batch^T V^-1 C_batch
  reduction <- colSums(C_batch * A)
  
  var_pred[idx] <- sigma2_hat + tau2_hat -
    reduction
}

# Check variance values before taking square roots
stopifnot(
  all(is.finite(var_pred)),
  all(var_pred > -1e-8)
)

sd_pred <- sqrt(pmax(var_pred, 0))

# 95% prediction intervals
lower <- mu_pred + qnorm(0.025) * sd_pred
upper <- mu_pred + qnorm(0.975) * sd_pred

# Stop total runtime clock
runtime_sec <- proc.time()[["elapsed"]] - start_time
runtime_min <- runtime_sec / 60
```

The prediction metrics are also introduced:
```r
# Prediction diagnostics metrics

prediction_metrics <- function(
    y_true, mu, sd,
    alpha = 0.05
) {
  
  stopifnot(
    length(y_true) == length(mu),
    length(mu) == length(sd),
    all(is.finite(y_true)),
    all(is.finite(mu)),
    all(is.finite(sd)),
    all(sd > 0)
  )
  
  error <- y_true - mu
  
  # Point prediction diagnostics
  mae <- mean(abs(error))
  mse <- mean(error^2)
  rmse <- sqrt(mse)
  
  # Gaussian CRPS
  z <- error / sd
  
  crps_i <- sd * (
    z * (2 * pnorm(z) - 1) +
      2 * dnorm(z) -
      1 / sqrt(pi)
  )
  
  crps <- mean(crps_i)
  
  # Prediction intervals
  lower <- mu + qnorm(alpha / 2) * sd
  upper <- mu + qnorm(1 - alpha / 2) * sd
  
  # Interval score
  int_i <- (upper - lower) +
    (2 / alpha) * (lower - y_true) *
    (y_true < lower) +
    (2 / alpha) * (y_true - upper) *
    (y_true > upper)
  
  int <- mean(int_i)
  
  # Coverage
  cvg <- mean(
    y_true >= lower & y_true <= upper
  )
  
  data.frame(
    MAE = mae,
    MSE = mse,
    RMSE = rmse,
    CRPS = crps,
    INT = int,
    CVG = cvg
  )
}


#Apply those functions to the dataset

metrics <- prediction_metrics(
  y_true = Y_test,
  mu = mu_pred,
  sd = sd_pred
)

print(metrics)

runtime_min
```

