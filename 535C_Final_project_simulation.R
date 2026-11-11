library(readr)
library(dplyr)
library(ggplot2)
library(geoR)
library(MASS)
library(rjags)
library(coda)
library(fields)
library(spdep)
library(geodata)
library(stringr)
library(viridis)
library(scales)
library(Matrix)
library(dbscan)
library(patchwork)

#simulating geodata
set.seed(123)

L <- 4

grid <- list(
  x = seq(0, L, length.out = 100),
  y = seq(0, L, length.out = 100)
)

# covariance matrix following the Matern covariogram
cov_obj <- fields::circulantEmbeddingSetup(
  grid = grid,
  M = c(512,512),
  cov.args = list(
    Covariance = "Matern",
    aRange = 0.2,
    smoothness = 1
  )
)

# Simulate the spatial field
w <- fields::circulantEmbedding(cov_obj)

# Add independent nugget noise
Y <- as.vector(w) +
  rnorm(length(w), mean = 0, sd = sqrt(0.1))

# Create a data frame
sim_data <- expand.grid(
  x = grid$x,
  y = grid$y
)

sim_data$response <- Y

# Adding fixed effect

sim_data$X1 <- rnorm(10000, mean = 0, sd = 1)

sim_data$X2 <- rbinom(10000, size = 1, prob = 0.7)

sim_data$X3 <- runif(10000, min = -1, max = 1)

beta_true <- c(2, 0.5, -1.2, 0.8)

X <- model.matrix(
  ~ X1 + X2 + X3, data = sim_data
)

sim_data$Y <- as.vector(
  X %*% beta_true + sim_data$response
)


# Covert our simulated data to geoR data
geo_sim <- geoR::as.geodata(
  sim_data,
  coords.col = c("x", "y"),
  data.col = "Y"
)

# a summary of our data
summary(geo_sim)

# Plotting our data

sim_df <- data.frame(
  x = geo_sim$coords[, 1],
  y = geo_sim$coords[, 2],
  value = geo_sim$data
)

ggplot(sim_df, aes(x = x, y = y, color = value)) +
  geom_point(size = 1.5, alpha = 0.85) +
  scale_color_viridis_c(option = "viridis") +
  coord_equal() +
  labs(
    title = "Simulated Matérn Gaussian Random Field",
    subtitle = "",
    x = "X coordinate",
    y = "Y coordinate",
    color = "Simulated value"
  ) +
  theme_minimal(base_size = 13)



#------------Double Tapering------------------

# Setting up training set and testing set

set.seed(123)

num_sim <- nrow(sim_data)

train_idx <- sample(
  seq_len(num_sim),
  size = floor(0.8 * num_sim)
)

sim_train <- sim_data[train_idx, ]
sim_test <- sim_data[-train_idx, ]

#-------------------------------------------------------------------



#Model set up
coords <- as.matrix(sim_train[, c("x", "y")])

Y <- sim_train$Y

X <- model.matrix(
  ~ X1 + X2 + X3,
  data = sim_train
)

n <- length(Y)

stopifnot(
  nrow(coords) == n,
  nrow(X) == n,
  all(is.finite(coords)),
  all(is.finite(Y)),
  all(is.finite(X))
)

# True parameters for beta
beta_true <- c(2, 0.5, -1.2, 0.8)

# True parameters for range phi, sill sigma^2 and nugget tau^2
theta_true <- c(
  sigma2 = 1,
  tau2 = 0.1,
  phi = 0.2
)
#--------------------------------------------------------------------

# The Matern covariogram (with smoothing parameter chosen to be 1)
matern_corr <- function(d, phi, nu) {
  
  z <- d / phi
  
  result <- numeric(length(z))
  
  result[z == 0] <- 1
  
  idx <- z > 0
  
  result[idx] <-
    2^(1 - nu) / gamma(nu) *
    z[idx]^nu *
    besselK(z[idx], nu)
  
  return(result)
}
#-----------------------------------------------------------------

# Taper function
wendland <- function(d, taper_range) {
  
  u <- d / taper_range
  
  result <- numeric(length(u))
  
  idx <- u < 1
  
  result[idx] <- (1 - u[idx])^4 *
    (1 + 4 * u[idx])
  
  return(result)
}
#--------------------------------------------------------------------

# Constructing sparse taper matrix
taper_range <- 0.3

# Find pairs within taper range
nn <- dbscan::frNN(
  coords,
  eps = taper_range,
  search = "kdtree"
)

i_all <- rep(seq_len(n), lengths(nn$id))
j_all <- unlist(nn$id, use.names = FALSE)
d_all <- unlist(nn$dist, use.names = FALSE)

# Keep each unordered pair only once
keep <- j_all > i_all

i <- i_all[keep]
j <- j_all[keep]
d <- d_all[keep]

# Evaluate Wendland taper at neighbor distances
taper_values <- wendland(d, taper_range)

# Construct sparse symmetric matrix
T_mat <- Matrix::sparseMatrix(
  i = c(seq_len(n), i, j),
  j = c(seq_len(n), j, i),
  x = c(rep(1, n), taper_values, taper_values),
  dims = c(n, n)
)

T_mat <- Matrix::forceSymmetric(T_mat)

cat("Observations:", n, "\n")
cat("Nonzero taper entries:", Matrix::nnzero(T_mat), "\n")



build_R_taper <- function(phi, nu = 1) {
  
  corr <- matern_corr(d, phi, nu) *
    taper_values
  
  R <- Matrix::sparseMatrix(
    i = c(seq_len(n), i, j),
    j = c(seq_len(n), j, i),
    x = c(rep(1, n), corr, corr),
    dims = c(n, n)
  )
  
  return(Matrix::forceSymmetric(R))
}
#------------------------------------------------------

# Estimating the linear regression coefficients using OLS
beta_ols <- as.vector(qr.solve(X, Y))

#print(beta_ols): intercept 2.0226515 X1 0.4877844 X2 -1.2006679 X3 0.8150569

names(beta_ols) <- colnames(X)

# Residuals used in double-taper criterion
r <- Y - as.vector(X %*% beta_ols)
#------------------------------------------------------

# Hutchinson random probes for inverse covariance matrix estimation
set.seed(123)

n_probes <- 64

Z <- matrix(
  sample(
    c(-1, 1),
    n * n_probes,
    replace = TRUE
  ),
  nrow = n,
  ncol = n_probes
)
#----------------------------------------------------

#Double tapering objective function
two_taper_objective <- function(
    log_par,
    r,
    T_mat,
    Z,
    nu = 1
) {
  
  # Parameters on original scale
  sigma2 <- exp(log_par[1])
  tau2   <- exp(log_par[2])
  phi    <- exp(log_par[3])
  
  n <- length(r)
  K <- ncol(Z)
  
  # Build tapered Matérn correlation
  R_taper <- build_R_taper(phi, nu)
  
  # Tapered covariance matrix
  V <- sigma2 * R_taper +
    Matrix::Diagonal(n, tau2)
  
  # Sparse Cholesky factorization
  V_chol <- tryCatch(
    Matrix::Cholesky(
      V,
      LDL = FALSE,
      super = TRUE
    ),
    error = function(e) NULL
  )
  
  if (is.null(V_chol)) {
    return(1e20)
  }
  
  # Log determinant of covariance
  logdet <- as.numeric(
    determinant(
      V_chol,
      logarithm = TRUE,
      sqrt = FALSE
    )$modulus
  )
  
  # Solve V U = Z
  U <- as.matrix(solve(V_chol, Z))
  
  # Compute (r r^T o T) U
  B_U <- r * (T_mat %*% (r * U))
  
  # Hutchinson trace approximation
  trace_est <- sum(Z * B_U) / K
  
  # Normalized two-taper criterion
  obj <- 0.5 * (logdet + trace_est) / n
  
  if (!is.finite(obj)) {
    return(1e20)
  }
  
  return(obj)
}


#-------------------------------

# Estimating the parameters
nu <- 1

start_time <- proc.time()[["elapsed"]]

fit_two_taper <- optim(
  par = log(c(1, 0.1, 0.15)),
  
  fn = two_taper_objective,
  
  r = r,
  T_mat = T_mat,
  Z = Z,
  nu = nu,
  
  method = "L-BFGS-B",
  
  lower = log(c(1e-4, 1e-4, 0.02)),
  upper = log(c(10, 10, 2)),
  
  control = list(
    maxit = 100,
    factr = 1e7
  )
)

optimization_time <- proc.time()[["elapsed"]] -
  start_time

# Extract covariance estimates
theta_hat <- exp(fit_two_taper$par)

names(theta_hat) <- c(
  "sigma2",
  "tau2",
  "phi"
)

#print(theta_hat) : sigma2 1.0311153; tau2 0.1004160 phi 0.2127228

#----------------------------------------------------------------


#Prediction

#Preparing (we can also do it on the entire dataset)
coords_test <- as.matrix(sim_test[, c("x", "y")])

X_test <- model.matrix(
  ~ X1 + X2 + X3,
  data = sim_test
)

Y_test <- sim_test$Y

n_test <- length(Y_test)

# Estimated parameters from training
sigma2_hat <- unname(theta_hat["sigma2"])
tau2_hat   <- unname(theta_hat["tau2"])
phi_hat    <- unname(theta_hat["phi"])



#-------------------------------------------------------------

# Covariance matrix of training obs
R_hat <- build_R_taper(phi_hat, nu)

V_train <- sigma2_hat * R_hat +
  Matrix::Diagonal(n, tau2_hat)

V_chol <- Matrix::Cholesky(
  V_train,
  LDL = FALSE,
  super = TRUE
)


#---------------------------------------------------------------

# Traing-testing cross variance term
nn_test <- dbscan::frNN(
  coords,
  eps = taper_range,
  query = coords_test,
  search = "kdtree"
)

# Indices of training and testing locations
train_id <- unlist(nn_test$id, use.names = FALSE)

test_id <- rep(
  seq_len(n_test),
  lengths(nn_test$id)
)

d_cross <- unlist(nn_test$dist, use.names = FALSE)

# Cross-covariance values
cross_cov <- sigma2_hat *
  matern_corr(d_cross, phi_hat, nu) *
  wendland(d_cross, taper_range)

# Matrix dimensions: n_train x n_test
C_cross <- Matrix::sparseMatrix(
  i = train_id,
  j = test_id,
  x = cross_cov,
  dims = c(n, n_test)
)


#-------------------------------------------------------------------------

# Predict at a new location y
# Regression prediction on testing set
Y_reg_pred <- as.vector(
  X_test %*% beta_ols
)

# Spatial residuals from training observations
r_train <- Y - as.vector(X %*% beta_ols)

# Solve V_train * a = r_train
a <- as.vector(
  solve(V_chol, r_train)
)

# Predict the spatial effect at testing locations
w_pred <- as.vector(
  crossprod(C_cross, a)
)

# FINAL PREDICTION
Y_pred <- Y_reg_pred + w_pred


#-----------------------------------------------------------------------

# Basic disgnostics
prediction_results <- data.frame(
  x = sim_test$x,
  y = sim_test$y,
  Observed = Y_test,
  Predicted = Y_pred,
  Residual = Y_test - Y_pred
)

head(prediction_results)


MSE <- mean((Y_test - Y_pred)^2)
RMSE <- sqrt(MSE)
MAE <- mean(abs(Y_test - Y_pred))

cat("Test MSE:", MSE, "\n")
cat("Test RMSE:", RMSE, "\n")
cat("Test MAE:", MAE, "\n")

#----------------------------------------------------------------------


# Prediction variance
var_pred <- numeric(n_test)

batch_size <- 100

for (start in seq(1, n_test, by = batch_size)) {
  
  idx <- start:min(start + batch_size - 1, n_test)
  
  C_batch <- as.matrix(
    C_cross[, idx, drop = FALSE]
  )
  
  A <- as.matrix(
    solve(V_chol, C_batch)
  )
  
  reduction <- colSums(C_batch * A)
  
  # Predictive variance for a new noisy observation Y(s)
  var_pred[idx] <- sigma2_hat + tau2_hat - reduction
}

# Numerical safety
var_pred <- pmax(var_pred, 0)

sd_pred <- sqrt(var_pred)
#------------------------------------------------------------------------

prediction_results <- data.frame(
  x = sim_test$x,
  y = sim_test$y,
  Observed = Y_test,
  Predicted = Y_pred,
  Residual = Y_test - Y_pred,
  Pred_SD = sd_pred
)

#-----------------------------------------------------------------------


# Plotting
zlim <- range(
  c(prediction_results$Observed,
    prediction_results$Predicted),
  na.rm = TRUE
)

# Symmetric range for residual map
res_lim <- max(abs(prediction_results$Residual), na.rm = TRUE)


# Observed plot
p_obs <- ggplot(
  prediction_results,
  aes(x = x, y = y, color = Observed)
) +
  geom_point(shape = 15, size = 2) +
  coord_equal() +
  scale_color_viridis_c(
    option = "viridis",
    limits = zlim
  ) +
  labs(
    title = "Observed Surface (Test Set)",
    x = "X coordinate",
    y = "Y coordinate",
    color = "Observed Y"
  ) +
  theme_minimal(base_size = 14)

# Predicted plot
p_pred <- ggplot(
  prediction_results,
  aes(x = x, y = y, color = Predicted)
) +
  geom_point(shape = 15, size = 2) +
  coord_equal() +
  scale_color_viridis_c(
    option = "viridis",
    limits = zlim
  ) +
  labs(
    title = "Predicted Surface",
    x = "X coordinate",
    y = "Y coordinate",
    color = "Predicted Y"
  ) +
  theme_minimal(base_size = 14)

p_pred

# Residual map
p_resid <- ggplot(
  prediction_results,
  aes(x = x, y = y, color = Residual)
) +
  geom_point(shape = 15, size = 2) +
  coord_equal() +
  scale_color_gradient2(
    low = "blue",
    mid = "white",
    high = "red",
    midpoint = 0,
    limits = c(-res_lim, res_lim)
  ) +
  labs(
    title = "Residual Map",
    x = "X coordinate",
    y = "Y coordinate",
    color = "Y - Pred"
  ) +
  theme_minimal(base_size = 14)

p_resid

# Predictive SD 

p_sd <- ggplot(
  prediction_results,
  aes(x = x, y = y, color = Pred_SD)
) +
  geom_point(shape = 15, size = 2) +
  coord_equal() +
  scale_color_viridis_c(option = "magma") +
  labs(
    title = "Predictive SD Map",
    x = "X coordinate",
    y = "Y coordinate",
    color = "Pred SD"
  ) +
  theme_minimal(base_size = 14)

p_sd











# Plotting the prediction results.







































