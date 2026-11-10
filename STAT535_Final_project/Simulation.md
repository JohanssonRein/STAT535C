In the Readme file, we briefly went through our goal of the project and the basics of spatial modelling. In this file, we will be implementing some methods to a simulated data. The next task is to implement them to the actual rent data.

# Simulations

We first simulate $10000$ spatial data points. In R, load the following packages:

```r
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
```

The following code will simulate spatial data on a $100\times 100$ grid, where correlations between two grids are modelled by a Matern covariogram function with smoothing parameter $\nu=1$. We set the range ($\phi$) parameter to be $0.2$ and the sill ($\sigma^2$) is by default set to be $1$.

```r
set.seed(123)

grid <- list(
  x = seq(0, 1, length.out = 100),
  y = seq(0, 1, length.out = 100)
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
```

We then add the nugget $(\tau^2)$ into our model. We set the nugget parameter to be $\tau^2=0.1$ and the i.i.d error term follows a normal distribution with mean $0$ and variance $\tau^2$.

```r
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
```

Finally, we will simulate a fixed effect term, we simulate $3$ covariates $X_1, X_2, X_3$, plus an intercept. The covariates follows different distributions, and $\vec\beta$ is the true parameter we set. That is, we assume the fixed effect $m(x)$ follows

$$
m(x) = 2 + 0.5 X_1  - 1.2 X_2 + 0.8 X_3.
$$

```r
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
```

And we finally convert them into GeoR data type:

```r
geo_sim <- geoR::as.geodata(
  sim_data,
  coords.col = c("x", "y"),
  data.col = "Y"
)
```

To get a graphical illustration of our simulated data, the following code will create a plot:

```r
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
  ```

  Whereas the result is given as follow:

<figure style="width: 280px; max-width: 100%; margin: 1.5em auto; text-align: center;">
  <img 
    src="/figures/simulated_data_plot.PNG" 
    alt="" 
    style="width: 400px; height: 400px; object-fit: cover; border-radius: 50%;"
  >
  <figcaption style="font-size: 0.95em; margin-top: 0.5em;">
    <em></em>
  </figcaption>
</figure>


