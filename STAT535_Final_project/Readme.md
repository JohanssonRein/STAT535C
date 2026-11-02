# Introduction

The purpose of this project is to build up a model that predicts the housing price in rental market. In this particular project, we are interested in the rental market in the province of Alberta, partially because it has the most number of observations ($13777$ observations).

# Methods

We will be working with spatial data, which means the observations are not necessarily independent. Think about the rainfall data: The rainfall collected at UBC and downtown Vancouver may not be independent because they are geographically close to each other. However, we may assume the rainfall collected at UBC and at UofT have very small correlations. 

To investigate the rental price price $y_i$, it is usually common to model the log-observed-price $\log(y_i)$. Suppose we have observations $(\vec x_1(s),y_1),\cdots,(\vec x_n(s),y_n)$, where $\vec x_i(s_i)$ is one observation with its spatial location vector $\vec s_i$. The model we use is given by

$$
\log(y_i) = \vec x_i(\vec s_i)^\top\vec\beta + \omega(\vec s_i) + \gamma(\vec s_i),
$$

where

* The first term $\vec x_i(\vec s_i)^\top\vec\beta$ is the fixed effect in our model, which is a linear regression with covariates independent of the location. For example, we may consider the number of bedrooms, the number of bathrooms, the apartment type, etc.
* The second term $\omega(\vec s_i)$ is a Gaussian process with mean zero, i.e $\omega(\vec s_i) \sim N(\vec 0, \vec\Sigma)$. Its goal is to capture the spatial dependency which cannot be explained by the linear predictor term. The matrix $\vec\Sigma$ is an $n\times n$ variance-covariance matrix, with its entries defined as the spatial dependency between any two observations. It has $2$ parameters: $\phi, \sigma^2$.
* The third term $\epsilon(\vec s_i)$ is called follows a normal distribution $N(0,\tau^2)$, and is independent of both the linear predictor and the Gaussian process. The parameter $\tau^2$ is called the nugget, and we may think of it as the i.i.d error in our model.

# Models

Once we have all the set up, the observed-log-price follows the following distribution:

$$
\vec Y(\vec s) | \vec\theta \sim MVN \Big( \vec X(\vec s) \vec\beta , \vec\Sigma(\sigma^2,\tau^2,\phi)\Big),
$$

where $MVN$ denotes multivariate normal. So based on the joint likelihood, we have different approaches:

* Bayesian approach: We impose a prior on each parameter: $\vec\theta = ( \vec\beta, \sigma^2,\phi,\tau^2)^\top$, and the posterior is given by

$$
p(\vec\theta|\vec y) \propto p(\vec y|\vec\theta) \pi(\vec\beta)\pi(\sigma^2)\pi(\phi)\pi(\tau^2),
$$

and we can proceed with MCMC methods. My older version invokes [HMC](https://arxiv.org/pdf/1701.02434) (Hamiltonian Monte Carlo), and we can try some other methods. For example [INLA](https://arxiv.org/abs/1604.00860)

* Optimization approach: We can impose optimization methods directly on the joint distribution. There are many common methods: Like covariance tapering, NNGP, HSGP. [This paper](https://arxiv.org/abs/1710.05013) includes many interesting methods which we can use.







