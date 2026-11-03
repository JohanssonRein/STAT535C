# Introduction

The purpose of this project is to build up a model that predicts the housing price in the rental market. In this particular project, we are interested in the rental market in the province of Alberta, partially because it has the most number of observations. The dataset we will be using is available [here](rentfaster.csv). The following figure shows the average rental price in each municipality of Alberta.

<figure style="width: 700px; max-width: 100%; margin: 1.5em auto; text-align: center;">
  <img 
    src="figures/Average rental price in Alberta.png" 
    alt="" 
    style="width: 700px; height: 650px; object-fit: cover; border-radius: 50%;"
  >
  <figcaption style="font-size: 0.95em; margin-top: 0.5em;">
    <em></em>
  </figcaption>
</figure>

From the plot we can actually see that Calgary and Edmonton metropolis take up most of our observations, where many municipalities has no or few observations, and that could be an potential issue, where a few extreme observations would greatly impact the average rental price. Hence for this project we may want to focus on the city of Calgary and Edmonton only. Also, we may train our model based on Calgary data and predict the rental price in Edmonton.

# Model

We will be working with spatial data, which means the observations are not necessarily independent. Think about the rainfall data: The rainfall collected at UBC and downtown Vancouver may not be independent because they are geographically close to each other. However, we may assume the rainfall collected at UBC and at UofT have very small correlations. 

To investigate the rental price $y_i$, it is usually common to model the log-observed-price $\log(y_i)$. Suppose we have observations $(\vec x_1(s_1),y_1),\cdots,(\vec x_n(s_n),y_n)$, where $\vec x_i(s_i)$ is the $i$th observation with its spatial location vector $\vec s_i$. The model we use is given by

$$
\log(y_i) = \vec x_i(\vec s_i)^\top\vec\beta + \vec\omega(\vec s_i) + \epsilon(\vec s_i),
$$

or

$$
\log(\vec Y_i)=\vec X(\vec s)\vec\beta + \vec\omega(\vec s)+\vec \epsilon(\vec s)
$$

where the above three terms are defined as follow:

### Fixed Effect

The first term $\vec x_i(\vec s_i)^\top\vec\beta$ is the **fixed effect** in our model, which is a linear regression term with covariates independent from its spatial location. For example, those covariates may include the number of bedrooms, the number of bathrooms, the apartment type, etc.


### Spatial Gaussian Processes (SGP)
The second term $\omega(\vec s_i)$ is a **Gaussian process** with mean zero, i.e $\omega(\vec s_i) \sim N(\vec 0, \vec\Sigma)$. Its goal is to capture the spatial dependency which cannot be explained by the linear predictor term. The matrix $\vec\Sigma$ is an $n\times n$ variance-covariance matrix, with its entries defined as the spatial dependency between any two observations. The covariance function of $\omega(\vec s)$ is monotonically decreasing, modelling the idea that the further the distance, the less the spatial location. It has many parameters:  $d=\Vert\vec s-\vec s'\Vert$ define the spatial distance; $\phi$ is the spatial decay parameter, called **range**,  which describes the spatial dependency strength; and finally $\sigma^2$ is called **sill**, which measures how much the variogram (which is, $\text{Var}(\omega(\vec s+\vec h)-\omega(\vec s))$ for a spatial site $\vec s$ and distance vector $\vec h$) has increased after reaching stable.
The popular choice of the covariance function is the Matérn model: For two spatial locations $\vec s,\vec s'$, 

$$
\text{Cov}(w(\vec s),w(\vec s')) = \begin{cases}
\frac{\sigma^2}{2^{\nu-1}\Gamma(\nu)} (\phi d)^\nu K_\nu(\phi d) & \text{if $d>0$} \\
\sigma^2 & \text{if $d=0$}
\end{cases}.
$$

where $K_\nu$, $\nu$ are pre-chosen. 

In practice, we will use the empirical variogram given by

$$
\hat{\gamma}(d) = \frac{1}{2N(d)} \sum_{(\vec s_i,\vec s_j) \in N(d)} | Y(\vec s_i) - Y(\vec s_j)|^2
$$

where $N(d)$ is the set of spatial observations such that $\Vert \vec s_i - \vec s_j \Vert = d$. We often plot the empirical variogram first, and then choose proper $K_\nu$, $\nu$ mentioned before as a parametric estimate.

### Nugget
The third term $\epsilon(\vec s_i)$ follows a normal distribution $N(0,\tau^2)$, and is independent of both the linear predictor and the Gaussian process. The parameter $\tau^2$ is called the **nugget**, and we may think of it as the i.i.d error in our model.



# Methods

Once we have all the set up, the observed-log-price follows the following distribution:

$$
\vec Y(\vec s) | \vec\theta \sim MVN \Big( \vec X(\vec s) \vec\beta , \vec\Sigma(\sigma^2,\tau^2,\phi)\Big),
$$

where $MVN$ denotes multivariate normal. So based on the joint likelihood, we have different approaches:

* Bayesian approach: We impose a prior on each parameter: $\vec\theta = ( \vec\beta, \sigma^2,\phi,\tau^2)^\top$, and the posterior is given by

$$
p(\vec\theta|\vec y) \propto p(\vec y|\vec\theta) \pi(\vec\beta)\pi(\sigma^2)\pi(\phi)\pi(\tau^2),
$$

and we can proceed with MCMC methods. My older version invokes [HMC](https://arxiv.org/pdf/1701.02434) (Hamiltonian Monte Carlo), and we can try some other methods. For example: [SPDE-INLA](https://arxiv.org/abs/1604.00860), which uses stochastic partial differential equations (SPDEs); [NNGP](https://arxiv.org/abs/1406.7343) (Nearest Neighbor Gaussian Process); [HSGP](https://arxiv.org/pdf/2004.11408) (Hilbert Space Gaussian Process), which approximate the covariance matrix by eigenfunctions of the negative Laplacian ($-\Delta$), and many other methods. [This paper](https://arxiv.org/abs/1710.05013) has many other interesting methods.

* Frequentist approach: We can impose optimization methods directly on the joint distribution. There are many common methods: Covariance Tapering (which takes advantage of the sparsity structure of the matrix); Fixed rank kriging, and so on.
[This paper (the same one as above)](https://arxiv.org/abs/1710.05013) includes many interesting methods which we can use, as well.

# Further Reading

To get a more comprehensive idea of spatial modeling, please see [my report](https://www.math.mcgill.ca/gsams/drp/papers/papers2026/Report8.pdf) or refer to the [textbook](https://www.taylorfrancis.com/books/mono/10.1201/9781003401728/hierarchical-modeling-analysis-spatial-data-sudipto-banerjee-alan-gelfand-bradley-carlin).







