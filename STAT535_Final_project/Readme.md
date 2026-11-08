# Introduction

The purpose of this project is to build up a model that predicts the housing price in the rental market. In this particular project, we are interested in the rental market in the province of Alberta, partially because it has the most number of observations. The dataset we will be using is available [here](rentfaster.csv). The following figure shows the average rental price in each municipality of Alberta.


For this project we may want to focus on the city of Calgary and Edmonton only, since $95\%$ of our observations are concentrated within that area and many counties do not have enough observations. Also, we may train our model based on Calgary data and predict the rental price in Edmonton (if their variograms are similar).


This rental data is the one I have in mind at this moment, if we have some other interesting spatial dataset, we can definitely work on them. However I would prefer we use housing/rental data (not necessary in Alberta, big cities like New York, London, will also work), since modelling other types require a re-evaluation of the model and hence contribute to more workload. [This book](https://www.taylorfrancis.com/books/mono/10.1201/9781003401728/hierarchical-modeling-analysis-spatial-data-sudipto-banerjee-alan-gelfand-bradley-carlin) provides detailed modelling of different types of spatial data, you may take a look if you are interested in them (for example, forest fire).

# Model

We will be working with spatial data, which means the observations are not necessarily independent. Think about the rainfall data: The rainfall collected at UBC and downtown Vancouver may not be independent because they are geographically close to each other. However, we may assume the rainfall collected at UBC and at UofT have very small correlations. 

To investigate the rental price $y_i$, it is usually common to model the log-observed-price $\log(y_i)$. Suppose we have observations $(\vec x_1(s_1),y_1),\cdots,(\vec x_n(s_n),y_n)$, where $\vec x_i(s_i)$ is the $i$ th observation with its spatial location vector $\vec s_i$. The model we use is given by

$$
\log(y_i) = \vec x_i(\vec s_i)^\top\vec\beta + \vec\omega(\vec s_i) + \epsilon(\vec s_i),
$$

or

$$
\log(\vec Y_i)=\vec X(\vec s)\vec\beta + \vec\omega(\vec s)+\vec \epsilon(\vec s)
$$

where the above three terms are defined as follow:

### Fixed Effect

The first term $\vec x_i(\vec s_i)^\top\vec\beta$ is the **fixed effect** in our model, which is a linear regression term with covariates independent from its spatial location. For example, those covariates may include the number of bedrooms, the number of bathrooms, the apartment type, etc. In our dataset, we have $10$ spatial-independent covariates in total. 


### Spatial Gaussian Processes (SGP)
The second term $\omega(\vec s_i)$ is a **Gaussian process** with mean zero, i.e $\omega(\vec s_i) \sim N(\vec 0, \vec\Sigma)$. Its goal is to capture the spatial dependency which cannot be explained by the linear predictor term. The matrix $\vec\Sigma$ is an $n\times n$ variance-covariance matrix, with its entries defined as the spatial dependency between any two observations. The covariance function of $\omega(\vec s)$ is monotonically decreasing, modelling the idea that the further the distance, the less the spatial location. We will assume the Gaussian process is *isotropic*, i.e, the spatial correlation between two points $\vec s, \vec s'$ depends only on their euclidean distance $d = \Vert \vec s - \vec s' \Vert$, also $E[\vec\omega(\vec s)]=E[\vec\omega(\vec s+\vec h)]$ where $\vec h = \vec s - \vec s'$. With covariance function defined, the **variogram** (denote as $\gamma(\vec h)$ as a function of spatial distance) is defined as

$$
\gamma(d) = \frac{1}{2}Var(\vec\omega(\vec s+\vec h) - \vec w(\vec s)).
$$

Theoretically, the variogram increases within a certain range; Behaviorally, we would expect similar values of $E[\vec\omega(\vec s)]$ and $E[\vec\omega(\vec s+\vec h)]$ at short distance. When $\Vert \vec h \Vert$ gets large the similarity decreases resulting in a larger variogram. When $\Vert \vec h \Vert$ is large enough so that the spatial dependency is allmost negligible, the variogram will stay stable at a certain threshold. A variogram is modeled by $3$ parameters: Range $\phi$, sill $\sigma^2$ and nugget $\tau^2$. A common choice of the variogram is the Matérn variogram:

$$
\gamma(d) = \begin{cases}  \sigma^2 \left[ 1- \frac{(2\sqrt{\nu} d \phi)^{\nu}}{2^{\nu-1} \Gamma(\nu)} K_{\nu}(2\sqrt{\nu} d \phi) \right] +\tau^2& \text{if $d>0$} \\ 0 & \text{if $d=0$}\end{cases},
$$

where $d=\Vert h \Vert$, and $K_\nu$ is the modified Bessel function of order $\nu$, and $\nu$ is the smoothing parameter of our choice. The empirical variogram is given by

$$
\hat{\gamma}(d) = \frac{1}{2N(\vec d)} \sum_{(\vec s_i,\vec s_j)\in N(d)} | y(\vec s_i)-y(\vec s_j)|^2,
$$

where $N(d)$ is the set of pairs such that $\Vert\vec s_i - \vec s_j \Vert=d$. 

Apart from variogram, we also define the **covariogram** (denote as $C(\vec h)$ ) as the covariance between two spatial sites. In fact

$$
C(\vec h) = \lim_{\Vert \vec u \Vert \to \infty} \gamma(\vec u)-\gamma(\vec h),
$$

and using Matérn model, the covariogram becomes

$$
C(d) = \begin{cases} 
	\frac{\sigma^2}{2^{\nu-1}\Gamma(\nu)}(\phi d)^\nu K_\nu(\phi d) & \text{if $d>0$} \\
 \sigma^2+\tau^2 & \text{if $d=0$} \quad 
\end{cases}.
$$

## Nugget

We have already introduced nugget before, but in general nugget can be viewed as the pure error independent from linear effect and spatial Gaussian process. We have $\vec\epsilon(\vec s) \sim N(0,\tau^2)$.


# Methods

Once we have all the set up, the observed-log-price follows the following distribution:

$$
\vec Y(\vec s) | \vec\theta \sim MVN \Big( \vec X(\vec s) \vec\beta , \vec\Sigma(\sigma^2,\tau^2,\phi)\Big),
$$

where $MVN$ denotes multivariate normal, and the entries of the variance-covariance matrix is given by the covariogram of our choice:

$$
[\vec\Sigma(\sigma^2,\tau^2,\phi)]_{ij} = \text{Cov}(r(\vec s_i),r(\vec s_j)) = \begin{cases} \sigma^2 \left(1+\frac{\sqrt{3}d}{\phi}\right) \exp\left(-\frac{\sqrt{3}d}{\phi} \right) & \text{if $d>0$}\\ \tau^2+\sigma^2 & \text{if $d=0$} \end{cases},
$$

where $d=\Vert h \Vert$.So based on the joint likelihood, we have different approaches:

* Bayesian approach: We impose a prior on each parameter: $\vec\theta = ( \vec\beta, \sigma^2,\phi,\tau^2)^\top$, and the posterior is given by

$$
p(\vec\theta|\vec y) \propto p(\vec y|\vec\theta) \pi(\vec\beta)\pi(\sigma^2)\pi(\phi)\pi(\tau^2),
$$

and we can proceed with MCMC methods. My older version invokes [HMC](https://arxiv.org/pdf/1701.02434) (Hamiltonian Monte Carlo), and we can try some other methods. For example: [SPDE-INLA](https://arxiv.org/abs/1604.00860), which uses stochastic partial differential equations (SPDEs); [NNGP](https://arxiv.org/abs/1406.7343) (Nearest Neighbor Gaussian Process); [HSGP](https://arxiv.org/pdf/2004.11408) (Hilbert Space Gaussian Process), which approximate the covariance matrix by eigenfunctions of the negative Laplacian ($-\Delta$) [This link](https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Boundary-Conditions.html?utm_source=chatgpt.com) has the HSGP implementation in python; Variational Inference (suggested by Charles), and many other methods. [This paper](https://arxiv.org/abs/1710.05013) has many other interesting methods.

* Frequentist approach: We can impose optimization methods directly on the joint distribution. There are many common methods: Covariance Tapering (which takes advantage of the sparsity structure of the matrix); Fixed rank kriging, and so on.
[This paper (the same one as above)](https://arxiv.org/abs/1710.05013) includes many interesting methods which we can use, as well.

# Questions Worth Thinking

* If we have time, we can first simulate some spatil data and fit those models on simulated data first, so we have two comparisons, and we can see if those methods work well in both theory and practice.

* As I mentioned before, we have $10$ location-independent covariates. How should we select the best variables? We may want to implement variable selection methods (such as Bayesian Lasso) and run some diagnostics (such as AIC, BIC). 

* In our data, we have multiple observations with identical spatial locations, where they are likely from the same apartment building. To address this issue, one approach is to average those responses values and treat them as one observation. However, we can do this only if the residuals of the linear model we fit are small and have no common pattern among those observations. The other approach is to add a small random noise to those locations so we end up with more data plus unique spatial site. But according to EDA this severely impact the empirical variogram.

*As soon as we get satisfactory linear and spatal model diagnostics, it all goes down to the implementation of algorithms.*

* I do need to mention that, the core part of the project should be the implementation/ comparison/ prediction of those algorithms that get the estimates of our parameters. The theory behind spatial modelling is very rich and I can't fully grasp it despite already having spent $1$ semester on that topic as part of my undergrad project.

# Further Reading

To get a more comprehensive idea of spatial modeling, please see [my report](https://www.math.mcgill.ca/gsams/drp/papers/papers2026/Report8.pdf) or refer to the [textbook](https://www.taylorfrancis.com/books/mono/10.1201/9781003401728/hierarchical-modeling-analysis-spatial-data-sudipto-banerjee-alan-gelfand-bradley-carlin).







