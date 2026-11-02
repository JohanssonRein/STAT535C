# Introduction

The purpose of this project is to build up a model that predicts the housing price in rental market. In this particular project, we are interested in the rental market in the province of Alberta, partially because it has the most number of observations ($13777$ observations).

# Methods

We will be working with spatial data, which means the observations are not necessarily independent. Think about the rainfall data: The rainfall collected at UBC and downtown Vancouver may not be independent because they are geographically close to each other. However, we may assume the rainfall collected at UBC and at UofT have very small correlations. 

To investigate the rental price price $y_i$, it is usually common to model the log-observed-price $\log(y_i)$. Suppose we have observations $(\vec x_1,y_1),\cdots,(\vec x_n,y_n)$, then the model we use is given by

$$
\log(y_i) = \vec x_i^\top\vec\beta + W(s) + \gamma(s).
$$
