library(geoR)
library(MASS)
library(rjags)
library(coda)
library(fields)
library(spdep)
library(tidyverse)

##Part I: Loading data and data cleaning

#Read the data， the data is also available on my Github: https://github.com/JohanssonRein/STAT535C/blob/main/rentfaster.csv
rental <- read_csv("rentfaster.csv", na = c("", "NA", "none"))

#By running nrow(rental) and ncol(rental),
#the data has 25771 rows (observations) and 18 columns (covariates)

#We restrict to observations in Calgary, Alberta. This creates 9836 observations
rental_AB <- rental %>%
  filter(province == "Alberta", city == "Calgary")

# Data cleaning processes:

rent_AB <- rental_AB %>%
  # This makes sure to assign 0 bedrooms to a studio apartment, as well as make the 
  # number of bedrooms, as well as square feet numeric, by using parse_number function 
  mutate(
    
    beds = if_else(
      grepl("Studio|none", beds, ignore.case = TRUE),
      "0",
      as.character(beds)
    ),
    beds = parse_number(beds),
    
    baths = case_match(baths, NA ~ 0, .default = baths),
    
    #Here we decode whether pets are allowed as a binary variable
    i_cats_allowed = as.integer(cats),
    i_dogs_allowed = as.integer(dogs),
    i_pets_allowed = as.integer(cats | dogs),
    
    sq_feet = as.numeric(
      stringr::str_extract(
        gsub(",","", as.character(sq_feet)),
        "[0-9]+\\.?[0-9]*"
      )
    ),
    
    #We divide the property into the following main types:
    property_type = case_when(
      type %in% c("Apartment","Condo Unit") ~ "Apartment",
      type %in% c("House","Townhouse") ~ "House", 
      type == "Basement" ~ "Basement",
      type == "Room For Rent" ~ "Room For Rent",
      TRUE ~ "Others"
    ),
    
    #Encoding property types as indicators for modelling.
    i_Apartment = as.integer(property_type == "Apartment"),
    i_House = as.integer(property_type == "House"),
    i_Basement = as.integer(property_type == "Basement"),
    i_Room = as.integer(property_type == "Room For Rent"),
    
    property_type = factor(
        property_type,
        levels = c(
          "Apartment",
          "Basement",
          "House",
          "Room For Rent",
          "Others"
        )
    )
    
  ) %>%
  # We would like to remove some extreme observations, or bad data points. 
  # Especially, we restrict our attention to the Calgary-Edmonton area to avoid extremely little observations across remote places, since Calgary-Edmonton area itself takes up to 10510 observations.
  filter(
    !is.na(price),
    price > 200,
    price < 10000,
    beds <= 6,
    !type %in% c("Office Space", "Parking Spot", "Storage"),
    is.finite(latitude),
    is.finite(longitude),
    longitude >= -114.5,
    longitude <= -113.0,
    latitude >= 50.8,
    latitude <= 53.8
  )

#Finally, the cleaned data has 9657 observations (Calgary, AB)

rent_AB <- rent_AB %>%
  filter(!is.na(sq_feet))


rent_AB <- rent_AB %>%

  mutate(
         i_one_bath = as.integer(baths == 1),
         i_one_bed = as.integer(beds == 1),
         i_1b1b = i_one_bed*i_one_bath,
         beds_c3 = pmin(beds, 3),
         beds_c4 = pmin(beds, 4),
         beds_c5 = pmin(beds, 5),
         i_Furnished = as.integer(furnishing == "Furnished"),
         i_Unfurnished = as.integer(furnishing == "Unfurnished"),
         i_beds_ge4 = as.integer(beds >= 4),
         i_beds_ge5 = as.integer(beds >= 5),
         i_baths_gt3 = as.integer(baths > 3),
         i_immed_avail = as.integer(availability_date == "Immediate"),
         i_Furnished_i_Apartment = i_Furnished*i_Apartment,
         i_Furnished_i_House = i_Furnished*i_House,
         sq_feet_i_Apartment = sq_feet*i_Apartment,
         sq_feet_i_House = sq_feet*i_House)


#Since prices are positive, we work with log-price (maybe it is worth to examine using price instead of log-price)
rent_AB <- rent_AB %>% mutate(log_price = log(price))

## Part 2 Regression model

#We first fit a baseline regression model where we do not consider any spatial dependency.

# The covariates we consider are the number of bedrooms, number of bathrooms, is pets allowed (Binary) and the property type (5 main types)

# for later purposes, we randomly select 80% of our data into a training set and the remaining as the testing set.

set.seed(123)

train <- sample(
  seq_len(nrow(rent_AB)),
  size = floor(0.9 * nrow(rent_AB)) # Notice here I use 0.9, because when we first split into training and testing set, we did not specify spatial location. If we do this later on trainig set and testing set separately then the number will be different, an overall trend I found is that testing sample is always significantly larger than trainign sample.
)

rent_AB_train <- rent_AB[train, ] # Training set has 6305 observations
rent_AB_test  <- rent_AB[-train, ] # Testing set has 1577 observations

# Note that many missing values (1171) in ''sq_feet'' so based on our new model the available datasize is much smaller


rent_AB_lm <- lm(log_price ~ sq_feet + beds_c3 + i_pets_allowed + i_Room + i_Furnished +
               i_Furnished_i_Apartment + sq_feet_i_Apartment + sq_feet_i_House, data = rent_AB_train)


#rent_AB_lm <- lm(
#  price ~ beds + baths + i_pets_allowed + property_type,
 # data = rent_AB_train,
 #na.action = na.exclude
#)


summary(rent_AB)
summary(rent_AB_train)
summary(rent_AB_lm)

# Residuals
rent_AB_train <- rent_AB_train %>%
  mutate(
    fitted_log_price = as.numeric(
      predict(rent_AB_lm, newdata = rent_AB_train)
    ),
    resid_lm = log_price - fitted_log_price
  )

# Test-set predictions from the baseline regression
rent_AB_test <- rent_AB_test %>%
  mutate(
    predicted_log_price = as.numeric(
      predict(rent_AB_lm, newdata = rent_AB_test)
    ),
    predicted_residual = log_price - predicted_log_price
  )



fitted_log_price <- predict(rent_AB_lm, newdata = rent_AB)
residuals <- rent_AB$log_price - fitted_log_price

# Plotting fitted log-price versus actual log-price
plot(
  fitted_log_price, rent_AB$log_price,
  xlab = "Fitted log-price",
  ylab = "Actual log-price",
  main = "Fitted v.s actual log price on training set",
  pch = 16,
  cex = 0.6
)
abline(0, 1, col = "red", lwd = 2)

# Plotting fitted log-price versus the residuals on the training set
plot(
  fitted_log_price, residuals,
  xlab = "Fitted log-price",
  ylab = "Residuals",
  main = "",
  pch = 16,
  cex = 0.6
)
abline(h=0, col = "red", lwd = 2)

# Run our model for the testing set and check residuals
predicted_log_price <- predict(rent_AB_lm, newdata = rent_AB_test)

predicted_residual <- rent_AB_test$log_price - predicted_log_price

plot(
  predicted_log_price, rent_AB_test$log_price,
  xlab = "Predicted log price",
  ylab = "Actual log-price",
  main = "",
  pch = 16,
  cex = 0.6
)
abline(0, 1, col = "red", lwd=2)


plot(
  predicted_log_price, predicted_residual,
  xlab = "Precidted log-price",
  ylab = "Residuals on testing set",
  main = "",
  pch = 16,
  cex = 0.6
)
abline(h=0, col = "red", lwd = 2)


# We want to see if our linear model works well when spatial dependency is removed. 



# We start by grouping the observations with the same spatial location (they are likely from the same apartment building)
rent_AB_train <- rent_AB_train %>%
  group_by(longitude, latitude) %>%
  summarise(
    n_site = n(),
    mean_log_price = mean(log_price),
    mean_fitted = mean(fitted_log_price),
    .groups = "drop"
  )
# 2569 training data


rent_AB_test <- rent_AB_test %>%
  group_by(longitude, latitude) %>%
  summarise(
    n_site = n(),
    mean_log_price = mean(log_price),
    mean_fitted = mean(fitted_log_price),
    .groups = "drop"
  )

#  Testing data (so see it heavily depend on how we sample)



# By doing so we have 2636 unique data

# A big question mark here: What do we do with those 1171 missing data?

rent_AB_train <- rent_AB_train %>%
  mutate(
    spatial_response = mean_log_price - mean_fitted
  )

# Convert spatial coordinates to kilometres for model simplicity


library(sf)

rent_uni_sf <- sf::st_as_sf(
  rent_AB_train,
  coords = c("longitude", "latitude"),
  crs = 4326,
  remove = FALSE
)

# Project to NAD83 / Statistics Canada Lambert
rent_uni_sf <- sf::st_transform(
  rent_uni_sf,
  crs = 3347
)

# Projected coordinates are in metres
coords_km <- sf::st_coordinates(rent_uni_sf) / 1000

rent_AB_train$x_km <- coords_km[, 1]
rent_AB_train$y_km <- coords_km[, 2]

# Check coordinate validity
#stopifnot(
  #all(is.finite(coords_km)),
  #!anyDuplicated(as.data.frame(coords_km))
#)

#summary(rent_AB_uni[, c("x_km", "y_km")])


#Create GeoR data type
#geo_cal <- geoR::as.geodata(
#  rent_AB_train,
#  coords.col = c("x_km", "y_km"),
#  data.col = "mean_log_price"
#)

#summary(geo_cal)

#vario_cal <- variog(
#  geo_cal,
#  uvec = seq(0,20, by = 0.5),
#  estimator.type = "classical"
#)

#plot(vario_cal)


# Coordinate matrices in kilometres
coords_train <- as.matrix(
  rent_AB_train[, c("x_km", "y_km")]
)

#coords_test <- as.matrix(
#  rent_AB_test[, c("x_km", "y_km")]
#)


# Setting up NNGP
#NNGP

library(spNNGP)

# Spatial response: site-averaged regression residual
formula_nngp <- spatial_response ~ 1

# Empirical variance of the mean residuals
v <- var(rent_AB_train$spatial_response)

# Initial regression coefficient
beta_initial <- mean(rent_AB_train$spatial_response)

# Prior distributions
priors_new <- list(
  sigma.sq.IG = c(2, v / 2),
  tau.sq.IG = c(2, v / 2),
  phi.Unif = c(0.01, 10),
  nu.Unif = c(0.5,1.5)
)

# Metropolis-Hastings tuning
tuning_response <- list(
  sigma.sq = 0.05,
  tau.sq = 0.05,
  phi = 0.05,
  nu = 0
)

#Different starting values for four chains
sigma2_initial <- v * c(0.2, 0.4, 0.6, 0.8)

tau2_initial <- v * c(0.8, 0.6, 0.4, 0.2)

phi_initial <- c(0.2, 0.5, 1, 2)





n_chains <- 4
n_iter <- 5000

fits_nngp <- vector("list", n_chains)
runtime_nngp <- numeric(n_chains)

total_start <- proc.time()[["elapsed"]]

for (i in seq_len(n_chains)) {
  
  set.seed(123 + i)
  
  starting_i <- list(
    beta = beta_initial,
    sigma.sq = sigma2_initial[i],
    tau.sq = tau2_initial[i],
    phi = phi_initial[i],
    nu = 1
  )
  
  start_time <- proc.time()[["elapsed"]]
  
  result <- spNNGP::spNNGP(
    formula = formula_nngp,
    data = rent_AB_train,
    coords = coords_train,
    method = "response",
    family = "gaussian",
    n.neighbors = 25,
    starting = starting_i,
    tuning = tuning_response,
    priors = priors_new,
    cov.model = "exponential",
    n.samples = n_iter,
    n.omp.threads = 4,
    fit.rep = FALSE,
    verbose = TRUE,
    n.report = 500
  )
  
  runtime_nngp[i] <-
    proc.time()[["elapsed"]] - start_time
  
  stopifnot(
    all(is.finite(result$p.theta.samples))
  )
  
  fits_nngp[[i]] <- result
  
  saveRDS(
    result,
    paste0("NNGP_simple_chain_", i, ".rds")
  )
  
  saveRDS(
    runtime_nngp,
    "NNGP_simple_runtime.rds"
  )
}

total_runtime_nngp <-
  proc.time()[["elapsed"]] - total_start

cat(
  "Total runtime:",
  total_runtime_nngp / 60,
  "minutes\n"
)


#Examine posterior density
burn_in <- 2000


# MCMC Diagnostics
library(tidyverse)
library(coda)

posterior_nngp <- bind_rows(
  lapply(seq_along(fits_nngp), function(i) {
    
    theta <- as.matrix(
      fits_nngp[[i]]$p.theta.samples
    )
    
    # Keep only the three estimated parameters
    theta <- theta[
      (burn_in + 1):nrow(theta),
      c("sigma.sq", "tau.sq", "phi"),
      drop = FALSE
    ]
    
    data.frame(
      iteration = seq.int(burn_in + 1, burn_in + nrow(theta)),
      theta,
      chain = factor(i),
      check.names = FALSE
    )
  })
) %>%
  pivot_longer(
    cols = c("sigma.sq", "tau.sq", "phi"),
    names_to = "parameter",
    values_to = "value"
  )

# Trace plots
ggplot(
  posterior_nngp,
  aes(
    x = iteration,
    y = value,
    color = chain
  )
) +
  geom_line(
    linewidth = 0.3,
    alpha = 0.7
  ) +
  facet_wrap(
    ~ parameter,
    scales = "free_y",
    ncol = 1
  ) +
  labs(
    title = "NNGP MCMC Trace Plots",
    x = "Iteration",
    y = "Parameter value",
    color = "Chain"
  ) +
  theme_minimal(base_size = 13)



# Density plots
ggplot(
  posterior_nngp,
  aes(
    x = value,
    color = chain
  )
) +
  geom_density(
    linewidth = 1
  ) +
  facet_wrap(
    ~ parameter,
    scales = "free",
    ncol = 3
  ) +
  labs(
    title = "NNGP Posterior Densities",
    x = "Parameter value",
    y = "Density",
    color = "Chain"
  ) +
  theme_minimal(base_size = 13)


# convergence diagnostics
theta_chains <- coda::mcmc.list(
  lapply(fits_nngp, function(fit) {
    
    theta <- as.matrix(fit$p.theta.samples)
    
    theta <- theta[
      (burn_in + 1):nrow(theta),
      c("sigma.sq", "tau.sq", "phi"),
      drop = FALSE
    ]
    
    coda::mcmc(theta)
  })
)

# Gelman-Rubin diagnostic
gelman.diag(theta_chains)

# Effective sample sizes
effectiveSize(theta_chains)

# Posterior summaries
summary(theta_chains)








