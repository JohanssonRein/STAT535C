library(readr)
library(dplyr)
library(ggplot2)
library(geoR)
library(MASS)
library(rjags)
library(coda)
library(fields)
library(spdep)


##Part I: Loading data and data cleaning

#Read the data， the data is also available on my Github: https://github.com/JohanssonRein/STAT535C/blob/main/rentfaster.csv
rental <- read_csv("rentfaster.csv")

#By running nrow(rental) and ncol(rental),
#the data has 25771 rows (observations) and 18 columns (covariates)

#We restrict to observations in Alberta. This creates 13777 observations
rental_AB <- rental %>%
  filter(province == "Alberta")

# Data cleaning processes:

rent_AB <- rental_AB %>%
  # This makes sure to assign 0 bedrooms to a studio apartment, as well as make the 
  # number of bedrooms, as well as square feet numeric, by using parse_number function 
  mutate(
    
    beds = case_when(
      grepl("studio", beds, ignore.case = TRUE) ~ 0,
      TRUE ~ parse_number(beds)
    ),
    
    #Here we decode whether pets are allowed as a binary variable
    cats_allowed = as.integer(cats),
    dogs_allowed = as.integer(dogs),
    pets_allowed = as.integer(cats | dogs),
    
    sq_feet = parse_number(sq_feet),
    
    #We divide the property into the following main types:
    property_type = case_when(
      type %in% c("Apartment","Condo Unit") ~ "Apartment",
      type %in% c("House","Townhouse") ~ "House", 
      type == "Basement" ~ "Basement",
      type == "Room For Rent" ~ "Room For Rent",
      TRUE ~ "Others"
    ),
    
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
    !is.na(sq_feet),
    !is.na(longitude),
    !is.na(latitude),
    sq_feet > 250,
    beds <= 6,
    !type %in% c("Office Space", "Parking Spot", "Storage"),
    is.finite(latitude),
    is.finite(longitude),
    longitude >= -114.5,
    longitude <= -113.0,
    latitude >= 50.8,
    latitude <= 53.8
  )

#Finally, the cleaned data has 10510 observations (Calgary-Edmonton area)




#Since prices are positive, we work with log-price
rent_AB <- rent_AB %>% mutate(log_price = log(price))

## Part 2 Regression model

#We first fit a baseline regression model where we do not consider any spatial dependency.

# The covariates we consider are the number of bedrooms, number of bathrooms, is pets allowed (Binary) and the property type (5 main types)

# for later purposes, we randomly select 80% of our data into a training set and the remaining as the testing set.
# Using our 80-20 training/testing split, we have 8408 observations.
set.seed(123)

train <- sample(
  seq_len(nrow(rent_AB)),
  size = floor(0.8 * nrow(rent_AB))
)

rent_AB_train <- rent_AB[train, ]
rent_AB_test  <- rent_AB[-train, ]

rent_AB_lm <- lm(
  log_price ~ beds + baths + pets_allowed + property_type,
  data = rent_AB_train,
  na.action = na.exclude
)

summary(rent_AB)
summary(rent_AB_train)
summary(rent_AB_lm)

fitted_log_price <- predict(rent_AB_lm, newdata = rent_AB_train)
residuals <- rent_AB_train$log_price - fitted_log_price

# Plotting fitted log-price versus actual log-price
plot(
  fitted_log_price, rent_AB_train$log_price,
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


## Part III: Modeling spatial dependency

resid1 <- residuals(rent_AB_lm)

lm_trend <- lm(
  resid1 ~ longitude + latitude +
    I(longitude^2) + I(latitude^2) + I(longitude * latitude),
  data = rent_AB_train 
)

summary(lm_trend)

#Since multiple listings may have the exact spatial coordinate (coming from the same unit),
# we add a random noise to those duplicates so the spatial coordinates are unique.

rent_AB_train$resid1 <- residuals(rent_AB_lm)

rent_spatial <- rent_AB_train %>%
  mutate(
    resid_spatial = resid1
  ) %>%
  filter(
    is.finite(longitude),
    is.finite(latitude),
    is.finite(resid_spatial)
  ) %>%
  group_by(longitude, latitude) %>%
  mutate(
    n_at_site = n(),
    duplicate_site = n_at_site > 1
  ) %>%
  ungroup() %>%
  mutate(
    lon_jitter = longitude,
    lat_jitter = latitude
  )

# Add random noise only to duplicate sites, so we have 8408 observations with unique spatial coordinates
set.seed(123)
idx <- which(rent_spatial$duplicate_site)

rent_spatial$lon_jitter[idx] <-
  rent_spatial$longitude[idx] +
  runif(length(idx), -1e-6, 1e-6)

rent_spatial$lat_jitter[idx] <-
  rent_spatial$latitude[idx] +
  runif(length(idx), -1e-6, 1e-6)

#Using GeoR package to get some illustrations of our data
AB_geo <- as.geodata(
  rent_spatial,
  coords.col = c("lon_jitter", "lat_jitter"),
  data.col = "resid_spatial"
)

summary(AB_geo)
plot(AB_geo) 

# variogram
#vario_cloud <- variog(AB_geo, option = "cloud")
#plot(vario_cloud)

vario <- variog(AB_geo, uvec = seq(0, 0.2, by = 0.01))
plot(vario)






