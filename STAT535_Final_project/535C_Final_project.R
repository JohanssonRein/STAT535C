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




#Since prices are positive, we work with log-price (maybe it is worth to examine using price instead of log-price)
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

rent_AB_train <- rent_AB[train, ] # Training set has 7725 observations
rent_AB_test  <- rent_AB[-train, ] # Testing set has 1932 observations

rent_AB_lm <- lm(
  log_price ~ beds + baths + i_pets_allowed + property_type,
  data = rent_AB_train,
  na.action = na.exclude
)

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

#nrow(rent_AB) = 9657

# We start by grouping the observations with the same spatial location (they are likely from the same apartment building)


rent_AB_uni <- rent_AB_train %>%
  filter(
    is.finite(longitude),
    is.finite(latitude),
    is.finite(resid_lm)
  ) %>%
  group_by(longitude, latitude) %>%
  summarise(
    n_site = n(),
    mean_residual = mean(resid_lm),
    .groups = "drop"
  )

cat(
  "Training observations with valid residuals:",
  sum(is.finite(rent_AB_train$resid_lm)), # 7721 training observations with valid residual
  "\n"
)

cat(
  "Unique training locations:",
  nrow(rent_AB_uni), #3125 Unique training locations
  "\n"
)

#Create GeoR data type
geo_cal <- geoR::as.geodata(
  rent_AB_uni,
  coords.col = c("longitude", "latitude"),
  data.col = "mean_residual"
)

summary(geo_cal)

vario_cal <- variog(
  geo_cal,
  uvec = seq(0,0.2, by = 0.01),
  estimator.type = "classical"
)

plot(vario_cal)



vario <- variog(geo_cal, uvec = seq(0, 0.15, by = 0.01))
plot(vario)








