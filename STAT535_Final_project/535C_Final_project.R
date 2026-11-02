library(readr)
library(dplyr)
library(ggplot2)
library(geoR)
library(MASS)
library(rjags)
library(coda)
library(fields)

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
  # We would like to remove some extreme observations, or bad data points. After running this chunk of code, we have 10803 observations
  filter(
    !is.na(price),
    price > 200,
    price < 10000,
    !is.na(sq_feet),
    sq_feet > 250,
    beds <= 6,
    !type %in% c("Office Space", "Parking Spot", "Storage"),
    is.finite(latitude),
    is.finite(longitude)
  )

#Since prices are positive, we work with log-price
rent_AB <- rent_AB %>% mutate(log_price = log(price))

## Part 2 Regression model

#We first fit a baseline regression model where we do not consider any spatial dependency.

# The covariates we consider are the number of bedrooms, number of bathrooms, is pets allowed (Binary) and the property type (5 main types)

rent_AB_lm <- lm(
  log_price ~ beds + baths + pets_allowed + property_type,
  data = rent_AB
)

summary(rent_AB)
summary(rent_AB_lm)


fitted_log_price <- predict(rent_AB_lm, newdata = rent_AB)
residuals <- rent_AB$log_price - fitted_log_price

# Plotting fitted log-price versus actual log-price
plot(
  fitted_log_price, rent_AB$log_price,
  xlab = "Fitted log-price",
  ylab = "Actual log-price",
  main = "",
  pch = 16,
  cex = 0.6
)
abline(0, 1, col = "red", lwd = 2)

# Plotting fitted log-price versus the residuals
plot(
  fitted_log_price, residuals,
  xlab = "Fitted log-price",
  ylab = "Residuals",
  main = "",
  pch = 16,
  cex = 0.6
)
abline(h=0, col = "red", lwd = 2)







