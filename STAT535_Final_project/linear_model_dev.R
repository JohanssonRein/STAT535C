
model_df <- rent_AB %>%
            filter(!is.na(sq_feet)) %>%
            mutate(log_sq_feet = log(sq_feet),
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

set.seed(535)
train <- sample(1:nrow(model_df), size = floor(0.8 * nrow(model_df)))

model_df_train <- model_df[train, ]
model_df_test  <- model_df[-train, ]

rnp1 <- function(var, n_bins = 20) {
  x <- model_df_train[[var]]
  unique_x <- sort(unique(x))
  x_bins <- ntile(unique_x, n_bins)
  
  plot_df <- model_df_train %>%
              dplyr::select(price) %>%
              mutate(x = x,
                     x_grouped = x_bins[match(x, unique_x)]) %>%
              group_by(x_grouped) %>%
              summarise(x_mean = mean(x),
                        x_count = n(),
                        price_mean = mean(price))
  print(plot_df, n = n_bins)
  plot(plot_df[["x_mean"]], plot_df[["price_mean"]])
}

rnp2 <- function(var, mod, n_bins = 20) {
  x <- model_df_train[[var]]
  unique_x <- sort(unique(x))
  x_bins <- ntile(unique_x, n_bins)
  
  plot_df <- tibble(res = residuals(mod),
                    x = x,
                    x_grouped = x_bins[match(x, unique_x)]) %>%
              group_by(x_grouped) %>%
              summarise(x_mean = mean(x),
                        x_count = n(),
                        res_mean = mean(res))
  print(plot_df, n = n_bins)
  plot(plot_df[["x_mean"]], plot_df[["res_mean"]])
}

rnp1("sq_feet", 50)
rnp1("sq_feet")
rnp1("sq_feet_i_Apartment")
rnp1("sq_feet_i_House")
rnp1("beds")
rnp1("beds_c5")
rnp1("baths")

mod_1 <- lm(price ~ sq_feet, data = model_df_train)
summary(mod_1)

rnp2("sq_feet", mod_1, 50)
rnp2("sq_feet", mod_1)
rnp2("sq_feet_i_Apartment", mod_1)
rnp2("sq_feet_i_House", mod_1)
rnp2("sq_feet", mod_1, 10)
rnp2("beds_c5", mod_1)
rnp2("baths", mod_1)
rnp2("i_Apartment", mod_1)
rnp2("i_1b1b", mod_1, 2)

mod_2 <- lm(price ~ sq_feet + i_Apartment, data = model_df_train)
summary(mod_2)

rnp2("sq_feet", mod_2, 50)
rnp2("i_Apartment", mod_2)
rnp2("baths", mod_2)
rnp2("beds_c3", mod_2)
rnp2("beds_c4", mod_2)
rnp2("beds_c5", mod_2)

mod_3 <- lm(price ~ sq_feet + i_Apartment + beds_c5, data = model_df_train)
summary(mod_3)

rnp2("i_pets_allowed", mod_3)
rnp2("baths", mod_3)
rnp2("beds", mod_3)
rnp2("i_1b1b", mod_3)
rnp2("i_Furnished", mod_3)
rnp2("i_Unfurnished", mod_3)
rnp2("i_immed_avail", mod_3)

mod_4 <- lm(price ~ sq_feet + i_Apartment + beds_c3 + i_beds_ge5 + i_pets_allowed + i_Furnished, data = model_df_train)
summary(mod_4)

rnp2("sq_feet", mod_4, 10)
rnp2("i_pets_allowed", mod_4)
rnp2("beds_c3", mod_4)
rnp2("beds", mod_4)
rnp2("i_1b1b", mod_4)
rnp2("i_Furnished", mod_4)
rnp2("i_beds_ge5", mod_4)
rnp2("i_immed_avail", mod_4)
rnp2("i_House", mod_4)

mod_5 <- lm(price ~ sq_feet + i_Apartment + beds_c3 + i_beds_ge5 + i_pets_allowed + i_Furnished + i_House, data = model_df_train)
summary(mod_5)


mod_6 <- lm(price ~ sq_feet + beds_c3 + i_beds_ge5 + i_pets_allowed +
                    i_Furnished*i_Apartment + i_Furnished*i_House, data = model_df_train)
summary(mod_6)

mod_7 <- lm(price ~ sq_feet + i_Apartment + beds_c3 + i_beds_ge5 + i_pets_allowed + i_House +
                    i_Furnished_i_Apartment + i_Furnished_i_House, data = model_df_train)
summary(mod_7)

rnp2("sq_feet", mod_7)
rnp2("i_Apartment", mod_7)
rnp2("i_pets_allowed", mod_7)
rnp2("beds_c3", mod_7)
rnp2("beds", mod_7)
rnp2("i_1b1b", mod_7)
rnp2("i_Furnished", mod_7)
rnp2("i_beds_ge5", mod_7)
rnp2("i_immed_avail", mod_7)
rnp2("i_House", mod_7)
rnp2("i_Furnished_i_Apartment", mod_7)
rnp2("i_Furnished_i_House", mod_7)

mod_8 <- lm(price ~ sq_feet + i_Apartment + beds_c3 + i_beds_ge5 + i_pets_allowed + i_House +
                    i_Furnished_i_Apartment, data = model_df_train)
summary(mod_8)

rnp2("i_Furnished", mod_8)
rnp2("sq_feet", mod_8)
rnp2("i_Room", mod_8)
rnp2("i_Basement", mod_8)

mod_9 <- lm(price ~ sq_feet + i_Apartment + beds_c3 + i_beds_ge5 + i_pets_allowed + i_House + i_Room +
              i_Furnished_i_Apartment, data = model_df_train)
summary(mod_9)

rnp2("sq_feet", mod_9)
rnp2("sq_feet", mod_9, 50)
rnp2("sq_feet_i_Apartment", mod_9)
rnp2("sq_feet_i_House", mod_9)

mod_10 <- lm(price ~ sq_feet + beds_c3 + i_beds_ge5 + i_pets_allowed + i_Room + i_Furnished +
              i_Furnished_i_Apartment + sq_feet_i_Apartment + sq_feet_i_House, data = model_df_train)
summary(mod_10)

rnp2("sq_feet", mod_10)
rnp2("sq_feet_i_Apartment", mod_10)
rnp2("sq_feet_i_House", mod_10)
rnp2("i_Apartment", mod_10)
rnp2("i_pets_allowed", mod_10)
rnp2("beds_c3", mod_10)
rnp2("beds", mod_10)
rnp2("i_1b1b", mod_10)
rnp2("i_Furnished", mod_10)
rnp2("i_beds_ge5", mod_10)
rnp2("i_immed_avail", mod_10)
rnp2("i_House", mod_10)
rnp2("i_Furnished_i_Apartment", mod_10)
rnp2("i_Furnished_i_House", mod_10)

rnp2("baths", mod_10)
rnp2("i_baths_gt3", mod_10)

mod_11 <- lm(log_price ~ sq_feet + beds_c3 + i_pets_allowed + i_Room + i_Furnished +
                          i_Furnished_i_Apartment + sq_feet_i_Apartment + sq_feet_i_House, data = model_df_train)
summary(mod_11)

rnp2("i_Basement", mod_11)
rnp2("i_Apartment", mod_11)

fitted_pred <- predict(mod_11, newdata = model_df_test)

plot(fitted_pred, model_df_test$log_price)
lines(x = 6:10, y = 6:10)







