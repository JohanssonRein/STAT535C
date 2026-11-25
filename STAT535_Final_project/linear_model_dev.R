
model_df_1 <- rent_AB %>%
              mutate(log_sq_feet = log(sq_feet),
                     i_one_bath = as.integer(baths == 1),
                     i_one_bed = as.integer(beds == 1),
                     i_1b1b = i_one_bed*i_one_bath,
                     beds_c3 = pmin(beds, 3),
                     beds_c4 = pmin(beds, 4),
                     beds_c5 = pmin(beds, 5),
                     baths_f1_c3 = pmax(1, pmin(baths, 3)),
                     i_Furnished = as.integer(furnishing == "Furnished"),
                     i_Unfurnished = as.integer(furnishing == "Unfurnished"),
                     i_beds_ge4 = as.integer(beds >= 4),
                     i_beds_ge5 = as.integer(beds >= 5),
                     i_baths_gt3 = as.integer(baths > 3),
                     i_immed_avail = as.integer(availability_date == "Immediate"),
                     i_Furnished_i_Apartment = i_Furnished*i_Apartment,
                     i_Furnished_i_House = i_Furnished*i_House,
                     sq_feet_i_Apartment = sq_feet*i_Apartment,
                     sq_feet_i_House = sq_feet*i_House,
                     baths_c3 = pmin(baths, 3),
                     baths_c4 = pmin(baths, 4),
                     baths_f1_c4 = pmax(1, pmin(baths, 4)),
                     beds_c5_i_Apartment = beds_c5 * i_Apartment,
                     baths_c4_i_Apartment = baths_c4 * i_Apartment,
                     beds_c5_i_House = beds_c5 * i_House,
                     beds_c5_i_Basement = beds_c5 * i_Basement,
                     baths_c4_i_House = baths_c4 * i_House,
                     i_not_Room = as.integer(i_Room == 0),
                     beds_c5_i_not_Room = beds_c5 * i_not_Room,
                     baths_c4_i_not_Room = baths_c4 * i_not_Room)

model_df <- model_df_1 %>%
            filter(!is.na(sq_feet))

set.seed(535)
train <- sample(1:nrow(model_df), size = floor(0.8 * nrow(model_df)))

model_df_train <- model_df[train, ]
model_df_test  <- model_df[-train, ]

rnp1 <- function(var, n_bins = 20, data = model_df_train) {
  x <- data[[var]]
  unique_x <- sort(unique(x))
  x_bins <- ntile(unique_x, n_bins)
  
  plot_df <- data %>%
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

rnp2 <- function(var, mod, n_bins = 20, data = model_df_train) {
  x <- data[[var]]
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

##### Removing sq_feet ####

set.seed(535)
train <- sample(1:nrow(model_df_1), size = floor(0.8 * nrow(model_df_1)))

model_df_1_train <- model_df_1[train, ]
model_df_1_test  <- model_df_1[-train, ]

rnp1("beds", data = model_df_1_train)
rnp1("beds_c5", data = model_df_1_train)
rnp1("baths", data = model_df_1_train)
rnp1("baths_f1_c3", data = model_df_1_train)

mod_12 <- lm(price ~ beds_c5, data = model_df_1_train)
summary(mod_12)

rnp2("beds", mod_12, data = model_df_1_train)
rnp2("baths_c3", mod_12, data = model_df_1_train)
rnp2("baths_c4", mod_12, data = model_df_1_train)
rnp2("baths_f1_c3", mod_12, data = model_df_1_train)
rnp2("baths_f1_c4", mod_12, data = model_df_1_train)

mod_13 <- lm(price ~ beds_c5 + baths_c4, data = model_df_1_train)
summary(mod_13)

rnp2("beds", mod_13, data = model_df_1_train)
rnp2("baths", mod_13, data = model_df_1_train)
rnp2("i_pets_allowed", mod_13, data = model_df_1_train)
rnp2("i_Room", mod_13, data = model_df_1_train)

mod_14 <- lm(price ~ beds_c5 + baths_c4 + i_pets_allowed + i_Room, data = model_df_1_train)
summary(mod_14)

rnp2("beds", mod_14, data = model_df_1_train)
rnp2("baths", mod_14, data = model_df_1_train)
rnp2("i_pets_allowed", mod_14, data = model_df_1_train)
rnp2("i_Room", mod_14, data = model_df_1_train)

rnp2("i_House", mod_14, data = model_df_1_train)
rnp2("i_Furnished", mod_14, data = model_df_1_train)
rnp2("i_Apartment", mod_14, data = model_df_1_train)
rnp2("i_Furnished_i_Apartment", mod_14, data = model_df_1_train)

mod_15 <- lm(price ~ beds_c5 + baths_c4 + i_pets_allowed + i_Room + i_Basement + i_Furnished +
                      i_Furnished_i_Apartment, data = model_df_1_train)
summary(mod_15)

rnp2("beds", mod_15, data = model_df_1_train)
rnp2("baths", mod_15, data = model_df_1_train)
rnp2("i_pets_allowed", mod_15, data = model_df_1_train)
rnp2("i_Room", mod_15, data = model_df_1_train)
rnp2("i_Basement", mod_15, data = model_df_1_train)
rnp2("i_Furnished", mod_15, data = model_df_1_train)
rnp2("i_Apartment", mod_15, data = model_df_1_train)
rnp2("i_Furnished_i_Apartment", mod_15, data = model_df_1_train)

rnp2("i_House", mod_15, data = model_df_1_train)
rnp2("beds_c5_i_Apartment", mod_15, data = model_df_1_train)
rnp2("baths_c4_i_Apartment", mod_15, data = model_df_1_train)
rnp2("beds_c5_i_House", mod_15, data = model_df_1_train)
rnp2("baths_c4_i_House", mod_15, data = model_df_1_train)

mod_16 <- lm(price ~ i_pets_allowed + i_Room + i_Basement + i_Furnished +
               i_Furnished_i_Apartment + beds_c5_i_Apartment + beds_c5_i_not_Room + baths_c4_i_not_Room, data = model_df_1_train)
summary(mod_16)

rnp2("beds_c5_i_House", mod_16, data = model_df_1_train)
rnp2("beds_c5", mod_16, data = model_df_1_train)

rnp1("beds_c5", data = model_df_1_train %>% filter(i_Basement == 1))
rnp1("beds_c5", data = model_df_1_train %>% filter(i_Room == 1))
rnp1("baths_c4", data = model_df_1_train %>% filter(i_Basement == 1))
rnp1("baths_c4", data = model_df_1_train %>% filter(i_Room == 1))
rnp1("beds_c5", data = model_df_1_train %>% filter(i_not_Room == 1))
rnp1("baths_c4", data = model_df_1_train %>% filter(i_not_Room == 1))

mod_17 <- lm(log_price ~ i_pets_allowed + i_Room + i_Basement + i_Furnished +
               i_Furnished_i_Apartment + beds_c5_i_Apartment + beds_c5_i_not_Room + baths_c4_i_not_Room, data = model_df_1_train)
summary(mod_17)

rnp2("beds_c5", mod_17, data = model_df_1_train)
rnp2("baths_c4", mod_17, data = model_df_1_train)
rnp2("i_Apartment", mod_17, data = model_df_1_train)
rnp2("i_House", mod_17, data = model_df_1_train)

fitted_pred <- predict(mod_17, newdata = model_df_1_test)

plot(fitted_pred, model_df_1_test$log_price)
lines(x = 6:10, y = 6:10)



