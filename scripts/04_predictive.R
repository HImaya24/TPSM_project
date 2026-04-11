install.packages("tidyverse")
install.packages("caret")
install.packages("rpart")
install.packages("randomForest")
install.packages("Metrics")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("effsize")
install.packages("e1071")
install.packages("gbm")

# STEP 1 — Load Libraries

library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(Metrics)
library(ggplot2)
library(gbm)

# STEP 2 — Load Data

setwd("C:/Users/User/Desktop/TPSM_Project/TPSM_project")

# Load raw data
raw   <- read.csv("data/fifa21_raw.csv") %>%
  select(where(is.numeric)) %>%
  na.omit()

# Load clean data
clean <- read.csv("data/fifa21_clean.csv") %>%
  select(where(is.numeric)) %>%
  na.omit()

#Step 3 — Train/Test Split (80/20)
set.seed(42)


# Raw split
trainIndex_raw <- createDataPartition(raw$X.OVA, p=0.8, list=FALSE)
train_raw <- raw[trainIndex_raw, ]
test_raw  <- raw[-trainIndex_raw, ]

# Clean split
trainIndex_clean <- createDataPartition(clean$X.OVA, p=0.8, list=FALSE)
train_clean <- clean[trainIndex_clean, ]
test_clean  <- clean[-trainIndex_clean, ]

cat("Train raw rows:",   nrow(train_raw),   "| Test raw rows:",   nrow(test_raw), "\n")
cat("Train clean rows:", nrow(train_clean), "| Test clean rows:", nrow(test_clean), "\n")

# STEP 4 — Metrics Function
getMetrics <- function(actual, predicted) {
  r2   <- cor(actual, predicted)^2
  rmse <- rmse(actual, predicted)
  mae  <- mae(actual, predicted)
  return(c(R2   = round(r2,   4),
           RMSE = round(rmse, 4),
           MAE  = round(mae,  4)))
}

# STEP 5 — Remove Leaky Columns
# Remove columns that are too closely related to X.OVA
leaky_cols <- c("POT", "BOV", "Growth", "Total.Stats", "Base.Stats")

train_raw   <- train_raw   %>% select(-any_of(leaky_cols))
test_raw    <- test_raw    %>% select(-any_of(leaky_cols))
train_clean <- train_clean %>% select(-any_of(leaky_cols))
test_clean  <- test_clean  %>% select(-any_of(leaky_cols))


# STEP 6 — Model 1: Linear Regression
cat("\nTraining Linear Regression...\n")

# Train
lr_raw   <- lm(X.OVA ~ ., data=train_raw)
lr_clean <- lm(X.OVA ~ ., data=train_clean)

# Predict
lr_raw_pred   <- predict(lr_raw,   test_raw)
lr_clean_pred <- predict(lr_clean, test_clean)

# Metrics
lr_raw_m   <- getMetrics(test_raw$X.OVA,   lr_raw_pred)
lr_clean_m <- getMetrics(test_clean$X.OVA, lr_clean_pred)

cat("Linear Regression - Raw:  ", lr_raw_m,   "\n")
cat("Linear Regression - Clean:", lr_clean_m, "\n")


# STEP 7 — Model 2: Decision Tree

cat("\nTraining Decision Tree...\n")
# Tuned Decision Tree with more depth
dt_raw   <- rpart(X.OVA ~ ., data=train_raw,
                  control=rpart.control(maxdepth=10,
                                        minsplit=10,
                                        cp=0.001))

dt_clean <- rpart(X.OVA ~ ., data=train_clean,
                  control=rpart.control(maxdepth=10,
                                        minsplit=10,
                                        cp=0.001))

# Predict and get metrics same as before
dt_raw_pred   <- predict(dt_raw,   test_raw)
dt_clean_pred <- predict(dt_clean, test_clean)

dt_raw_m   <- getMetrics(test_raw$X.OVA,   dt_raw_pred)
dt_clean_m <- getMetrics(test_clean$X.OVA, dt_clean_pred)

cat("Decision Tree Tuned - Raw:  ", dt_raw_m,   "\n")
cat("Decision Tree Tuned - Clean:", dt_clean_m, "\n")

# STEP 8 — Model 3: Random Forest

cat("\nTraining Random Forest ...\n")

# Train
rf_raw   <- randomForest(X.OVA ~ ., data=train_raw,   ntree=100)
rf_clean <- randomForest(X.OVA ~ ., data=train_clean, ntree=100)

# Predict
rf_raw_pred   <- predict(rf_raw,   test_raw)
rf_clean_pred <- predict(rf_clean, test_clean)

# Metrics
rf_raw_m   <- getMetrics(test_raw$X.OVA,   rf_raw_pred)
rf_clean_m <- getMetrics(test_clean$X.OVA, rf_clean_pred)

cat("Random Forest - Raw:  ", rf_raw_m,   "\n")
cat("Random Forest - Clean:", rf_clean_m, "\n")


# STEP 9 — Model 4: Gradient Boosting

cat("\nTraining Gradient Boosting ...\n")

gb_raw   <- train(X.OVA ~ ., data=train_raw,
                  method="gbm",
                  verbose=FALSE)

gb_clean <- train(X.OVA ~ ., data=train_clean,
                  method="gbm",
                  verbose=FALSE)

gb_raw_pred   <- predict(gb_raw,   test_raw)
gb_clean_pred <- predict(gb_clean, test_clean)

gb_raw_m   <- getMetrics(test_raw$X.OVA,   gb_raw_pred)
gb_clean_m <- getMetrics(test_clean$X.OVA, gb_clean_pred)

cat("Gradient Boosting - Raw:  ", gb_raw_m,   "\n")
cat("Gradient Boosting - Clean:", gb_clean_m, "\n")


# STEP 10 — Full Comparison Table (4 Models)
# ============================================

comparison <- data.frame(
  Model      = c("Linear Regression",
                 "Decision Tree",
                 "Random Forest",
                 "Gradient Boosting"),
  Raw_R2     = c(lr_raw_m["R2"],  dt_raw_m["R2"],
                 rf_raw_m["R2"],  gb_raw_m["R2"]),
  Clean_R2   = c(lr_clean_m["R2"],dt_clean_m["R2"],
                 rf_clean_m["R2"],gb_clean_m["R2"]),
  Raw_RMSE   = c(lr_raw_m["RMSE"],dt_raw_m["RMSE"],
                 rf_raw_m["RMSE"],gb_raw_m["RMSE"]),
  Clean_RMSE = c(lr_clean_m["RMSE"],dt_clean_m["RMSE"],
                 rf_clean_m["RMSE"],gb_clean_m["RMSE"]),
  Raw_MAE    = c(lr_raw_m["MAE"], dt_raw_m["MAE"],
                 rf_raw_m["MAE"], gb_raw_m["MAE"]),
  Clean_MAE  = c(lr_clean_m["MAE"],dt_clean_m["MAE"],
                 rf_clean_m["MAE"],gb_clean_m["MAE"])
)

cat("\n===== MODEL COMPARISON TABLE (4 MODELS) =====\n")
print(comparison)

write.csv(comparison,
          "outputs/results/model_comparison.csv",
          row.names=FALSE)
cat("Comparison table saved!\n")


# STEP 11 — R² Bar Chart (4 Models)
# ============================================

r2_df <- data.frame(
  Model = rep(c("Linear Regression", "Decision Tree",
                "Random Forest",     "Gradient Boosting"), 2),
  R2    = c(lr_raw_m["R2"],  dt_raw_m["R2"],
            rf_raw_m["R2"],  gb_raw_m["R2"],
            lr_clean_m["R2"],dt_clean_m["R2"],
            rf_clean_m["R2"],gb_clean_m["R2"]),
  Type  = rep(c("Raw", "Clean"), each=4)
)

r2_plot <- ggplot(r2_df, aes(x=Model, y=R2, fill=Type)) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_manual(values=c("Raw"="tomato", "Clean"="steelblue")) +
  geom_text(aes(label=round(R2, 3)),
            position=position_dodge(width=0.9),
            vjust=-0.5, size=3.5) +
  labs(title="R² Score: Raw vs Preprocessed Data (4 Models)",
       subtitle="Higher R² = Better model accuracy",
       x="Model", y="R² Score") +
  theme_minimal() +
  ylim(0, 1.1)

print(r2_plot)

png("outputs/charts/r2_comparison.png", width=900, height=500)
print(r2_plot)
dev.off()
cat("R² chart saved!\n")

# STEP 12 — RMSE Bar Chart (4 Models)
# ============================================

rmse_df <- data.frame(
  Model = rep(c("Linear Regression", "Decision Tree",
                "Random Forest",     "Gradient Boosting"), 2),
  RMSE  = c(lr_raw_m["RMSE"],  dt_raw_m["RMSE"],
            rf_raw_m["RMSE"],  gb_raw_m["RMSE"],
            lr_clean_m["RMSE"],dt_clean_m["RMSE"],
            rf_clean_m["RMSE"],gb_clean_m["RMSE"]),
  Type  = rep(c("Raw", "Clean"), each=4)
)

rmse_plot <- ggplot(rmse_df, aes(x=Model, y=RMSE, fill=Type)) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_manual(values=c("Raw"="tomato", "Clean"="steelblue")) +
  geom_text(aes(label=round(RMSE, 3)),
            position=position_dodge(width=0.9),
            vjust=-0.5, size=3.5) +
  labs(title="RMSE: Raw vs Preprocessed Data (4 Models)",
       subtitle="Lower RMSE = Better model accuracy",
       x="Model", y="RMSE") +
  theme_minimal()

print(rmse_plot)

png("outputs/charts/rmse_comparison.png", width=900, height=500)
print(rmse_plot)
dev.off()
cat("RMSE chart saved!\n")

# STEP 13 — MAE Bar Chart (4 Models)
# ============================================

mae_df <- data.frame(
  Model = rep(c("Linear Regression", "Decision Tree",
                "Random Forest",     "Gradient Boosting"), 2),
  MAE   = c(lr_raw_m["MAE"],  dt_raw_m["MAE"],
            rf_raw_m["MAE"],  gb_raw_m["MAE"],
            lr_clean_m["MAE"],dt_clean_m["MAE"],
            rf_clean_m["MAE"],gb_clean_m["MAE"]),
  Type  = rep(c("Raw", "Clean"), each=4)
)

mae_plot <- ggplot(mae_df, aes(x=Model, y=MAE, fill=Type)) +
  geom_bar(stat="identity", position="dodge") +
  scale_fill_manual(values=c("Raw"="tomato", "Clean"="steelblue")) +
  geom_text(aes(label=round(MAE, 3)),
            position=position_dodge(width=0.9),
            vjust=-0.5, size=3.5) +
  labs(title="MAE: Raw vs Preprocessed Data (4 Models)",
       subtitle="Lower MAE = Better model accuracy",
       x="Model", y="MAE") +
  theme_minimal()

print(mae_plot)

png("outputs/charts/mae_comparison.png", width=900, height=500)
print(mae_plot)
dev.off()
cat("MAE chart saved!\n")

# STEP 14 — Final Summary
# ============================================

cat("\n========== FINAL SUMMARY (4 MODELS) ==========\n")
cat(sprintf("%-20s | Raw R² | Clean R² | Raw RMSE | Clean RMSE | Improvement\n", "Model"))
cat(rep("-", 75), "\n", sep="")

models   <- c("Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting")
raw_rmse <- c(lr_raw_m["RMSE"], dt_raw_m["RMSE"], rf_raw_m["RMSE"], gb_raw_m["RMSE"])
cln_rmse <- c(lr_clean_m["RMSE"],dt_clean_m["RMSE"],rf_clean_m["RMSE"],gb_clean_m["RMSE"])
raw_r2   <- c(lr_raw_m["R2"],  dt_raw_m["R2"],  rf_raw_m["R2"],  gb_raw_m["R2"])
cln_r2   <- c(lr_clean_m["R2"],dt_clean_m["R2"],rf_clean_m["R2"],gb_clean_m["R2"])

for(i in 1:4) {
  improvement <- round((raw_rmse[i] - cln_rmse[i]) / raw_rmse[i] * 100, 1)
  cat(sprintf("%-20s | %.4f | %.4f   | %.4f   | %.4f     | %.1f%% better\n",
              models[i], raw_r2[i], cln_r2[i], raw_rmse[i], cln_rmse[i], improvement))
}

cat(rep("=", 75), "\n", sep="")
cat("All outputs saved to outputs/ folder\n")
cat("Predictive Analytics COMPLETE\n")










