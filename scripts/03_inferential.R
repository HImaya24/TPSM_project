# STEP 1 — Install & Load Libraries (run once)
# install.packages(c("tidyverse","caret","ranger","effsize","ggplot2"))

library(tidyverse)
library(caret)
library(ranger)
library(effsize)
library(ggplot2)

# STEP 2 — Set Working Directory
setwd("C:/Users/HP/Downloads/TPSM_dataset_forAssignment")

# STEP 3 — Load Datasets
raw <- read.csv("fifa21_raw_data.csv", stringsAsFactors = FALSE)
clean <- read.csv("fifa21_clean_final.csv", stringsAsFactors = FALSE)

cat("Raw:", dim(raw), "\n")
cat("Clean:", dim(clean), "\n")

# STEP 4 — Keep COMMON columns only (important for fair comparison)
common_cols <- intersect(names(raw), names(clean))
raw <- raw[, common_cols]
clean <- clean[, common_cols]

# STEP 5 — Select NUMERIC columns
raw_numeric <- raw %>% select(where(is.numeric))
clean_numeric <- clean %>% select(where(is.numeric))

# STEP 6 — Handle missing values consistently
raw_numeric <- raw_numeric %>% drop_na()
clean_numeric <- clean_numeric %>% drop_na()

# STEP 7 — Align row counts (VERY IMPORTANT)
min_rows <- min(nrow(raw_numeric), nrow(clean_numeric))
raw_numeric <- raw_numeric[1:min_rows, ]
clean_numeric <- clean_numeric[1:min_rows, ]

cat("Final rows used:", min_rows, "\n")

# STEP 8 — Separate Features and Target (OVA)
X_raw <- raw_numeric %>% select(-X.OVA)
y_raw <- raw_numeric$X.OVA

X_clean <- clean_numeric %>% select(-X.OVA)
y_clean <- clean_numeric$X.OVA

# STEP 9 — Hypothesis
cat("\n========== HYPOTHESIS ==========\n")
cat("H₀: Preprocessing has NO significant effect on model accuracy\n")
cat("H₁: Preprocessing has a SIGNIFICANT effect on model accuracy\n")
cat("α = 0.05\n")

# STEP 10 — Cross Validation Setup
set.seed(42)

ctrl <- trainControl(
  method = "cv",
  number = 10
)


# RANDOM FOREST (RANGER)
# =========================================================

cat("\nRunning Random Forest (RAW)...\n")

model_raw <- train(
  x = X_raw,
  y = y_raw,
  method = "ranger",
  trControl = ctrl,
  metric = "Rsquared",
  num.trees = 50,
  tuneLength = 1
)

raw_scores <- model_raw$resample$Rsquared

cat("Raw Mean R²:", mean(raw_scores), "\n")

cat("\nRunning Random Forest (CLEAN)...\n")

model_clean <- train(
  x = X_clean,
  y = y_clean,
  method = "ranger",
  trControl = ctrl,
  metric = "Rsquared",
  num.trees = 50,
  tuneLength = 1
)

clean_scores <- model_clean$resample$Rsquared

cat("Clean Mean R²:", mean(clean_scores), "\n")


# VISUALIZATION


scores_df <- data.frame(
  Fold = rep(1:10, 2),
  R2 = c(raw_scores, clean_scores),
  Type = rep(c("Raw", "Clean"), each = 10)
)

ggplot(scores_df, aes(Fold, R2, color = Type)) +
  geom_line() +
  geom_point() +
  ggtitle("R² Comparison (Raw vs Clean)")

# PAIRED T-TEST (CORE ANALYSIS)


cat("\n========== PAIRED T-TEST ==========\n")

t_test <- t.test(clean_scores, raw_scores, paired = TRUE)

print(t_test)


#  EFFECT SIZE (Hedges_G)


hedges_g <- cohen.d(clean_scores, raw_scores, paired = TRUE, hedges.correction = TRUE)

cat("\n Hedges g :" ,hedges_g$estimate,"\n")


# SUMMARY METRICS


mean_raw <- mean(raw_scores)
mean_clean <- mean(clean_scores)

cat("\nMean Raw:", mean_raw)
cat("\nMean Clean:", mean_clean)
cat("\nDifference:", mean_clean - mean_raw)


# FINAL DECISION


cat("\n========== FINAL DECISION ==========\n")

if (t_test$p.value < 0.05) {
  cat("Reject H₀ → Preprocessing has SIGNIFICANT effect\n")
} else {
  cat("Fail to Reject H₀ → No significant effect\n")
}

mean_diff <- mean_clean - mean_raw
improvement_pct <- (mean_diff / mean_raw) * 100

t_test_result <- t_test
effect <- hedges_g
alpha <- 0.05

cat("\n========== SUMMARY RESULTS TABLE ==========\n")

results_table <- data.frame(
  Metric = c(
    "Mean R² (Raw Data)",
    "Mean R² (Clean Data)",
    "Standard Deviation (Raw)",
    "Standard Deviation (Clean)",
    "Mean Difference (Clean - Raw)",
    "Improvement Percentage",
    "t-statistic",
    "Degrees of Freedom",
    "p-value",
    "Significance Level (α)",
    "Statistical Decision",
    "Hedges g (Effect Size)",
    "Effect Size Interpretation",
    "95% CI Lower Bound",
    "95% CI Upper Bound"
  ),
  Value = c(
    round(mean_raw, 4),
    round(mean_clean, 4),
    round(sd(raw_scores), 4),
    round(sd(clean_scores), 4),
    round(mean_clean-mean_raw, 4),
    paste0(round(improvement_pct, 2), "%"),
    round(t_test_result$statistic, 4),
    t_test_result$parameter,
    format(t_test_result$p.value, scientific = TRUE),
    alpha,
    ifelse(t_test_result$p.value < alpha, "Reject H₀", "Fail to Reject H₀"),
    round(effect$estimate, 4),
    ifelse(abs(effect$estimate) >= 0.8, "Large",
           ifelse(abs(effect$estimate) >= 0.5, "Medium",
                  ifelse(abs(effect$estimate) >= 0.2, "Small", "Negligible"))),
    round(t_test_result$conf.int[1], 6),
    round(t_test_result$conf.int[2], 6)
  )
)

print(results_table)
