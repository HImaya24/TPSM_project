# STEP 1 — Load Libraries
library(tidyverse)
library(ggplot2)
library(reshape2)
library(gridExtra)

# STEP 2 — Load Datasets
# Note: Using relative paths for portability
raw <- read.csv("data/fifa21_raw.csv", stringsAsFactors = FALSE)
clean <- read.csv("data/fifa21_clean.csv", stringsAsFactors = FALSE)

# STEP 3 — Advanced Data Quality Profiling
cat("\n--- DATA QUALITY OVERVIEW ---\n")

dq_summary <- data.frame(
  Metric = c("Total Rows", "Total Columns", "Missing Values (Total)", "Duplicate Rows"),
  Raw = c(nrow(raw), ncol(raw), sum(is.na(raw)), sum(duplicated(raw))),
  Clean = c(nrow(clean), ncol(clean), sum(is.na(clean)), sum(duplicated(clean)))
)
print(dq_summary)
dir.create("outputs/descriptive", showWarnings = FALSE, recursive = TRUE)
write.csv(dq_summary, "outputs/descriptive/data_quality_summary.csv", row.names = FALSE)
cat("Data Quality Summary saved to outputs/descriptive/data_quality_summary.csv\n")

# STEP 4 — Distribution Analysis (Before vs After)
# Let's look at 'Wage' as a key indicator of preprocessing impact (outlier removal + scaling)
# Note: We need to handle the fact that 'clean' is scaled and 'raw' is messy strings/numeric
# For comparison, we'll try to find overlapping numeric names

# Wage Processing impact
p1 <- ggplot(raw %>% select(Wage) %>% head(1000), aes(x=1:1000, y=Wage)) + 
  geom_point(alpha=0.3) + 
  ggtitle("Raw Wage Data (Messy/String)") +
  theme_minimal()

# Assuming clean has a 'Wage' column that is now numeric and scaled
p2 <- ggplot(clean, aes(x=Wage)) + 
  geom_density(fill="steelblue", alpha=0.7) + 
  ggtitle("Clean Wage Distribution (Standardized)") +
  theme_minimal()

# STEP 5 — Correlation Network/Heatmap Comparison
# This shows how preprocessing restores/clarifies linear relationships
get_cor_matrix <- function(df) {
  df %>% select(where(is.numeric)) %>% cor(use="complete.obs")
}

# Select top 10 most correlated features with OVA for visualization
top_cols <- c("OVA", "Potential", "Special", "Value", "Wage", "Release.Clause", "Weight", "Height")
# (Adjustment: names might vary between raw and clean, let's use what's likely there)

# Check available columns in clean
clean_num <- clean %>% select(where(is.numeric))
cols_to_use <- intersect(names(clean_num), c("Age", "Wage", "Value", "Height", "Weight", "X.OVA", "Potential"))

if(length(cols_to_use) > 2) {
  cor_clean <- cor(clean_num[, cols_to_use], use="complete.obs")
  
  # Heatmap
  melten_cor <- melt(cor_clean)
  p3 <- ggplot(melten_cor, aes(x=Var1, y=Var2, fill=value)) +
    geom_tile() +
    scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("Correlation Structure (Clean Data)")
  
  print(p3)
}

# STEP 6 — Save Descriptive Analytics Report
dir.create("outputs/descriptive", showWarnings = FALSE, recursive = TRUE)
png("outputs/descriptive/distribution_impact.png", width=800, height=400)
grid.arrange(p1, p2, ncol=2)
dev.off()

cat("\nDescriptive Analytics Complete. Report saved to outputs/descriptive/\n")
