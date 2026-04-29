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

# STEP 4.5 — Outlier Removal Visualization (Boxplots)
# Parse raw currencies to make them numeric for plotting
clean_currency <- function(x) {
  if (is.na(x) || x == "") return(NA)
  x <- gsub("[€£]", "", x)
  multiplier <- 1
  if(grepl("M", x)) multiplier <- 1000000
  else if(grepl("K", x)) multiplier <- 1000
  x <- gsub("[MK]", "", x)
  return(as.numeric(x) * multiplier)
}

raw_parsed <- raw %>% 
  mutate(
    Wage_num = sapply(Wage, clean_currency),
    Value_num = sapply(Value, clean_currency)
  )

bp1 <- ggplot(raw_parsed, aes(y=Wage_num)) + 
  geom_boxplot(fill="lightcoral") + 
  ggtitle("Raw Wage (with Outliers)") +
  ylab("Wage (Numeric)") +
  theme_minimal() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

bp2 <- ggplot(clean, aes(y=Wage)) + 
  geom_boxplot(fill="lightgreen") + 
  ggtitle("Clean Wage (Outliers Removed & Scaled)") +
  ylab("Wage (Scaled)") +
  theme_minimal() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

bp3 <- ggplot(raw_parsed, aes(y=Value_num)) + 
  geom_boxplot(fill="lightcoral") + 
  ggtitle("Raw Value (with Outliers)") +
  ylab("Value (Numeric)") +
  theme_minimal() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

bp4 <- ggplot(clean, aes(y=Value)) + 
  geom_boxplot(fill="lightgreen") + 
  ggtitle("Clean Value (Outliers Removed & Scaled)") +
  ylab("Value (Scaled)") +
  theme_minimal() + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

png("outputs/descriptive/outliers_boxplots.png", width=800, height=600)
grid.arrange(bp1, bp2, bp3, bp4, ncol=2)
dev.off()
cat("Outlier boxplots saved to outputs/descriptive/outliers_boxplots.png\n")

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
  
  # Clean Heatmap
  melten_cor_clean <- melt(cor_clean)
  p3_clean <- ggplot(melten_cor_clean, aes(x=Var1, y=Var2, fill=value)) +
    geom_tile() +
    scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("Correlation Structure (Clean Data)")
  
  # Raw Heatmap (using available numeric columns in raw)
  raw_num <- raw %>% select(where(is.numeric))
  cols_to_use_raw <- intersect(names(raw_num), c("Age", "X.OVA", "POT", "Total.Stats", "Base.Stats"))
  if(length(cols_to_use_raw) > 2) {
    cor_raw <- cor(raw_num[, cols_to_use_raw], use="complete.obs")
    melten_cor_raw <- melt(cor_raw)
    p3_raw <- ggplot(melten_cor_raw, aes(x=Var1, y=Var2, fill=value)) +
      geom_tile() +
      scale_fill_gradient2(low="blue", high="red", mid="white", midpoint=0) +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      ggtitle("Correlation Structure (Raw Data - Missing Key Features)")
    
    # Save Side-by-Side Heatmaps
    png("outputs/descriptive/correlation_heatmap.png", width=1200, height=600)
    grid.arrange(p3_raw, p3_clean, ncol=2)
    dev.off()
    cat("Correlation heatmap comparison saved to outputs/descriptive/correlation_heatmap.png\n")
  } else {
    png("outputs/descriptive/correlation_heatmap_backup.png", width=700, height=600)
    print(p3_clean)
    dev.off()
    cat("Correlation heatmap saved to outputs/descriptive/correlation_heatmap.png\n")
  }
}

# STEP 6 — Save Descriptive Analytics Report
dir.create("outputs/descriptive", showWarnings = FALSE, recursive = TRUE)
png("outputs/descriptive/distribution_impact.png", width=800, height=400)
grid.arrange(p1, p2, ncol=2)
dev.off()

cat("\nDescriptive Analytics Complete. Report saved to outputs/descriptive/\n")

# STEP 7 — Impact of Preprocessing on Relationships (Scatter plots)
# Let's plot Value vs OVA to see how outliers obscure relationships
target_col_raw <- if("X.OVA" %in% names(raw)) "X.OVA" else if("OVA" %in% names(raw)) "OVA" else NA
target_col_clean <- if("X.OVA" %in% names(clean)) "X.OVA" else if("OVA" %in% names(clean)) "OVA" else NA

if (!is.na(target_col_raw) && !is.na(target_col_clean)) {
  sp1 <- ggplot(raw_parsed, aes(x=Value_num, y=.data[[target_col_raw]])) +
    geom_point(alpha=0.3, color="darkred") +
    ggtitle("Raw Data: Value vs Rating") +
    xlab("Value (Numeric with Outliers)") +
    ylab("Overall Rating") +
    theme_minimal()
    
  sp2 <- ggplot(clean, aes(x=Value, y=.data[[target_col_clean]])) +
    geom_point(alpha=0.3, color="darkblue") +
    ggtitle("Clean Data: Value vs Rating") +
    xlab("Value (Scaled & Outliers Removed)") +
    ylab("Overall Rating") +
    theme_minimal()

  png("outputs/descriptive/scatter_impact_value.png", width=800, height=400)
  grid.arrange(sp1, sp2, ncol=2)
  dev.off()
  cat("Scatter plot comparison saved to outputs/descriptive/scatter_impact_value.png\n")
}

# STEP 8 — Impact of Standardization
# Let's show Age distribution before and after scaling
if ("Age" %in% names(raw) && "Age" %in% names(clean)) {
  dp1 <- ggplot(raw, aes(x=Age)) + 
    geom_density(fill="orange", alpha=0.5) +
    ggtitle("Raw Age (Unscaled)") + 
    theme_minimal()

  dp2 <- ggplot(clean, aes(x=Age)) + 
    geom_density(fill="purple", alpha=0.5) +
    ggtitle("Clean Age (Standardized)") + 
    theme_minimal()

  png("outputs/descriptive/scaling_impact.png", width=800, height=400)
  grid.arrange(dp1, dp2, ncol=2)
  dev.off()
  cat("Scaling impact comparison saved to outputs/descriptive/scaling_impact.png\n")
}

# STEP 9 — OVA Target Variable Distribution (Raw vs Clean)
# Shows how dropping outliers in features affects the target distribution
if (!is.na(target_col_raw) && !is.na(target_col_clean)) {
  op1 <- ggplot(raw, aes(x=.data[[target_col_raw]])) + 
    geom_histogram(fill="darkorange", color="black", alpha=0.7, bins=30) +
    ggtitle("Raw OVA (Right Skewed & Irregular)") + 
    xlab("Overall Rating (OVA)") +
    ylab("Count") +
    theme_minimal()

  op2 <- ggplot(clean, aes(x=.data[[target_col_clean]])) + 
    geom_histogram(fill="darkcyan", color="black", alpha=0.7, bins=30) +
    ggtitle("Clean OVA (Smoother Bell Curve)") + 
    xlab("Overall Rating (OVA)") +
    ylab("Count") +
    theme_minimal()

  png("outputs/descriptive/ova_distribution_comparison.png", width=800, height=400)
  grid.arrange(op1, op2, ncol=2)
  dev.off()
  cat("OVA distribution comparison saved to outputs/descriptive/ova_distribution_comparison.png\n")
}
