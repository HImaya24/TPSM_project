library(tidyverse)

# Step 1 — Load Dataset
df <- read.csv("data/fifa21_raw.csv")

# Explore
dim(df)
str(df)
summary(df)
head(df, 10)

# Step 2 — Check Issues
colSums(is.na(df))
sum(duplicated(df))
sapply(df, class)

# Step 3 — Remove Duplicates
df <- df %>% distinct()

# Step 4 & 5 — Clean Value and Wage Columns
clean_currency <- function(x) {
  if (is.na(x) || x == "") return(NA)
  x <- gsub("[€£]", "", x)
  multiplier <- 1
  if(grepl("M", x)) multiplier <- 1000000
  else if(grepl("K", x)) multiplier <- 1000
  x <- gsub("[MK]", "", x)
  return(as.numeric(x) * multiplier)
}

df$Value <- sapply(df$Value, clean_currency)
df$Wage <- sapply(df$Wage, clean_currency)

# Step 6 — Fix Height
convert_height <- function(h) {
  if (is.na(h) || h == "") return(NA)
  if (grepl("'", h)) {
    parts <- strsplit(h, "'")[[1]]
    feet <- as.numeric(parts[1])
    inches <- as.numeric(gsub('"', '', parts[2]))
    return(round((feet * 30.48) + (inches * 2.54), 1))
  } else {
    return(as.numeric(gsub("cm", "", h)))
  }
}
df$Height <- sapply(df$Height, convert_height)

# Fix Weight
convert_weight <- function(w) {
  if (is.na(w) || w == "") return(NA)
  if (grepl("lbs", w)) {
    return(round(as.numeric(gsub("lbs", "", w)) * 0.453592, 1))
  } else {
    return(as.numeric(gsub("kg", "", w)))
  }
}
df$Weight <- sapply(df$Weight, convert_weight)

# Step 7 — (Not applicable — columns not present)

# Step 8 — Handle Missing Values
df <- df %>%
  mutate(across(where(is.numeric),
                ~ ifelse(is.na(.), median(., na.rm=TRUE), .)))

getMode <- function(x) { names(sort(table(x), decreasing=TRUE))[1] }

df <- df %>%
  mutate(across(where(is.character),
                ~ ifelse(is.na(.), getMode(.), .)))

# Step 9 — Encode Categorical Columns
if("foot" %in% colnames(df)){
  df$foot <- ifelse(df$foot == "Right", 1, 0)
}

if("Positions" %in% colnames(df)){
  df$Positions <- as.numeric(as.factor(df$Positions))
}

if("Nationality" %in% colnames(df)){
  df$Nationality <- as.numeric(as.factor(df$Nationality))
}

# Fix naming for columns with spaces
if("Release.Clause" %in% colnames(df)) {
  # Already read as Release.Clause by read.csv
}

# Step 10 — Remove Outliers
remove_outliers <- function(df, col) {
  Q1 <- quantile(df[[col]], 0.25, na.rm=TRUE)
  Q3 <- quantile(df[[col]], 0.75, na.rm=TRUE)
  IQR <- Q3 - Q1
  df <- df %>% filter(df[[col]] >= Q1 - 1.5*IQR &
                      df[[col]] <= Q3 + 1.5*IQR)
  return(df)
}

df <- remove_outliers(df, "Wage")
df <- remove_outliers(df, "Value")

# Step 11 — Scale Numeric Columns (Exempting the Target Variable)
target_col_name <- if("X.OVA" %in% colnames(df)) "X.OVA" else if("OVA" %in% colnames(df)) "OVA" else NA
numeric_cols <- df %>% select(where(is.numeric)) %>% names()
cols_to_scale <- setdiff(numeric_cols, target_col_name)
df[cols_to_scale] <- scale(df[cols_to_scale])

# Step 12 — Save Final Dataset
write.csv(df, "data/fifa21_clean.csv", row.names = FALSE)

cat("Clean dataset saved! Rows:", nrow(df), "Columns:", ncol(df))
