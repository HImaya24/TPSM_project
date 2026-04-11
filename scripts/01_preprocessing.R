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

# Step 4 — Clean Value Column
df$Value <- gsub("[€£]", "", df$Value)
df$Value <- ifelse(grepl("M", df$Value),
                   as.numeric(gsub("M", "", df$Value)) * 1000000,
                   as.numeric(gsub("K", "", df$Value)) * 1000)

# Step 5 — Clean Wage Column
df$Wage <- gsub("[€£]", "", df$Wage)
df$Wage <- ifelse(grepl("K", df$Wage),
                  as.numeric(gsub("K", "", df$Wage)) * 1000,
                  as.numeric(df$Wage))

# Step 6 — Fix Height
df$Height <- ifelse(grepl("'", df$Height), {
  parts <- strsplit(df$Height, "'")
  feet <- as.numeric(sapply(parts, `[`, 1))
  inches <- as.numeric(gsub('"', '', sapply(parts, `[`, 2)))
  round((feet * 30.48) + (inches * 2.54), 1)
}, as.numeric(gsub("cm", "", df$Height)))

# Fix Weight
df$Weight <- ifelse(grepl("lbs", df$Weight),
                    round(as.numeric(gsub("lbs", "", df$Weight)) * 0.453592, 1),
                    as.numeric(gsub("kg", "", df$Weight)))

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
if("Preferred.Foot" %in% colnames(df)){
  df$Preferred.Foot <- ifelse(df$Preferred.Foot == "Right", 1, 0)
}

if("Position" %in% colnames(df)){
  df$Position <- as.numeric(as.factor(df$Position))
}

if("Nationality" %in% colnames(df)){
  df$Nationality <- as.numeric(as.factor(df$Nationality))
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

# Step 11 — Scale Numeric Columns
numeric_cols <- df %>% select(where(is.numeric)) %>% names()
df[numeric_cols] <- scale(df[numeric_cols])

# Step 12 — Save Final Dataset
write.csv(df, "data/fifa21_clean.csv", row.names = FALSE)

cat("Clean dataset saved! Rows:", nrow(df), "Columns:", ncol(df))
