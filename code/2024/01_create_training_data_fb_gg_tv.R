library(data.table)
library(dplyr)
library(haven)
setwd("~/Documents/GitHub/party_classifier/code/2024")

# Outputs
path_training_data <- "~/Documents/GitHub/party_classifier/data/2024/2024_fb_gg_tv_training_data.csv.gz"
path_training_data_balanced <- "~/Documents/GitHub/party_classifier/data/2024/2024_fb_gg_tv_training_data_balanced.csv.gz"
path_training_data_balanced_platform <- "~/Documents/GitHub/party_classifier/data/2024/2024_fb_gg_tv_training_data_balanced_platform.csv.gz"
path_training_data_no_tv <- "~/Documents/GitHub/party_classifier/data/2024/2024_fb_gg_tv_training_data_no_tv.csv.gz"
#----
# Google

# Inputs
path_google_vars <- "~/Downloads/WMP/Digital/ad_goals_followup/ad_goal_fu/google2024_set1_20260312.csv.gz"
path_wmp_ent <- "/Users/ykim03/Documents/datasets/wmp_entity_files/2024_google_070125.csv"

# Ad id, pd_id, page_id
df <- fread(path_google_vars, encoding = "UTF-8")
df <- df %>% filter(set3b==1) %>% # set3b only 
  select(ad_id, advertiser_id)
  
# Merge with WMP entities file using pd_id
ent <- fread(path_wmp_ent)
ent$party_all[ent$party_all == ""] <- "NOTCODED"
ent$party_all[!ent$party_all %in% c("DEM", "REP", "NOTCODED")] <- "OTHER"
ent <- ent %>% select(advertiser_id, party_all)
df <- left_join(df, ent, by = "advertiser_id") 
df <- df %>% filter(grepl("DEM|REP|NOTCODED", party_all)==TRUE) # include only DEM|REP|NOTCODED 

# Usable party_all
# Count how many different parties a page_id contains
# It's theoretically possible for two pd_ids of the same page_id to have different parties
# This could for example happen when someone runs ads on GQ, Vogue, etc.
test <- aggregate(df$party_all, by = list(df$advertiser_id), table)
test$usable_party_all <- unlist(lapply(test$x, length)) == 1
test <- test %>% select(-x)
names(test)[1] <- "advertiser_id"
df <- left_join(df, test, by = "advertiser_id")
df$party_all_usable <- df$party_all
df$party_all_usable[df$usable_party_all == F] <- NA
df$party_all_usable[df$party_all_usable == "NOTCODED"] <- NA

df_g <- df
df_g$platform <- "Google"

#----

# Inputs
# fb_2024
path_fb_vars <- "~/Downloads/WMP/Digital/ad_goals_followup/ad_goal_fu/meta2024_set1_20260312.csv.gz"
path_wmp_ent <- "/Users/ykim03/Documents/datasets/wmp_entity_files/wmp_fb_2024_entities_wos_v062525.csv"

# Ad id, pd_id, page_id
df <- fread(path_fb_vars, encoding = "UTF-8")
df <- df %>% filter(set3b==1) %>% select(ad_id, pd_id, page_id) # limiting to set3b
# Merge with WMP entities file using pd_id
ent <- fread(path_wmp_ent)
ent <- ent %>% select(pd_id, party_cdptyonly)
ent$party_cdptyonly[ent$party_cdptyonly == ""] <- "NOTCODED"
df <- left_join(df, ent, by = "pd_id")
df <- df %>% filter(grepl("DEM|REP|NOTCODED", party_cdptyonly)==TRUE) # include only DEM|REP|NOTCODED 

# Usable party_all
# Then count how many different parties a page_id contains
# It's theoretically possible for two pd_ids of the same page_id to have different parties
# This could for example happen when someone runs ads on GQ, Vogue, etc.
test <- aggregate(df$party_cdptyonly, by = list(df$page_id), table)
test$usable_party_all <- unlist(lapply(test$x, length)) == 1
test <- test %>% select(-x)
names(test)[1] <- "page_id"
df <- left_join(df, test, by = "page_id")
df$party_all_usable <- df$party_cdptyonly
df$party_all_usable[df$usable_party_all == F] <- NA
df$party_all_usable[df$party_all_usable == "NOTCODED"] <- NA

df_f <- df
df_f$platform <- "Facebook"

# TV
path_tv_vars <- "/Users/ykim03/Documents/datasets/entity_linking_2024/tv24_cmag_meta_011426.csv.gz"
df <- fread(path_tv_vars, encoding = "UTF-8")
df <- df %>% filter(airdate_last >= "2024-09-03") %>%
  filter(grepl("US HOUSE|US SENATE|PRESIDENT", race)==TRUE) %>%
  filter(grepl("DEMOCRAT|REPUBLICAN", affiliation)==TRUE)

cand_wmp <- read.csv('/Users/ykim03/Documents/datasets/candidates/wmpcand_121724_wmpid.csv')
cand_wmp_st <- cand_wmp_merge %>% 
  filter(genelect_cd==1 & grepl("DEM|REP", cand_party_affiliation)==TRUE) %>% select(-cand_id)

df <- df %>% left_join(cand_wmp_st, by="wmpid") %>%
  select(alt, genelect_cd, sponsor_CMAG, cand_party_affiliation, affiliation)

# prioritizing WMP data for party (TV is basically for training only)
df <- df %>%
  mutate(
    party_all = case_when(
      !is.na(cand_party_affiliation) ~ cand_party_affiliation,
      affiliation %in% c("DEMOCRAT") ~ "DEM",
      affiliation %in% c("REPUBLICAN") ~ "REP",
      TRUE ~ NA_character_
    )
  )
df %>% group_by(party_all) %>% count()

# since it's unique id from TV (TV )
df$party_all_usable <- df$party_all

df_tv <- df
df_tv$platform <- "TV"


# Make advertiser_id for FB pd_id, advertiser_id for TV alt, then combine all
df_f$advertiser_id <- df_f$pd_id
df_f$party_all <- df_f$party_cdptyonly
df_f <- df_f %>% select(-c(pd_id, page_id, party_cdptyonly))

df_tv$ad_id <- df_tv$alt
df_tv$advertiser_id <- df_tv$sponsor_CMAG
df_tv <- df_tv %>% select(-c(cand_party_affiliation))

df <- bind_rows(df_f, df_g, df_tv)
df <- df %>% select(ad_id,  advertiser_id, party_all_usable, platform, party_all)
df %>% group_by(platform, party_all_usable) %>% count()
df %>% group_by(platform, party_all_usable, party_all) %>% count()

#----

# Create the train-test split
# Usable advertisers
df <- as.data.frame(df)
df_usable <- df[(is.na(df$party_all_usable) == F) & (is.na(df$advertiser_id) == F),]
# Count how many ads there are per advertiser
df_advertiser_id_counts <- df_usable %>% group_by(advertiser_id) %>% count()
df_advertiser_id_counts <- df_usable %>% select(advertiser_id, party_all) %>% 
  right_join(df_advertiser_id_counts, "advertiser_id") 

# Create the test set
# Sample 10 D and R advertisers who have between 500 and 1000 ads
df_advertiser_id_counts_500_1000 <- df_advertiser_id_counts[(df_advertiser_id_counts$n > 500) & (df_advertiser_id_counts$n < 1000),]
set.seed(123)
test_Dem <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "DEM"], 10)
set.seed(123)
test_Rep <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "REP"], 10)

df_usable$split <- "train"
df_usable$split[df_usable$advertiser_id %in% c(test_Dem, test_Rep)] <- "test"
table(df_usable$split)

fwrite(df_usable, path_training_data)

# Create balanced test set
set.seed(123)
# Find smallest class size
n_min <- df_usable %>%
  count(party_all) %>%
  summarise(min_n = min(n)) %>%
  pull(min_n)

# Sample equally
df_balanced <- df_usable %>%
  group_by(party_all) %>%
  slice_sample(n = n_min) %>%
  ungroup()

# Check
df_balanced %>% count(party_all)

# Create the test set
# Sample 10 D and R advertisers who have between 500 and 1000 ads
df_advertiser_id_counts <- df_balanced %>% group_by(advertiser_id) %>% count()
df_advertiser_id_counts <- df_balanced %>% select(advertiser_id, party_all) %>% 
  right_join(df_advertiser_id_counts, "advertiser_id") 

df_advertiser_id_counts_500_1000 <- df_advertiser_id_counts[(df_advertiser_id_counts$n > 500) & (df_advertiser_id_counts$n < 1000),]
set.seed(123)
test_Dem <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "DEM"], 10)
set.seed(123)
test_Rep <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "REP"], 10)

df_balanced$split <- "train"
df_balanced$split[df_balanced$advertiser_id %in% c(test_Dem, test_Rep)] <- "test"
table(df_balanced$split)

fwrite(df_balanced, path_training_data_balanced)

# Create platform-based test set
set.seed(123)

df_platform_balanced <- df_usable %>%
  group_by(platform, party_all) %>%
  mutate(n_group = n()) %>%
  ungroup() %>%
  group_by(platform) %>%
  mutate(n_min = min(n_group)) %>%
  ungroup() %>%
  group_by(platform, party_all) %>%
  slice_sample(n = unique(n_min)) %>%
  ungroup() %>%
  select(-n_group, -n_min)

# Check
df_platform_balanced %>%
  count(platform, party_all)

# Create the test set
# Sample 10 D and R advertisers who have between 500 and 1000 ads
df_advertiser_id_counts <- df_platform_balanced %>% group_by(advertiser_id) %>% count()
df_advertiser_id_counts <- df_platform_balanced %>% select(advertiser_id, party_all) %>% 
  right_join(df_advertiser_id_counts, "advertiser_id") 

df_advertiser_id_counts_500_1000 <- df_advertiser_id_counts[(df_advertiser_id_counts$n > 500) & (df_advertiser_id_counts$n < 1000),]
set.seed(123)
test_Dem <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "DEM"], 10)
set.seed(123)
test_Rep <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "REP"], 10)

df_platform_balanced$split <- "train"
df_platform_balanced$split[df_platform_balanced$advertiser_id %in% c(test_Dem, test_Rep)] <- "test"
table(df_platform_balanced$split)

fwrite(df_platform_balanced, path_training_data_balanced_platform)

## Robustness check without TV data
# Create the train-test split
# Usable advertisers
df_usable_no_tv <- df_usable %>% filter(platform!="TV")
table(df_usable_no_tv$platform)

# Count how many ads there are per advertiser
df_advertiser_id_counts <- df_usable_no_tv %>% group_by(advertiser_id) %>% count()
df_advertiser_id_counts <- df_usable_no_tv %>% select(advertiser_id, party_all) %>% 
  right_join(df_advertiser_id_counts, "advertiser_id") 

# Create the test set
# Sample 10 D and R advertisers who have between 500 and 1000 ads
df_advertiser_id_counts_500_1000 <- df_advertiser_id_counts[(df_advertiser_id_counts$n > 500) & (df_advertiser_id_counts$n < 1000),]
set.seed(123)
test_Dem <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "DEM"], 10)
set.seed(123)
test_Rep <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "REP"], 10)

df_usable_no_tv$split <- "train"
df_usable_no_tv$split[df_usable_no_tv$advertiser_id %in% c(test_Dem, test_Rep)] <- "test"
table(df_usable_no_tv$split)

fwrite(df_usable_no_tv, path_training_data_no_tv)
