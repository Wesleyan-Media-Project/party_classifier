# This file creates a training dataset
# Notably, we only use ads with coded party_all
# IF all pd_ids by the respective page_id are the same party

library(data.table)
library(dplyr)
library(haven)


# Outputs
path_training_data <- "data/2020_fb_and_google_with_page_id_based_training_data.csv.gz"

#----
# Google

# Inputs
path_google_2020_vars <- "../google_2020/google_2020_adid_var1.csv.gz"
path_wmp_ent <- "../datasets/wmp_entity_files/Google/2020/wmp_google_entities_v040521.dta"

# Ad id, pd_id, page_id
df <- fread(path_google_2020_vars, encoding = "UTF-8")
df <- df %>% select(ad_id, advertiser_id)
# Merge with WMP entities file using pd_id
ent <- read_dta(path_wmp_ent)
ent$party_all <- paste0(ent$hse_party, ent$sen_party, ent$pres_party, ent$cmte_party)
ent$party_all[ent$party_all == "DEMDEM"] <- "DEM"
ent$party_all[ent$party_all == "REPREP"] <- "REP"
ent$party_all[ent$party_all == ""] <- "NOTCODED"
ent$party_all[!ent$party_all %in% c("DEM", "REP", "NOTCODED")] <- "OTHER"
ent <- ent %>% select(advertiser_id, party_all)
df <- left_join(df, ent, by = "advertiser_id")

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
# fb_2020_140m_adid_var1.csv.gz is an output from repo fb_2020
path_fb_140m_vars <- "../fb_2020/fb_2020_140m_adid_var1.csv.gz"
path_wmp_ent <- "../datasets/wmp_entity_files/Facebook/2020/wmp_fb_entities_v090622.csv"

# Ad id, pd_id, page_id
df <- fread(path_fb_140m_vars, encoding = "UTF-8")
df <- df %>% select(ad_id, pd_id, page_id)
# Merge with WMP entities file using pd_id
ent <- fread(path_wmp_ent)
ent <- ent %>% select(pd_id, party_all)
ent <- ent %>% filter(party_all != "MISSING")
df <- left_join(df, ent, by = "pd_id")

# Usable party_all
# First of all, check which don't have party coded
df$party_all[is.na(df$party_all)] <- "NOTCODED"
# Then count how many different parties a page_id contains
# It's theoretically possible for two pd_ids of the same page_id to have different parties
# This could for example happen when someone runs ads on GQ, Vogue, etc.
test <- aggregate(df$party_all, by = list(df$page_id), table)
test$usable_party_all <- unlist(lapply(test$x, length)) == 1
test <- test %>% select(-x)
names(test)[1] <- "page_id"
df <- left_join(df, test, by = "page_id")
df$party_all_usable <- df$party_all
df$party_all_usable[df$usable_party_all == F] <- NA
df$party_all_usable[df$party_all_usable == "NOTCODED"] <- NA

df_f <- df
df_f$platform <- "Facebook"

# Make advertiser_id for FB pd_id, then combine FB and Google
df_f$advertiser_id <- df_f$pd_id
df_f <- df_f %>% select(-c(pd_id, page_id))
df <- bind_rows(df_f, df_g)

#----

# Create the train-test split
# Usable advertisers
df <- as.data.frame(df)
df_usable <- df[(is.na(df$party_all_usable) == F) & (is.na(df$advertiser_id) == F),]
# Count how many ads there are per advertiser
df_advertiser_id_counts <- df_usable %>% group_by(advertiser_id) %>% count()
df_advertiser_id_counts <- df_usable %>% select(advertiser_id, party_all) %>% right_join(df_advertiser_id_counts, "advertiser_id")

# Create the test set
# Sample 10 D and R advertisers who have between 500 and 1000 ads
df_advertiser_id_counts_500_1000 <- df_advertiser_id_counts[(df_advertiser_id_counts$n > 500) & (df_advertiser_id_counts$n < 1000),]
set.seed(123)
test_Dem <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "DEM"], 10)
set.seed(123)
test_Rep <- sample(df_advertiser_id_counts_500_1000$advertiser_id[df_advertiser_id_counts_500_1000$party_all == "REP"], 10)

df_usable$split <- "train"
df_usable$split[df_usable$advertiser_id %in% c(test_Dem, test_Rep)] <- "test"

fwrite(df_usable, path_training_data)
