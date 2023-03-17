# This file creates a training dataset
# Notably, we only use ads with coded party_all 
# IF all pd_ids by the respective page_id are the same party

library(data.table)
library(dplyr)

# Inputs
path_fb_140m_vars <- "../fb_2020/fb_2020_140m_adid_var1.csv.gz"
path_wmp_ent <- "../datasets/wmp_entity_files/Facebook/2020/wmp_fb_entities_v090622.csv"
# Outputs
path_training_data <- "data/facebook/118m_with_page_id_based_training_data.csv.gz"

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
names(test)[1] <- 'page_id'
df <- left_join(df, test, by = 'page_id')
df$party_all_usable <- df$party_all
df$party_all_usable[df$usable_party_all == F] <- NA
df$party_all_usable[df$party_all_usable == "NOTCODED"] <- NA

# Create the train-test split 
page_id_with_usable_party_all <- unique(df$page_id[is.na(df$party_all_usable) == F])
set.seed(123)
split <- sample(c('train', 'test'), length(page_id_with_usable_party_all), replace = T, prob = c(0.7, 0.3))
split <- data.frame(page_id = page_id_with_usable_party_all,
                    split = split)
df <- left_join(df, split, by = "page_id")

# Only keep ads in the train/test set
df <- df[is.na(df$split) == F,]

fwrite(df, path_training_data)
