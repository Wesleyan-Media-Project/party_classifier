# This file creates a training dataset
# Notably, we only use ads with coded party_all 
# if all pd_ids by the respective page_id are the same party

library(data.table)
library(dplyr)

# Need Jielu's data for pd_id, page_id
d118 <- fread("C:/Users/neuma/Downloads/fb_2020_adid_06092022.csv")
d118 <- d118 %>% select(ad_id, pd_id, page_id)
# My version has better text though
df <- fread('../data/facebook/118m_all_ads.csv', encoding = "UTF-8")
# So combine them
df <- left_join(df, d118, by = "ad_id")
# Then merge both with entities file
ent <- fread("../../data/wmp_entities/wmp_fb_entities_v051822.csv")
ent <- ent %>% select(pd_id, party_all)
ent <- ent %>% filter(party_all != "MISSING")
df <- left_join(df, ent, by = "pd_id")

# Usable party_all
df$party_all[is.na(df$party_all)] <- "NOTCODED"
test <- aggregate(df$party_all, by = list(df$page_id), table)
test$usable_party_all <- unlist(lapply(test$x, length)) == 1
#test2 <- aggregate(df$pd_id, list(df$page_id, df$party_all), unique)
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

fwrite(df, "../data/facebook/118m_with_page_id_based_training_data.csv")
