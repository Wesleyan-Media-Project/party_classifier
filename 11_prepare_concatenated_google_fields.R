# Concatenate all fields for Google into one
# Field order is somewhat random since the use is a bag of words model where it doesn't matter

library(dplyr)
library(tidyr)
library(data.table)

load("datasets/google/all_ads.rdata")

all <- unnest(all, ad_id)
all <- unnest(all, advertiser_id)
advertiser_df <- all %>% select(ad_id, advertiser_id)
advertiser_df <- advertiser_df[!duplicated(advertiser_df$ad_id),]
all <- select(all, ad_id, text, advertiser_name, scraped_ad_url)

# a few ads have 2 advertiser names and urls, but those are mostly just garbage
# so only use the first one
all$advertiser_name <- unlist(lapply(all$advertiser_name, function(x){x[1]}))
all$scraped_ad_url <- unlist(lapply(all$scraped_ad_url, function(x){x[1]}))

all <- pivot_longer(all, -ad_id)

all <- aggregate(all$value, by = list(all$ad_id), paste, collapse = " ")
names(all) <- c("ad_id", "text")

all <- left_join(all, advertiser_df, by = "ad_id")

fwrite(all, "data/google/all_fields_concatenated.csv")
