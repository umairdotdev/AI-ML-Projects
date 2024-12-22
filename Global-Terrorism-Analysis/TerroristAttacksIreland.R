rm(list = ls())

#required libraries

load_lb <- function()
{
  suppressPackageStartupMessages(require(Matrix))
  suppressPackageStartupMessages(require(ggplot2))
  suppressPackageStartupMessages(require(data.table))
  suppressPackageStartupMessages(require(treemap))
  suppressPackageStartupMessages(require(highcharter))
  suppressPackageStartupMessages(library(doMC))
  registerDoMC(cores = 8)
  suppressPackageStartupMessages(library(readxl))
  suppressPackageStartupMessages(library(tidyr))
  suppressPackageStartupMessages(library(dplyr))
  suppressPackageStartupMessages(library(caret))
  
}

load_lb()

# Load the files
df <- fread(file.choose())     # Choose a file

glimpse(df)
## 170,350 X 135 dimension  

summary(df$iyear)                 # 1970 to 2016 

## country wise killings 

df %>% 
  select (country_txt, nkill) %>% 
  filter(nkill > 0) %>% 
  group_by(country_txt) %>% 
  summarise(count = sum(nkill)) %>% 
  filter(count > 5000) %>% 
  ggplot(aes(x = reorder(country_txt,count), y = count, fill = count)) +
  geom_bar(stat = "identity", show.legend = FALSE) + 
  labs(title = "Number of kills by country", x = "Country", y = "Number of kills")+
  geom_text(aes(label = count), hjust = -0.2) +
  coord_flip() +
  theme_minimal()


## year wise killings 

df %>% 
  select (iyear, nkill) %>% 
  filter(nkill > 0) %>% 
  group_by(iyear) %>% 
  summarise(count = sum(nkill)) %>% 
  ggplot(aes(x = iyear, y = count, fill = count)) +
  geom_bar(stat = "identity", show.legend = FALSE) + 
  labs(title = "Number of kills in each year", x = "Year", y = "Number of kills")+
  #geom_text(aes(label = count), vjust = -0.2) +
  theme_minimal()

## 2012 to 2016 have maximum no. of kills




## Regionwise killings in world

df %>% 
  select (region_txt, nkill) %>% 
  filter(nkill > 0) %>% 
  group_by(region_txt) %>% 
  summarise(count = sum(nkill)) %>% 
  filter(count > 5000) %>% 
  ggplot(aes(x = reorder(region_txt,count), y = count, fill = count)) +
  geom_bar(stat = "identity", show.legend = FALSE) + 
  labs(title = "Number of kills by regions", x = "Region names", y = "Number of deaths")+
  geom_text(aes(label = count), hjust = -0.2) +
  coord_flip() +
  theme_minimal()



## Lets check the trend in Ireland

df %>% 
  select (iyear, nkill, country_txt) %>% 
  filter(nkill > 0,country_txt == "Ireland") %>% 
  group_by(iyear) %>% 
  summarise(count = sum(nkill)) %>% 
  ggplot(aes(x = reorder(iyear,count), y = count, fill = count)) +
  geom_bar(stat = "identity", show.legend = FALSE) + 
  labs(title = "Number of kills in each year in Ireland", x = "Year", y = "Number of kills")+
  geom_text(aes(label = count), hjust = -0.2) +
  coord_flip() +
  theme_minimal()

df %>% 
  select (iyear, nkill,everything()) %>% 
  filter(nkill > 0,country_txt == "Ireland", !is.na(latitude)) -> df_ireland

## no of casualities, clustered on map

library(leaflet)
leaflet(df_ireland) %>% 
  addTiles() %>% 
  addCircleMarkers(clusterOptions = markerClusterOptions(),
                   lat = df_ireland$latitude, lng = df_ireland$longitude)


## no of deaths by cities in Ireland

df_ireland %>% 
  group_by(provstate) %>% 
  summarise(count = sum(nkill)) %>% 
  ggplot(aes(x = reorder(provstate,count), y = count, fill = count)) + 
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "Number of deaths by Cities", x = "Cities", y = "Deaths") +
  geom_text(aes(label = count), hjust = -0.2) +
  coord_flip() +
  theme_light()

# Dublin has maximum deaths


# lets check for the attack_type for each incident in Ireland

df_ireland %>% 
  group_by(attacktype1_txt) %>% 
  summarise(count = sum(nkill)) %>% 
  ggplot(aes(x = reorder(attacktype1_txt,count), y = count, fill = count)) + 
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "Killings by attack type", x = "Attack type", y = "Deaths") +
  geom_text(aes(label = count), hjust = -0.2) +
  coord_flip() +
  theme_light()
## we do have unarmed assault incidents as well

# Did terrorists get success always?
df_ireland %>% 
  group_by(success) %>% 
  summarise(count = n()) %>% 
  ggplot(aes(x = reorder(success,count), y = count/sum(count), fill = success)) + 
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "Success ratio of terrorists attack", x = "Success or not", y = "Deaths") +
  geom_text(aes(label = round(count*100/sum(count),1)), vjust = -0.2) +
  theme_minimal()
## 96.9% of the attacks were successful

# Who were the targets?
head(df_ireland)

df_ireland %>% 
  group_by(targtype1_txt) %>% 
  summarise(count = sum(nkill)) %>% 
  ggplot(aes(x = reorder(targtype1_txt,count), y = count, fill = count)) + 
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "Killings by target type", x = "Target type", y = "Deaths") +
  geom_text(aes(label = count), hjust = -0.2) +
  coord_flip() +
  theme_light()

## The target has always been innocent people


