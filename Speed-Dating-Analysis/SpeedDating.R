library(data.table)

# Importing the data
#You can choose the file for data 
df <- fread(file.choose())
df[,c("race_o","field_cd", "race", "goal", "date", "go_out", "career_c", "met", "dec", "length", "numdat_2", "gender", "condtn")] <- lapply(df[,c("race_o","field_cd", "race", "goal", "date", "go_out", "career_c", "met", "dec", "length", "numdat_2", "gender", "condtn")], factor)
#income, tuition, and mn_sat have NA values so they aren't numeric
nrow(df[complete.cases(df[ , c("income","tuition","mn_sat")]),])
nrow(df)


# Prediction of rate of people liking you with constant fields


#  rate that people decide they wanted to date you after date
# aggregate certain variables for each individual (avg's of attracitness... etc. that others gave after date)
desirableness <- aggregate(dec_o ~ iid, df, mean)

#fields that remain constant for an individual
constantAttributes <- df[, c("iid", "gender","age", "field_cd", "mn_sat", "tuition", "imprace", "income", "goal", "date", "go_out", "career_c", "match_es","sports", "tvsports", "exercise", "dining", "museums", "art", "hiking", "gaming", "clubbing", "reading", "tv", "theater", "movies", "concerts", "music", "shopping", "yoga")]
# remove duplicates so there is only one obs per person
constantAttributes <- constantAttributes[!duplicated(constantAttributes),]
# binding back together
constantAttributes <- merge(desirableness, constantAttributes, by="iid")

fullConstFit <- lm(dec_o ~ .-iid, data = na.omit(constantAttributes))

summary(fullConstFit)

# Prediction of rate of people liking you with constant fields and partner-rankings
#Choose variables from partners rating
attractive <- aggregate(attr_o ~ iid, df, mean)
sincere <- aggregate(sinc_o ~ iid, df, mean)
intelligent <- aggregate(intel_o ~ iid, df, mean)
fun <- aggregate(fun_o ~ iid, df, mean)
ambitious <- aggregate( amb_o ~ iid, df, mean) # this was assigned incorrectly.. do any of the analyses change in model prediction section?
attractive <- aggregate(attr_o ~ iid, df, mean)
likeability <- aggregate(like_o ~ iid,df, mean)

#augment variables
augmentedAttributes <- Reduce(function(x, y) merge(x, y, all=TRUE), list(attractive, sincere, intelligent, fun, ambitious, constantAttributes, desirableness))

#Removing attributes with a lot of NAs
augmentedAttributes <- augmentedAttributes[, -which(names(augmentedAttributes) %in% c("tuition", "mn_sat", "income")) ]

#Fit for the full model

fullAugmentedFit <- lm(dec_o ~ .-iid, data = na.omit(augmentedAttributes))
summary(fullAugmentedFit)

#Reduction of model
library(MASS)
step <- stepAIC(fullAugmentedFit, direction="both")
step$anova # display results

reducedAugmentedFit <- lm(dec_o ~ attr_o + fun_o + gender + age + career_c + tvsports + yoga, data = na.omit(augmentedAttributes))
summary(reducedAugmentedFit)

#F test
anova(reducedAugmentedFit, lm(dec_o ~ attr_o + fun_o + gender + age + tvsports + yoga, data = na.omit(augmentedAttributes)) )

#Checking associations
pairs(augmentedAttributes[,c("dec_o", "attr_o", "fun_o", "gender","age", "career_c", "tvsports","yoga")]) 
#Inclusion of interaction term and comparison in ANOVA
anova(reducedAugmentedFit, lm(dec_o ~ attr_o + fun_o + career_c + gender + age + tvsports + yoga + attr_o*fun_o, data = na.omit(augmentedAttributes)) )

reducedAugmentedFit<-lm(dec_o ~ attr_o + fun_o + career_c + gender + age + tvsports + yoga + attr_o*fun_o, data = na.omit(augmentedAttributes))
summary(reducedAugmentedFit)

summary(lm(dec_o ~ attr_o + fun_o + gender + age + career_c + tvsports + 
             concerts + music + yoga + attr_o:fun_o, data = na.omit(augmentedAttributes)))
par(mfrow = c(2,2))
plot(reducedAugmentedFit)
#next step is to maybe introduce interaction terms
fullAugmentedFit <- lm(dec_o ~ . + attr_o*fun_o -iid, data = na.omit(augmentedAttributes))
step <- stepAIC(fullAugmentedFit, direction="both")
step$anova # display results
anova(lm(dec_o ~ attr_o + fun_o + gender + age + career_c + tvsports + 
           concerts + music + yoga + attr_o:fun_o, data = na.omit(augmentedAttributes)), reducedAugmentedFit)
anova(lm(dec_o ~ attr_o + fun_o + gender + age + career_c + tvsports + yoga + attr_o:fun_o, data = na.omit(augmentedAttributes)), lm(dec_o ~ attr_o + fun_o + gender + career_c + tvsports + yoga + attr_o:fun_o, data = na.omit(augmentedAttributes)))
summary(lm(dec_o ~ attr_o + fun_o + gender + career_c + tvsports + yoga + attr_o:fun_o, data = na.omit(augmentedAttributes)))
final_reduced<-lm(dec_o ~ attr_o + fun_o + gender + career_c + tvsports + yoga + attr_o:fun_o, data = na.omit(augmentedAttributes))
summary(final_reduced)


#Prediction of likeability
#Prediction of being fun
funAugmentedFit <- lm(fun_o ~ .-iid -attr_o -sinc_o -intel_o -dec_o -amb_o, data = na.omit(augmentedAttributes))
summary(funAugmentedFit)

step <- stepAIC(funAugmentedFit, direction="both")
step$anova # display results

funReducedAugmentedFit <- lm(fun_o ~ gender + go_out + career_c + sports + exercise + dining + 
                                  gaming + clubbing + tv + concerts + music, data = na.omit(augmentedAttributes))
summary(funReducedAugmentedFit)

smallerAugmentedAttribs <- augmentedAttributes[, -which(names(augmentedAttributes) %in% c("attr_o", "sinc_o", "intel_o", "dec_o")) ]
# since we have a bunch of predictor variables I'm going to run a stepwaise AIC
funAUgmentedFit2 <- lm(fun_o ~ 1, data = na.omit(smallerAugmentedAttribs))
summary(funAUgmentedFit2)
step <- stepAIC(funAUgmentedFit2, direction="forward", scope = list(upper = funAugmentedFit, lower = funAUgmentedFit2))
step$anova # display results

funReducedAUgmentedFit2 <- lm(fun_o ~ go_out + exercise + clubbing + gender + field_cd + tv, data = na.omit(smallerAugmentedAttribs))
summary(funReducedAUgmentedFit2)

plot(augmentedAttributes$fun_o ~ augmentedAttributes$go_out)
plot(augmentedAttributes$fun_o ~ as.factor(augmentedAttributes$clubbing))
plot(augmentedAttributes$fun_o ~ augmentedAttributes$career_c)


#Rate people like you
desirableness <- aggregate(dec_o ~ iid, df, mean)

#Additional variables that might explain
attractive <- aggregate(attr_o ~ iid, df, mean)
sincere <- aggregate(sinc_o ~ iid, df, mean)
intelligent <- aggregate( intel_o ~ iid, df, mean)
fun <- aggregate( fun_o ~ iid, df, mean)
ambitious <- aggregate( amb_o ~ iid, df, mean)
personalityAttributes <- df[, c("iid", "attr3_1", "sinc3_1", "fun3_1", "intel3_1", "amb3_1")]

personalityAttributes <- personalityAttributes[!duplicated(personalityAttributes),]

personalityAttributes <- merge(desirableness, personalityAttributes,by="iid")

personalityAttributes <- Reduce(function(x, y) merge(x, y, all=TRUE), list(attractive, sincere, intelligent, fun, ambitious, personalityAttributes, desirableness  ))

attr_lm <- lm(personalityAttributes$attr_o ~ personalityAttributes$attr3_1)
summary(attr_lm)
plot(personalityAttributes$attr_o ~ personalityAttributes$attr3_1, xlab="Self-Reported", ylab = "Partner Rating", main = "Perceived Attractiveness")
abline(attr_lm)

library(ggplot2)
personalityAttributes$attr3_1 <- as.factor(personalityAttributes$attr3_1)
p <- ggplot(na.omit(personalityAttributes), aes(x=attr3_1, y=attr_o, fill=attr3_1)) + 
  geom_violin(trim=FALSE)
# plot with median and quartile
p  + geom_boxplot(width=0.1, fill="white") + labs(title="Perceived Attractiveness", x="Self-Reported", y="Partner Rating") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
# is it possible to add the lm to this plot?

sinc_lm <- lm(personalityAttributes$sinc_o ~ as.numeric(personalityAttributes$sinc3_1))
intel_lm <- lm(personalityAttributes$intel_o ~ as.numeric(personalityAttributes$intel3_1))
fun_lm <- lm(personalityAttributes$fun_o ~ as.numeric(personalityAttributes$fun3_1))
amb_lm <- lm(personalityAttributes$amb_o ~ as.numeric(personalityAttributes$amb3_1))
summary(sinc_lm)
summary(intel_lm)
summary(fun_lm)
summary(amb_lm)
plot(personalityAttributes$sinc_o ~ as.numeric(personalityAttributes$sinc3_1))
abline(sinc_lm)
plot(personalityAttributes$fun_o ~ as.numeric(personalityAttributes$fun3_1))
abline(fun_lm)
plot(personalityAttributes$intel_o ~ as.numeric(personalityAttributes$intel3_1))
abline(intel_lm)
plot(personalityAttributes$amb_o ~ as.numeric(personalityAttributes$amb3_1))
abline(amb_lm)

personalityAttributes$sinc3_1 <- as.factor(personalityAttributes$sinc3_1)
personalityAttributes$fun3_1 <- as.factor(personalityAttributes$fun3_1)
personalityAttributes$intel3_1 <- as.factor(personalityAttributes$intel3_1)
personalityAttributes$amb3_1 <- as.factor(personalityAttributes$amb3_1)
ggplot(na.omit(personalityAttributes), aes(x=sinc3_1, y=sinc_o, fill=sinc3_1)) + 
  geom_violin(trim=FALSE) + geom_boxplot(width=0.1, fill="white") + labs(title="Perceived Sincerity", x="Self-Reported", y="Partner Rating") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(na.omit(personalityAttributes), aes(x=fun3_1, y=fun_o, fill=fun3_1)) + 
  geom_violin(trim=FALSE) + geom_boxplot(width=0.1, fill="white") + labs(title="Perceived Fun", x="Self-Reported", y="Partner Rating") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(na.omit(personalityAttributes), aes(x=intel3_1, y=intel_o, fill=intel3_1)) + 
  geom_violin(trim=FALSE) + geom_boxplot(width=0.1, fill="white") + labs(title="Perceived Intelligence", x="Self-Reported", y="Partner Rating") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(na.omit(personalityAttributes), aes(x=amb3_1, y=sinc_o, fill=amb3_1)) + 
  geom_violin(trim=FALSE) + geom_boxplot(width=0.1, fill="white") + labs(title="Perceived Ambition", x="Self-Reported", y="Partner Rating") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))

library(dplyr)
sincerity <- df[,c("sinc_o", "sinc3_1")]
sincerity <- sincerity %>% count(sinc3_1, sinc_o)
# round very few decimal ratings to integers
sincerity$sinc_o <- as.integer(sincerity$sinc_o)
sincerity$sinc3_1 <- as.factor(sincerity$sinc3_1)
sincerity$sinc_o <- as.factor(sincerity$sinc_o)
fun <- df[,c("fun_o", "fun3_1")]
fun <- fun %>% count(fun3_1, fun_o)
fun$fun3_1 <- as.factor(fun$fun3_1)
fun$fun_o <- as.factor(as.integer(fun$fun_o))
ambition <- df[,c("amb_o", "amb3_1")]
ambition <- ambition %>% count(amb3_1, amb_o)
ambition$amb3_1 <- as.factor(ambition$amb3_1)
ambition$amb_o <- as.factor(as.integer(ambition$amb_o))
intelligence <- df[,c("intel_o", "intel3_1")]
intelligence <- intelligence %>% count(intel3_1, intel_o)
intelligence$intel3_1 <- as.factor(intelligence$intel3_1)
intelligence$intel_o <- as.factor(as.integer(intelligence$intel_o))
hotness <- df[,c("attr_o", "attr3_1")]
hotness <- hotness %>% count(attr3_1, attr_o)
hotness$attr3_1 <- as.factor(hotness$attr3_1)
hotness$attr_o <- as.factor(as.integer(hotness$attr_o))
ggplot(data = na.omit(hotness), aes(x=attr3_1, y=n, fill=attr_o)) +
  geom_bar(position=position_fill(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))+labs(title="Perceived Attractiveness", x="Self-Reported", y="Partner Rating")
ggplot(data = na.omit(intelligence), aes(x=intel3_1, y=n, fill=intel_o)) +
  geom_bar(position=position_fill(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(data = na.omit(ambition), aes(x=amb3_1, y=n, fill=amb_o)) +
  geom_bar(position=position_fill(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(data = na.omit(fun), aes(x=fun3_1, y=n, fill=fun_o)) +
  geom_bar(position=position_fill(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(data = na.omit(sincerity), aes(x=sinc3_1, y=n, fill=sinc_o)) +
  geom_bar(position=position_fill(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))

ggplot(data = na.omit(hotness), aes(x=attr3_1, y=n, fill=attr_o)) +
  geom_bar(position=position_stack(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))+labs(title="Perceived Attractiveness", x="Self-Reported", y="Partner Rating")
ggplot(data = na.omit(intelligence), aes(x=intel3_1, y=n, fill=intel_o)) +
  geom_bar(position=position_stack(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(data = na.omit(ambition), aes(x=amb3_1, y=n, fill=amb_o)) +
  geom_bar(position=position_stack(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(data = na.omit(fun), aes(x=fun3_1, y=n, fill=fun_o)) +
  geom_bar(position=position_stack(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
ggplot(data = na.omit(sincerity), aes(x=sinc3_1, y=n, fill=sinc_o)) +
  geom_bar(position=position_stack(reverse = TRUE), stat="identity") + scale_fill_brewer(palette="Spectral", direction = -1) + guides(fill= guide_legend(reverse = TRUE))
