# Behavioral and single-trial EEG analysis of the data of 
# "Common Neural Choice Signals reflect Accumulated Evidence, not Confidence"
# authored by Kobe Desender et al.,

# Note, during coding the terms CPP and P3 are used interchangeably (whereas the paper consistently refers to the CPP)

#### 1. Load Data ####
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) #set wd

rm(list=ls()) #empties the environment
library(DEoptim) #DEoptim_2.2-8
library(lmerTest) #lmerTest_3.1-3, lme4_1.1-35.5 
library(effects) #effects_4.2-2  
library(tidyverse) #tidyverse_2.0.0  
library(BayesFactor) #BayesFactor_0.9.12-4.7

#set some ggplot defaults
theme_set(theme_classic())
theme_update(axis.title.y = element_text(colour="black", size=15),
             axis.title.x = element_text(colour="black", size=15),
             axis.text.y = element_text(margin = margin(l = 6, r = 6),colour="black", size=14),
             axis.text.x = element_text(margin = margin(t=4,b=4),colour="black", size=14),
             axis.ticks.length=unit(-0.15,"cm"),
             axis.ticks = element_line(linewidth=1,color="black"),
             axis.line = element_line(linewidth=1))

error.bar <- function(x, y, upper, lower=upper, length=0,...){
  if(length(x) != length(y) | length(y) !=length(lower) | length(lower) != length(upper))
    stop("vectors must be same length")
  arrows(x,y+upper, x, y-lower, angle=90, code=3, length=length, ...)
}

#load the behavioral data which contains single-trial EEG amplitudes
df <- readbulk::read_bulk(paste0(getwd(), "/Experiment/eeg data/eeg data for Herregods model/"),extension="stim.csv",verbose=F) #stim-locked
df_resp <- readbulk::read_bulk(paste0(getwd(), "/Experiment/eeg data/eeg data for Herregods model/"),extension="resp.csv",verbose=F) #resp-locked
df_cj <- readbulk::read_bulk(paste0(getwd(), "/Experiment/eeg data/eeg data for Herregods model/"),extension="cj.csv",verbose=F) #cj-locked

#Exclude Sub 28, who performed terribly on easy trials
df <- subset(df,sub!=28)
df_resp <- subset(df_resp,sub!=28)
df_cj <- subset(df_cj,sub!=28)

#Add the PE amplitude to the stim locked data (df)
df$average_PE_amplitude <- NA
df$rtcj_duplicate <- NA
ctr <- 1
for(i in 1:length(df$cj)){
  if(df$rt[i] == df_resp$rt[ctr]){
    df$average_PE_amplitude[i] <- df_resp$average_PE_amplitude[ctr]
    df$rtcj_duplicate[i] <- df_resp$rtcj[ctr] #for a double check
    ctr <- ctr+1
  } 
}
#Double check that the matching is done correctly
cor.test(df$rtcj,df$rtcj_duplicate,na.rm=T) #should be 1

#Add the frontal cj-locked amplitude to the stim&resp locked data (df)
df$frontal_amplitude <- NA
df$rtcj_duplicate <- NA
ctr <- 1
for(i in 1:length(df$cj)){
  if(df$rt[i] == df_cj$rt[ctr]){
    df$frontal_amplitude[i] <- df_cj$average_fr_amplitude[ctr]
    df$rtcj_duplicate[i] <- df_cj$rtcj[ctr] #for a double check
    ctr <- ctr+1
  } 
}
#Double check
cor.test(df$rtcj,df$rtcj_duplicate,na.rm=T) #should be 1

# Exclude extremely slow trials (note, some of these are already excluded during preprocessing)
df <- subset(df, rt < 3)
df <- subset(df, rtcj < 3000)

#Scale P3/PE for the lme's
df$P3_amplitude <- as.vector(scale(df$average_P3_amplitude))
df$PE_amplitude <- as.vector(scale(df$average_PE_amplitude))
df$fr_amplitude <- as.vector(scale(df$frontal_amplitude))
df$fr_P3_amplitude <- as.vector(scale(df$frontal_P3_amplitude))

#create some convenience variables
df$cj_cat <- as.factor(df$cj)
df$cond <- ifelse(df$condition=="easy" | df$condition=="positivefb","positive","negative")
df$mantype <- ifelse(df$condition=="easy" | df$condition == "hard", "traindiff","fb")
df$cond <- as.factor(df$cond) 
df$mantype <- as.factor(df$mantype) 
df$difficulty <- as.factor(df$difficulty) 
df$sub_cat <- as.factor(df$sub)
#variable that tracks subs
subs <- unique(sort(df$sub));N<-length(subs)

#Look at the data
par(mfrow=c(2,2))
at_chance <- matrix(NA,nrow=N,ncol=2)
for(i in 1:N){
  #main exp
  tempDat <- subset(df,sub==subs[i])
  tempDat$block <- rep(1:10,each=100)[1:dim(tempDat)[1]]
  acc_block <- with(tempDat,aggregate(cor,by=list(block=block),mean))
  #bias_block <- with(tempDat,aggregate(resp,by=list(block=block),mean))
  plot(acc_block,ylab="Acc (.) and bias (x)",frame=F,ylim=c(0,1),pch=19);abline(h=.5,lty=2,col="grey")
  #points(bias_block,pch=4,col="grey")
  plot(tempDat$rt,frame=F,main=paste('sub',subs[i]),ylab="RT",ylim=c(0,3))
  plot(tempDat$cj,frame=F,ylim=c(1,6),ylab="conf")
  plot(tempDat$rtcj,frame=F,ylab="RT_conf")
  #compare to chance level
  at_chance[i,] <- c(subs[i],chisq.test(table(tempDat$cor))$p.value)
}
# dev.off()
print(paste('There are', sum(at_chance[,2]>.05)," people at chance level")) #no one at chance level

#Participants 4 only pressed "cj=2" in the last 2 blocks, exclude those two blocks
df$outlier <- ifelse(df$sub==4 & df$X > 482,1,0)
df <- subset(df,outlier==0)

temp <- table(df$sub,df$cj)
par(mfrow=c(1,1))
plot(colMeans(temp),type='n',frame=F,xlab="Confidence",ylab="Number of trials",ylim=range(temp))
for(i in 1:N) lines(x=jitter(1:6,.3), temp[i,],col=rgb(.5,.5,.5,.1), bg=rgb(0,0,0,.25),type='b',pch=21)
# Save the current plot parameters
original_par <- par(no.readonly = TRUE)
par(fig = c(0.15, 0.5, 0.5, 0.95), new = TRUE)
plot(x=1:3,colMeans(temp[,1:3]),type='n',frame=F,xlab="Confidence",ylab="Number of trials",ylim=c(0,30),xlim=c(.75,3.25))
for(i in 1:N) lines(x=jitter(1:3,.3), temp[i,1:3],col=rgb(.5,.5,.5,.1), bg=rgb(0,0,0,.25),type='b',pch=21)
# Restore the original plot parameters
par(original_par)
colMeans(temp)

#Based on the above, concatenate cj 1-2 so that we roughly M=50 trials or more
df$cj[df$cj < 2] <- 2
colMeans(table(df$sub,df$cj))
df$cj <- 1 + (df$cj - min(df$cj)) * 4 / (max(df$cj) - min(df$cj)) #make the range between 1-5
df$cj_cat <- as.factor(df$cj)
colMeans(table(df$sub,df$cj))


#### 2. Behavioral Results ####
#2.1 Accuracy
temp <- with(df, aggregate(cor,by=list(sub=sub,cond=cond,difficulty=difficulty),mean))
temp$difficulty <- factor(temp$difficulty, levels = c("easy", "medium", "hard"))
temp_summary <- temp %>%
  group_by(cond, difficulty) %>%
  summarize(
    mean_x = mean(x),
    se_x = sd(x) / sqrt(n()),
    .groups = 'drop'
  )
ggplot(temp_summary, aes(x = difficulty, y = mean_x, color = cond, group = cond)) +
  geom_jitter(data = temp, aes(x = difficulty, y = x, color = cond), width = 0.2, alpha = 0.5) +
  geom_line(aes(linetype=cond, color=cond),linewidth=1.5) +
  geom_point(size=4) +
  geom_errorbar(aes(ymin = mean_x - se_x, ymax = mean_x + se_x), width = 0,linewidth=1.5) +
  labs(y="Accuracy",x="Difficulty", color="Prior condition", shape="Prior condition",linetype="Prior condition") 

#fit models with increasing random slopes complexity and compare these
fit <- glmer(cor~mantype*cond*difficulty+(1|sub),df,family=binomial)
fit_a <- glmer(cor~mantype*cond*difficulty+(mantype|sub),df,family=binomial)
fit_b <- glmer(cor~mantype*cond*difficulty+(cond|sub),df,family=binomial) #doesn't converge
fit_c <- glmer(cor~mantype*cond*difficulty+(difficulty|sub),df,family=binomial) #singular fit
anova(fit,fit_a) #p<.05 so continue with most complex model
car::Anova(fit_a)
generalTestBF(cor~mantype*cond*difficulty + difficulty:sub_cat + sub_cat, data = df,whichRandom = "sub_cat", neverExclude = "sub_cat",whichModels = "top") #compute BFs for n.s. effects
car::vif(fit_a)
data.frame(effect('difficulty',fit_a)) #inspect the fits

#2.2. RTs
temp <- with(subset(df,cor==1), aggregate(rt,by=list(sub=sub,cond=cond,difficulty=difficulty),mean))
temp$difficulty <- factor(temp$difficulty, levels = c("easy", "medium", "hard"))
temp_summary <- temp %>%
  group_by(cond, difficulty) %>%
  summarize(
    mean_x = mean(x),
    se_x = sd(x) / sqrt(n()),
    .groups = 'drop'
  )
ggplot(temp_summary, aes(x = difficulty, y = mean_x, color = cond, group = cond)) +
  geom_jitter(data = temp, aes(x = difficulty, y = x, color = cond), width = 0.2, alpha = 0.5) +
  geom_line(aes(linetype=cond, color=cond),linewidth=1.5) +
  geom_point(size=4) +
  geom_errorbar(aes(ymin = mean_x - se_x, ymax = mean_x + se_x), width = 0,linewidth=1.5) +
  labs(y="RTs (s)",x="Difficulty", color="Prior condition", shape="Prior condition",linetype="Prior condition") 

fit <- lmer(rt~mantype*cond*difficulty+(1|sub),subset(df,cor==1))
fit_a <- lmer(rt~mantype*cond*difficulty+(mantype|sub),subset(df,cor==1))
fit_b <- lmer(rt~mantype*cond*difficulty+(cond|sub),subset(df,cor==1))
fit_c <- lmer(rt~mantype*cond*difficulty+(difficulty|sub),subset(df,cor==1))
anova(fit,fit_a) #p<.001
anova(fit,fit_b) #p<.001
anova(fit,fit_c) #p<.001
fit2 <- lmer(rt~mantype*cond*difficulty+(mantype+cond+difficulty|sub),subset(df,cor==1))
fit2_a <- lmer(rt~mantype*cond*difficulty+(mantype*cond+difficulty|sub),subset(df,cor==1)) #doesn't work
fit2_b <- lmer(rt~mantype*cond*difficulty+(mantype+cond*difficulty|sub),subset(df,cor==1)) #doesn't work
fit2_c <- lmer(rt~mantype*cond*difficulty+(mantype*difficulty+cond|sub),subset(df,cor==1)) #doesn't work
anova(fit2)
generalTestBF(rt~mantype*cond*difficulty + difficulty:sub_cat + cond:sub_cat + mantype:sub_cat + sub_cat, data = subset(df,cor==1),whichRandom = "sub_cat", neverExclude = "sub_cat",whichModels = "top") #compute BFs for n.s. effects
car::vif(fit2)
data.frame(effect('difficulty',fit2))
data.frame(effects::effect('mantype:cond',fit2))
pairs(emmeans::emmeans(fit2, ~ mantype * cond),adjust="none")

#2.3. Confidence
temp <- with(subset(df,cor==1), aggregate(cj,by=list(sub=sub,cond=cond,difficulty=difficulty),mean))
temp$difficulty <- factor(temp$difficulty, levels = c("easy", "medium", "hard"))
temp_summary <- temp %>%
  group_by(cond, difficulty) %>%
  summarize(
    mean_x = mean(x),
    se_x = sd(x) / sqrt(n()),
    .groups = 'drop'
  )
ggplot(temp_summary, aes(x = difficulty, y = mean_x, color = cond, group = cond)) +
  geom_jitter(data = temp, aes(x = difficulty, y = x, color = cond), width = 0.2, alpha = 0.5) +
  geom_line(aes(linetype=cond, color=cond),linewidth=1.5) +
  geom_point(size=4) +
  geom_errorbar(aes(ymin = mean_x - se_x, ymax = mean_x + se_x), width = 0,linewidth=1.5) +
  labs(y="Confidence",x="Difficulty", color="Prior condition", shape="Prior condition",linetype="Prior condition") 

fit <- lmer(cj~mantype*cond*difficulty+(1|sub),df)
fit_a <- lmer(cj~mantype*cond*difficulty+(mantype|sub),df)
fit_b <- lmer(cj~mantype*cond*difficulty+(cond|sub),df) #issues
fit_c <- lmer(cj~mantype*cond*difficulty+(difficulty|sub),df) #doesn't converge
anova(fit,fit_a) #p<.001
anova(fit_a)
generalTestBF(cj ~ mantype*cond*difficulty + mantype:sub_cat + sub_cat, data = df,whichRandom = "sub_cat", neverExclude = "sub_cat",whichModels = "top") #compute BFs for n.s. effects
car::vif(fit_a)
performance::check_model(fit_a)
data.frame(effect('difficulty',fit_a))
data.frame(effect('cond',fit_a))

data.frame(effect('mantype:cond',fit_a))
pairs(emmeans::emmeans(fit_a, ~ cond*mantype),adjust="none")


#### 3. LME analysis of the single-trial EEG data ####
#1.1 Does confidence predict's the pe, and how about the prior beliefs manipulation
fit <- lmer(PE_amplitude~cj_cat*cond+(1|sub),df)
fit_a <- lmer(PE_amplitude~cj_cat*cond+(cj_cat|sub),df) #doesn't converge
fit_b <- lmer(PE_amplitude~cj_cat*cond+(cond|sub),df) #
anova(fit,fit_a) #non convergence
anova(fit,fit_b) #p<.001
anova(fit_b)
generalTestBF(PE_amplitude ~ cj_cat*cond + cond:sub_cat + sub_cat, data = subset(df,!is.na(df$PE_amplitude)),whichRandom = "sub_cat", neverExclude = "sub_cat",whichModels = "top") #compute BFs for n.s. effects
car::vif(fit_b)
pairs(emmeans::emmeans(fit_b, ~ cj_cat),adjust="none")
pairs(emmeans::emmeans(fit_b, ~ cj_cat*cond),adjust="none")

temp <- data.frame(effect('cj_cat:cond',fit_b))
temp$cond <- factor(temp$cond, levels = c("positive", "negative"))
ggplot(temp, aes(x=cj_cat, y=fit, group=cond)) +
  geom_line(aes(linetype=cond, color=cond),linewidth=1.5)+
  geom_point(aes(color=cond),size=4)+
  geom_errorbar(aes(ymin=fit-se, ymax=fit+se, color=cond), width=0,linewidth=1.5) +
  labs(y="Pe amplitude (z)",x="Confidence", color="Prior condition", shape="Prior condition",linetype="Prior condition") +
  theme(legend.position = "top")


#1.2 Does confidence predict the CPP, and how about the prior beliefs manipulation
fit <- lmer(P3_amplitude~cj_cat*cond+(1|sub),df)
fit_a <- lmer(P3_amplitude~cj_cat*cond+(cj_cat|sub),df) #doesn't converge
fit_b <- lmer(P3_amplitude~cj_cat*cond+(cond|sub),df) #
anova(fit,fit_b) #p<.001
car::vif(fit_b)
anova(fit_b)
pairs(emmeans::emmeans(fit_b, ~ cj_cat),adjust="none")
generalTestBF(P3_amplitude ~ cj_cat*cond + cond:sub_cat + sub_cat, data = subset(df,!is.na(df$P3_amplitude)),whichRandom = "sub_cat", neverExclude = "sub_cat",whichModels = "top") #compute BFs for n.s. effects

temp <- data.frame(effect('cj_cat:cond',fit_b))
temp$cond <- factor(temp$cond, levels = c("positive", "negative"))
ggplot(temp, aes(x=cj_cat, y=fit, group=cond)) +
  geom_line(aes(linetype=cond, color=cond),linewidth=1.5)+
  geom_point(aes(color=cond),size=4)+
  geom_errorbar(aes(ymin=fit-se, ymax=fit+se, color=cond), width=0,linewidth=1.5) +
  labs(y="CPP amplitude (z)",x="Confidence", color="Prior condition", shape="Prior condition",linetype="Prior condition") +
  theme(legend.position = "top")


#1.3 Does confidence predict the frontal signal (called fr_P3 here given it's stimulus-locked), and how about prior beliefs?
fit <- lmer(fr_P3_amplitude~cj_cat*cond+(1|sub),df)
fit_a <- lmer(fr_P3_amplitude~cj_cat*cond+(cj_cat|sub),df) #doesn't converge
fit_b <- lmer(fr_P3_amplitude~cj_cat*cond+(cond|sub),df)
anova(fit,fit_a) #no converge
anova(fit,fit_b) #p<.001
anova(fit_b)
generalTestBF(fr_P3_amplitude ~ cj_cat*cond + cond:sub_cat + sub_cat, data = subset(df,!is.na(df$fr_P3_amplitude)),whichRandom = "sub_cat", neverExclude = "sub_cat",whichModels = "top") #compute BFs for n.s. effects

temp <- data.frame(effect('cj_cat:cond',fit_b))
ggplot(temp, aes(x=cj_cat, y=fit, group=cond)) +
  geom_line(aes(linetype=cond, color=cond),linewidth=1.5)+
  geom_point(aes(color=cond),size=4)+
  geom_errorbar(aes(ymin=fit-se, ymax=fit+se, color=cond), width=0,linewidth=1.5) +
  labs(y="Frontal amplitude (z)",x="Confidence", color="Prior condition", shape="Prior condition",linetype="Prior condition") +
  theme(legend.position = "top")
pairs(emmeans::emmeans(fit_b, ~ cj_cat),adjust="none")
