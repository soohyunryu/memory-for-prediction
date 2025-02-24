rm(list=ls())
setwd('your-working-directory')
knitr::opts_chunk$set(echo = FALSE, warning=F)
options(knitr.table.format = "html")
library(tidyverse)
library(readr)
library(brms)
library(lme4)
library(rstan)
library(tidybayes)
library(knitr)
library(cmdstanr)
library(matrixStats)
library(magrittr)
library(coefplot)
theme_set(theme_bw())
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


#Set priors
priors <- c(
  set_prior("normal(1000,1000)", class = "Intercept"),
  set_prior("normal(0, 500)", class = "b"),
  set_prior("normal(0, 100)", class = "sd"),
  set_prior("lkj(1)", class = "cor"))


#### geco analysis 

#Fitting the model for go-past-time
GCdata <- read_csv("geco-annotated-data.csv")
GCdata<- drop_na(GCdata)
GCdata <- GCdata[which(GCdata$WORD_GAZE_DURATION!=0 &GCdata$WORD_GAZE_DURATION<3000),]



GCdata <- GCdata %>%  mutate(crnt_surprisal_z = scale(crnt_surprisal),
                             crnt_top_syn_entropy_30_z = scale(crnt_top_syn_entropies_30),
                             crnt_not_top_syn_entropy_30_z = scale(crnt_not_top_syn_entropies_30),
                             crnt_wfreq_z = scale(frequency),
                             crnt_wlength_z = scale(w_len),
                             
                             prev_surprisal_z = scale(prev_surprisal),
                             prev_top_syn_entropy_30_z = scale(prev_top_syn_entropies_30),
                             prev_not_top_syn_entropy_30_z = scale(prev_not_top_syn_entropies_30),
                             prev_wfreq_z = scale(prev_frequency),
                             prev_wlength_z = scale(prev_w_len))

GCdata$WORD_GO_PAST_TIME <- as.numeric(GCdata$WORD_GO_PAST_TIME)

GCmodel_gp <- brm( WORD_GO_PAST_TIME ~ position + crnt_wlength_z + prev_wlength_z+
                 crnt_surprisal_z*crnt_top_syn_entropy_30_z+ 
                 crnt_wfreq_z  +
                 prev_surprisal_z*prev_top_syn_entropy_30_z+
                 prev_wfreq_z + 
                 (1+crnt_surprisal_z+position+crnt_top_syn_entropy_30_z+crnt_wfreq_z+prev_surprisal_z+prev_top_syn_entropy_30_z+
                    prev_wfreq_z+crnt_wlength_z + prev_wlength_z|PP_NR)+(1|WORD),
               data=GCdata_gd, cores= 36, prior = priors, backend = "cmdstanr")
save(GCmodel_gp, file = "GCmodel-gp.RData")
summary(GCmodel_gp)


#Fitting the model for first-fixation-duration 
GCdata <- read_csv("geco-annotated-data.csv")
GCdata <- drop_na(GCdata)
GCdata_ff <- GCdata[which(GCdata$WORD_FIRST_FIXATION_DURATION!=0 &GCdata$WORD_FIRST_FIXATION_DURATION<3000),]
summary(GCdata_ff$WORD_FIRST_FIXATION_DURATION)
GCdata_ff <- GCdata_ff %>%  mutate(crnt_surprisal_z = scale(crnt_surprisal),
                             crnt_top_syn_entropy_30_z = scale(crnt_top_syn_entropies_30),
                             crnt_not_top_syn_entropy_30_z = scale(crnt_not_top_syn_entropies_30),
                             crnt_wfreq_z = scale(frequency),
                             crnt_wlength_z = scale(w_len),
                             
                             prev_surprisal_z = scale(prev_surprisal),
                             prev_top_syn_entropy_30_z = scale(prev_top_syn_entropies_30),
                             prev_not_top_syn_entropy_30_z = scale(prev_not_top_syn_entropies_30),
                             prev_wfreq_z = scale(prev_frequency),
                             prev_wlength_z = scale(prev_w_len))

GCmodel_ff <- brm( WORD_GAZE_DURATION ~ position + crnt_wlength_z + prev_wlength_z+
                     crnt_surprisal_z*crnt_top_syn_entropy_30_z+ 
                     crnt_wfreq_z  +
                     prev_surprisal_z*prev_top_syn_entropy_30_z+
                     prev_wfreq_z + 
                     (1+crnt_surprisal_z+position+crnt_top_syn_entropy_30_z+crnt_wfreq_z+prev_surprisal_z+prev_top_syn_entropy_30_z+
                        prev_wfreq_z+crnt_wlength_z + prev_wlength_z|PP_NR)+(1|WORD),
                   data=GCdata_ff, cores= 36, prior = priors, backend = "cmdstanr")

save(GCmodel_ff, file = "GCmodel-ff.RData")
summary(GCmodel_ff)


#### Natural Stories
SRdata <- read_csv("ns-annoated-data.csv")
SRdata <- drop_na(SRdata) 

SRdata <- SRdata %>%  mutate(crnt_surprisal_z = scale(surprisal),
                             crnt_top_syn_entropy_30_z = scale(top_syn_entropy_30),
                             crnt_not_top_syn_entropy_30_z = scale(not_top_syn_entropy_30),
                             crnt_wfreq_z = scale(frequency),
                             crnt_wlength_z = scale(w_len),
                             
                             prev_surprisal_z = scale(prev_surprisal),
                             prev_top_syn_entropy_30_z = scale(prev_top_syn_entropy_30),
                             prev_not_top_syn_entropy_30_z = scale(prev_not_top_syn_entropy_30),
                             prev_wfreq_z = scale(prev_frequency),
                             prev_wlength_z = scale(prev_w_len))

SRmodel <- brm(RT ~ position + crnt_wlength_z + prev_wlength_z+
                 crnt_surprisal_z*crnt_top_syn_entropy_30_z+ 
                 crnt_wfreq_z  +
                 prev_surprisal_z*prev_top_syn_entropy_30_z+
                 prev_wfreq_z + 
                 (1+crnt_surprisal_z+position+crnt_top_syn_entropy_30_z+crnt_wfreq_z+prev_surprisal_z+prev_top_syn_entropy_30_z+
                    prev_wfreq_z+crnt_wlength_z + prev_wlength_z|WorkerId)+(1|word),
               data=SRdata, cores= 36, prior = priors, backend = "cmdstanr")


save(SRmodel,file = "SRmodel.RData")
summary(SRmodel)
