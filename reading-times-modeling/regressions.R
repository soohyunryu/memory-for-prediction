rm(list=ls())
setwd('~/R/new-analysis')
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


show_summary <- function(model){
  intervals <- gather_draws(model, `b_.*`, regex=T) %>% mean_qi()
  stats <- gather_draws(model, `b_.*`, regex=T) %>% 
    mutate(above_0=ifelse(.value>0, 1,0)) %>% 
    group_by(.variable) %>% 
    summarize(pct_above_0=mean(above_0)) %>% 
    mutate(`P` = signif(2*pmin(pct_above_0,1-pct_above_0), digits=2)) %>% 
    left_join(intervals, by=".variable") %>% 
    mutate(lower=round(.lower, digits=1),
           upper=round(.upper, digits=1),
           E=round(.value, digits=1),
           `CI`=str_c("[",lower,", ", upper,"]"),
           Term=str_sub(.variable, 3, -1),
    ) %>% 
    select(Term, `E`, `CI`,`P`)
  stats
}


#### geco analysis 
GCdata <- read_csv("geco-annotated-data.csv")
GCdata<- drop_na(GCdata)
GCdata <- GCdata[which(GCdata$WORD_GO_PAST_TIME!=0 &GCdata$WORD_GO_PAST_TIME<3000),]

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

GCmodel_gp <- brm( WORD_GO_PAST_TIME ~ position + crnt_wlength_z + prev_wlength_z+
                 crnt_surprisal_z*crnt_top_syn_entropy_30_z+ 
                 crnt_wfreq_z  +
                 prev_surprisal_z*prev_top_syn_entropy_30_z+
                 prev_wfreq_z + 
                 (1+crnt_surprisal_z+position+crnt_top_syn_entropy_30_z+crnt_wfreq_z+prev_surprisal_z+prev_top_syn_entropy_30_z+
                    prev_wfreq_z+crnt_wlength_z + prev_wlength_z|PP_NR)+(1|WORD),
               data=GCdata_gd, cores= 36, prior = priors, backend = "cmdstanr")
save(GCmodel_gp, file = "GCmodel-gp.RData")




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



#### Natural Stories
SRdata <- read_csv("ns-annoated-data.csv")
SRdata <- drop_na(SRdata) # if no frequency, not recognized as single word (word + punctuation, etc), no previous values

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

