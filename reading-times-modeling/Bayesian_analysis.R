rm(list=ls())
setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Umich/Psych619/journal_code/natural-reading/eyetracking")
knitr::opts_chunk$set(echo = FALSE, warning=F)
options(knitr.table.format = "html")
library(tidyverse)
library(readr)
library(brms)
library(lme4)
library(rstan)
library(tidybayes)
library(knitr)
library(matrixStats)
library(magrittr)
library(coefplot)
theme_set(theme_bw())
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

#Read data

metrics_processed <- read.csv('data/metrics_processed_22sep14.csv',sep=',',header = TRUE)
metrics_processed <-metrics_processed[which(metrics_processed$ff!=0),]
dim(metrics_processed)

drops <- c('prev_nnull_entropy','this_nnull_entropy','this_pruned_entropy','prev_pruned_entropy')
metrics_processed <- metrics_processed[,!(names(metrics_processed) %in% drops)]
metrics_processed$trt_ff_diff <- metrics_processed$trt - metrics_processed$ff


# Centering 
metrics_processed <- metrics_processed %>% 
  mutate(ff_center=ff-mean(ff, na.rm=T),
         entropy_reduction_z = scale(entropy_reduction),
         this_surprisal_z = scale(this_surprisal),
         this_entropy_z = scale(this_entropy),
         this_wfreq_z = scale(this_wfreq),
         this_wlength_z = scale(this_wlength),
         prev_entropy_z = scale(prev_entropy),
         prev_surprisal_z = scale(prev_surprisal),
         prev_wfreq_z = scale(prev_wfreq),
         prev_wlength_z = scale(prev_wlength)
  )

#Set priors
## BRM models
#We include a by-subject effect for everything, and a by_word random intercept (full mixed effects). 
#Priors:
#- normal(1000,1000) for intercept -- we think RTs are about 1 second usually
#- normal(0,500) for beta and sd -- we don't really know what effects are
#- lkj(1) for correlations -- we don't have reason to think correlations might go any particular way 

#```{r}

priors <- c(
  set_prior("normal(1000, 1000)", class="Intercept"),
  set_prior("normal(0, 500)", class="b"),
  set_prior("normal(0, 500)", class="sd"),
  set_prior("lkj(1)",       class="cor"))


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


#### Geco Eye tracking

#total reading time: 
brm_trt_entropy_spill_with_w <- brm(trt ~ this_loc * this_wlength_z +
                                   this_surprisal_z* this_wlength_z +
                                   this_entropy_z *this_wlength_z+ 
                                   this_wfreq_z * this_wlength_z +
                                   prev_surprisal_z* prev_wlength_z+ 
                                   prev_entropy_z*prev_wlength_z+
                                   prev_wfreq_z * prev_wlength_z + 
                                   (1+this_surprisal_z+this_loc+this_entropy_z+this_wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|participant)+(1|this_word),
                                 data=metrics_processed)

gd_entropy_reduction_with_w <- lmer(gd ~ this_loc * this_wlength_z +
                                      this_surprisal_z* this_wlength_z +
                                      entropy_reduction_z *this_wlength_z+ 
                                      this_wfreq_z * this_wlength_z +
                                      prev_surprisal_z* prev_wlength_z +
                                      prev_wfreq_z * prev_wlength_z + 
                                    (1+this_surprisal_z+this_loc+entropy_reduction_z+this_wfreq_z+prev_surprisal_z+prev_wfreq_z|participant)+(1|this_word),
                                    data=metrics_processed)

gd_entropy_spill_with_w <- lmer(gd ~ this_loc * this_wlength_z +
                                    this_surprisal_z* this_wlength_z +
                                    this_entropy_z *this_wlength_z+ 
                                    this_wfreq_z * this_wlength_z +
                                    prev_surprisal_z* prev_wlength_z +
                                    prev_entropy_z* prev_wlength_z +
                                    prev_wfreq_z * prev_wlength_z + 
                                    (1+this_surprisal_z+this_loc+entropy_reduction_z+this_wfreq_z+prev_surprisal_z+prev_wfreq_z|participant)+(1|this_word),
                                    data=metrics_processed)

save(brm_trt_entropy_spill_with_w,file = "Rdata/brm_trt_entropy_spill_with_w.RData")
#load('Rdata/brm_trt_entropy_spill_with_w.RData')
load('Rdata/trt_entropy_spill_with_w.RData')

#first fixation: 
brm_ff_entropy_spill_with_w <- brm(ff ~ this_loc * this_wlength_z +
                                      this_surprisal_z* this_wlength_z +
                                      this_entropy_z *this_wlength_z+ 
                                      this_wfreq_z * this_wlength_z +
                                      prev_surprisal_z* prev_wlength_z+ 
                                      prev_entropy_z*prev_wlength_z+
                                      prev_wfreq_z * prev_wlength_z + 
                                      (1+this_surprisal_z+this_loc+this_entropy_z+this_wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|participant)+(1|this_word),
                                    data=metrics_processed)

save(brm_ff_entropy_spill_with_w,file = "Rdata/brm_trt_entropy_spill_with_w.RData")
#load('Rdata/brm_trt_entropy_spill_with_w.RData')


#trt- ff: 


trt_ff_diff_entropy_spill_with_w <- lmer(trt_ff_diff ~ this_loc * this_wlength_z +
                                    this_surprisal_z* this_wlength_z +
                                    this_entropy_z *this_wlength_z+ 
                                    this_wfreq_z * this_wlength_z +
                                    prev_surprisal_z* prev_wlength_z +
                                    prev_entropy_z* prev_wlength_z +
                                    prev_wfreq_z * prev_wlength_z + 
                                    (1+this_surprisal_z+this_loc+entropy_reduction_z+this_wfreq_z+prev_surprisal_z+prev_wfreq_z|participant)+(1|this_word),
                                    data=metrics_processed)



brm_trt_ff_diff_entropy_spill_with_w <- brm(trt_ff_diff ~ this_loc * this_wlength_z +
                                     this_surprisal_z* this_wlength_z +
                                     this_entropy_z *this_wlength_z+ 
                                     this_wfreq_z * this_wlength_z +
                                     prev_surprisal_z* prev_wlength_z+ 
                                     prev_entropy_z*prev_wlength_z+
                                     prev_wfreq_z * prev_wlength_z + 
                                     (1+this_surprisal_z+this_loc+this_entropy_z+this_wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|participant)+(1|this_word),
                                   data=metrics_processed)

save(brm_trt_ff_diff_entropy_spill_with_w,file = "Rdata/brm_trt_ff_diff_entropy_spill_with_w.RData")
load('Rdata/brm_trt_ff_diff_entropy_spill_with_w.RData')

#go past time: 
brm_gp_entropy_spill_with_w <- brm(gp ~ this_loc * this_wlength_z +
                                      this_surprisal_z* this_wlength_z +
                                      this_entropy_z *this_wlength_z+ 
                                      this_wfreq_z * this_wlength_z +
                                      prev_surprisal_z* prev_wlength_z+ 
                                      prev_entropy_z*prev_wlength_z+
                                      prev_wfreq_z * prev_wlength_z + 
                                      (1+this_surprisal_z+this_loc+this_entropy_z+this_wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|participant)+(1|this_word),
                                    data=metrics_processed)

save(brm_gp_entropy_spill_with_w,file = "Rdata/brm_gp_entropy_spill_with_w.RData")
#load('Rdata/brm_gp_entropy_spill_with_w.RData')



#gaze duration time: 
brm_gd_entropy_spill_with_w <- brm(gd ~ this_loc * this_wlength_z +
                                     this_surprisal_z* this_wlength_z +
                                     this_entropy_z *this_wlength_z+ 
                                     this_wfreq_z * this_wlength_z +
                                     prev_surprisal_z* prev_wlength_z+ 
                                     prev_entropy_z*prev_wlength_z+
                                     prev_wfreq_z * prev_wlength_z + 
                                     (1+this_surprisal_z+this_loc+this_entropy_z+this_wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|participant)+(1|this_word),
                                   data=metrics_processed)

save(brm_gd_entropy_spill_with_w,file = "Rdata/brm_gd_entropy_spill_with_w.RData")
#load('Rdata/brm_gp_entropy_spill_with_w.RData')

#gaze duration time: 
brm_spillover_entropy_spill_with_w <- brm(spillover ~ this_loc * this_wlength_z +
                                     this_surprisal_z* this_wlength_z +
                                     this_entropy_z *this_wlength_z+ 
                                     this_wfreq_z * this_wlength_z +
                                     prev_surprisal_z* prev_wlength_z+ 
                                     prev_entropy_z*prev_wlength_z+
                                     prev_wfreq_z * prev_wlength_z + 
                                     (1+this_surprisal_z+this_loc+this_entropy_z+this_wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|participant)+(1|this_word),
                                   data=metrics_processed)

save(brm_spillover_entropy_spill_with_w,file = "Rdata/brm_spillover_entropy_spill_with_w.RData")
#load('Rdata/brm_gp_entropy_spill_with_w.RData')

load('Rdata/brm_gd_entropy_spill_with_w.RData')
summary(brm_gp_entropy_spill_with_w)
summary(brm_gd_entropy_spill_with_w)


#summary(brm_trt_entropy_spill_with_w)
#fixef(brm_trt_entropy_spill_with_w)
#traceplot(brm_trt_entropy_spill_with_w, pars = c("beta","sigma_e"),inc_warmup = FALSE)


#brm_trt_entropy_spill_with_w %>%
#  ggplot(aes(y = condition, x = condition_mean, fill = stat(abs(x) < .8))) +
#  stat_halfeye() +
#  geom_vline(xintercept = c(-.8, .8), linetype = "dashed") +
#  scale_fill_manual(values = c("gray80", "skyblue"))

#summary(brm_trt_entropy_spill_with_w) %>% 
#  ggplot(aes(x = b_Intercept, y = 0)) +
#  stat_halfeye() +
#  scale_y_continuous(NULL, breaks = NULL)




#### Natural Stories Self-paced reading
setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Umich/Psych619/wd/NaturalStoriesRT")

NSdata <- read_csv('metrics_jan12.csv')
NSdata <- drop_na(NSdata)
NSdata$entropy_reduction <- NSdata$prev_all_entropy - NSdata$all_entropy
dim(NSdata)


NSdata <- drop_na(NSdata)
#NSdata %>% drop_na()
NSdata <- NSdata %>% mutate(meanItemRT_center=meanItemRT-mean(meanItemRT, na.rm=T),
                            entropy_reduction_center = entropy_reduction - mean(entropy_reduction,na.rm=T),
                            entropy_reduction_z = scale(entropy_reduction),
                            entropy_center = all_entropy - mean(all_entropy, na.rm=T),
                            entropy_z = scale(all_entropy),
                            prev_entropy_center = prev_all_entropy - mean(prev_all_entropy, na.rm=T),
                            prev_entropy_z = scale(prev_all_entropy),
                            surprisal_center = surprisal - mean(surprisal, na.rm=T),
                            surprisal_z = scale(surprisal),
                            prev_surprisal_center = prev_surprisal - mean(prev_surprisal, na.rm=T),
                            prev_surprisal_z = scale(prev_surprisal),
                            wlength_center = w_len - mean(w_len, na.rm=T),
                            wlength_z = scale(w_len),
                            prev_wlength_center = prev_w_len - mean(prev_w_len, na.rm=T),
                            prev_wlength_z = scale(prev_w_len),
                            wfreq_center = log_freq- mean(log_freq, na.rm=T),
                            wfreq_z = scale(log_freq),
                            prev_wfreq_center = prev_log_freq- mean(prev_log_freq, na.rm=T),
                            prev_wfreq_z = scale(prev_log_freq))


brm_NS_entropy_spill_with_w <- brm(RT ~ loc * wlength_z +
                                     surprisal_z* wlength_z +
                                     entropy_z* wlength_z +
                                     wfreq_z * wlength_z+
                                     prev_surprisal_z* prev_wlength_z +
                                     prev_entropy_z * prev_wlength_z +
                                     prev_wfreq_z * prev_wlength_z + 
                                     (1+surprisal_z+loc+entropy_z+wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|WorkerId)+(1|word),
                                   data=NSdata)
brm_NS_entropy_spill_with_w <- add_criterion(brm_NS_entropy_spill_with_w, "loo")

brm_NS_base_entropy_spill_with_w <- brm(RT ~ loc * wlength_z +
                                 surprisal_z* wlength_z +
                                 wfreq_z * wlength_z+
                                 prev_surprisal_z* prev_wlength_z +
                                 prev_wfreq_z * prev_wlength_z + 
                                 (1+surprisal_z+loc+entropy_z+wfreq_z+prev_surprisal_z+prev_entropy_z+prev_wfreq_z|WorkerId)+(1|word),
                               data=NSdata)
brm_NS_base_entropy_spill_with_w <- add_criterion(brm_NS_base_entropy_spill_with_w, "loo")

setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Umich/Psych619/wd/ReadingTime")
load('Rdata/brm_NS_entropy_spill_with_w.RData')
load('Rdata/brm_NS_base_entropy_spill_with_w.RData')



summary(brm_NS_entropy_spill_with_w)
fixef(brm_NS_entropy_spill_with_w)
plot(brm_NS_entropy_spill_with_w)

summary(brm_NS_base_entropy_spill_with_w)
anova(brm_NS_entropy_spill_with_w,brm_NS_base_entropy_spill_with_w)

loo_compare(brm_NS_entropy_spill_with_w,brm_NS_base_entropy_spill_with_w,criterion = "loo")
