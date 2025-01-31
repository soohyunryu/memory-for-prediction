rm(list=ls()) 
setwd("~/Library/Mobile Documents/com~apple~CloudDocs/Umich/JLM/memory-for-prediction/reading-times-modeling/public/embeddings")
library(ggplot2)
library(dplyr)

stat_sum_single <- function(fun, geom="point", ...) {
  stat_summary(fun=fun, colour="red", geom=geom, size = 3, ...)
}

##### Relative Clause

Staub <- read.csv('results/staub-results-new.csv', sep=',',header=TRUE)
Staub[which(Staub$type == "rcn"),]$type = "Noun Onset"
Staub[which(Staub$type == "rcv"),]$type = "Embedded Verb"

# surprisal
ggplot(aes(x = sent_type, y = surprisal), data = Staub) +
  geom_point() + xlab("") + 
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean) + theme(text = element_text(family="Times New Roman",size=35))

# attention entropy -syntactic
ggplot(aes(x = sent_type, y = entropy_syntactic), data = Staub) +
  geom_point() + xlab("") + ylab("syntactic attention entropy") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20)  + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean) + theme(text = element_text(family="Times New Roman",size=35))


# attention entropy - global
ggplot(aes(x = sent_type, y = entropy_global), data = Staub) +
  geom_point() + xlab("") + ylab("global attention entropy") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20)  + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean) + theme(text = element_text(family="Times New Roman",size=35))



##### Center Embeddings vs. Right Branching

Stolz <- read.csv('results/stolz-results-new.csv', sep=',',header=TRUE)
Stolz[which(Stolz$type=="CE"),]$type = 'Center Embedding'
Stolz[which(Stolz$type=="RB"),]$type = 'Right Branching'
Stolz$level <-as.factor(Stolz$level)

# surprisal

ggplot(aes(x = level, y = surprisal), data = Stolz) +
  geom_point() + xlab("") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean, colour="blue") + theme(text = element_text(family="Times New Roman",size=35))+
  ylab('surprisal') + xlab('depth level')

# synatctic attention entropy

ggplot(aes(x = level, y = entropy_syntactic), data = Stolz) +
  geom_point() + xlab("") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean, colour="blue") + theme(text = element_text(family="Times New Roman",size=35))+
  ylab('syntactic attention entropy') + xlab('depth level')

# glabal attention entropy

ggplot(aes(x = level, y = entropy_global), data = Stolz) +
  geom_point() + xlab("") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean, colour="blue") + theme(text = element_text(family="Times New Roman",size=35))+
  ylab('global attention entropy') + xlab('depth level')

# attention entropy at head43

ggplot(aes(x = level, y = entropy_43), data = Stolz) +
  geom_point() + xlab("") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean, colour="blue") + theme(text = element_text(family="Times New Roman",size=35))+
  ylab('attention entropy at head43') + xlab('depth level')

# attention paid to the correct target

ggplot(aes(x = level, y = attn_to_target), data = Stolz) +
  geom_point() + xlab("") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.1, size = 0.5) +
  stat_sum_single(mean, colour="blue") + theme(text = element_text(family="Times New Roman",size=35))+
  ylab('attention to correct subject') + xlab('depth level')



# Attention to nouns from verb-level2

Stolz_level2 = Stolz[which(Stolz$level==2),]
Stolz_level2_CE = Stolz_level2[which(Stolz_level2$type=="Center Embedding"),]

x<-rep(c("noun_lv1","noun_lv2","noun_lv3"),each=15)
type <- rep(c("Center Embedding"), each = 45)
y <- Stolz_level2_CE$attn_to_lv1_noun
y <- c(y,Stolz_level2_CE$attn_to_lv2_noun)
y <- c(y,Stolz_level2_CE$attn_to_lv3_noun)

lv2_df_CE <- data.frame(x,y, type)

Stolz_level2 = Stolz[which(Stolz$level==2),]
Stolz_level2_RB = Stolz_level2[which(Stolz_level2$type=="Right Branching"),]

x<-rep(c("noun_lv1","noun_lv2","noun_lv3"),each=15)
type <- rep(c("Right Branching"), rep = 45)
y <- Stolz_level2_RB$attn_to_lv1_noun
y <- c(y,Stolz_level2_RB$attn_to_lv2_noun)
y <- c(y,Stolz_level2_RB$attn_to_lv3_noun)
lv2_df_RB <- data.frame(x,y,type)

lv2_df <- rbind(lv2_df_RB, lv2_df_CE)

ggplot(aes(x = x, y = y), data = lv2_df) +
  geom_point() + ylab("attention from verb_lv2") +xlab("") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.3) +
  stat_sum_single(mean, colour="blue") + theme(text = element_text(family="Times New Roman",size=35))

# Attention to nouns from verb-level3

Stolz_level3 = Stolz[which(Stolz$level==3),]
Stolz_level3_CE = Stolz_level3[which(Stolz_level3$type=="Center Embedding"),]

x<-rep(c("noun_lv1","noun_lv2","noun_lv3"),each=15)
type <- rep(c("Center Embedding"), each = 45)
y <- Stolz_level3_CE$attn_to_lv1_noun
y <- c(y,Stolz_level3_CE$attn_to_lv2_noun)
y <- c(y,Stolz_level3_CE$attn_to_lv3_noun)

lv3_df_CE <- data.frame(x,y, type)

Stolz_level3 = Stolz[which(Stolz$level==3),]
Stolz_level3_RB = Stolz_level3[which(Stolz_level3$type=="Right Branching"),]

x<-rep(c("noun_lv1","noun_lv2","noun_lv3"),each=15)
type <- rep(c("Right Branching"), rep = 45)
y <- Stolz_level3_RB$attn_to_lv1_noun
y <- c(y,Stolz_level3_RB$attn_to_lv2_noun)
y <- c(y,Stolz_level3_RB$attn_to_lv3_noun)
lv3_df_RB <- data.frame(x,y,type)

lv3_df <- rbind(lv3_df_RB, lv3_df_CE)

ggplot(aes(x = x, y = y), data = lv3_df) +
  geom_point() + ylab("attention from verb_lv3") +xlab("") + #Verb agreement across four levels of clausal embedding in 'it' clefts") +
  theme_bw(base_size=20) + theme(legend.position = "bottom") +
  facet_grid(.~type) + 
  stat_summary(fun.data = "mean_cl_boot", geom="errorbar", colour="red", width=0.3) +
  stat_sum_single(mean, colour="blue") +theme(text = element_text(family="Times New Roman",size=35))

