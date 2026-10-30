#Author: Heng Zhang (Hank) 
#Email: heng.zhang@eawag.ch
#Date: 19.11.2025
#Aim: to plot all taxa of terrestrial-aquatic linkage of biodiversity with null model.

library(ggplot2)
library(Rmisc)

root<-"XXX"
aquacsvfolderdir<-paste0(root,"/","data/biodiv/table/species_richness_taxa")
dirto<-paste0(root,"/","result/figure/terra_aqua_model/terra_aqua_linkage_all_taxa_V03")
if(dir.exists(dirto)==FALSE){dir.create(dirto,recursive = TRUE)}

all_taxa_label<-"aquatic"       #aquatic    all
if(all_taxa_label=="all"){
  terracsvfolderdir<-paste0(root,"/","data/env_var/catch_C_r_mat/catch_C_r_mat_all_taxa")
  aquafilename<-paste0("species_richness_all_taxa.csv")
  aquafiledir<-paste0(aquacsvfolderdir,"/",aquafilename)
  plot_legend_pair<-c("aquatic - terrestrial")
  
  terra_biodiv_header<-"all"
  aqua_biodiv_header<-"all"

  plot_y_max<-0.7
  plot_y_min<-0.0
}else{
  terracsvfolderdir<-paste0(root,"/","data/env_var/catch_C_r_mat/catch_C_r_mat_terra_taxa")
  aquafilename<-paste0("species_richness_aquatic_taxa.csv")
  aquafiledir<-paste0(aquacsvfolderdir,"/",aquafilename)
  plot_legend_pair<-c("aquatic (only) - terrestrial (only)")
  
  terra_biodiv_header<-"terrestrial"
  aqua_biodiv_header<-"aquatic"
  
  plot_y_max<-0.35
  plot_y_min<-0.0
}

num_k_fold<-100

terrafilename<-paste0("terra_effect_matrix_",terra_biodiv_header,"_decay_func_exp.csv")
terrafiledir<-paste0(terracsvfolderdir,"/",terrafilename)
terraDF<-read.csv(terrafiledir)
terra_dim<-dim(terraDF)
# num_terra_biodiv<-length(terra_biodiv_headers)
  
aquaDF<-read.csv(aquafiledir)
aqua_dim<-dim(aquaDF)
#obtain aquatic biodiv and terrestrial biodiv headers
aqua_biodiv_headers<-names(aquaDF)[9:aqua_dim[2]]
num_aqua_biodiv<-length(aqua_biodiv_headers)
  
site_header<-"SiteID"
num_sites<-aqua_dim[1]

bool_biodiv_sqrt<-TRUE
bool_rm_zero_rec<-TRUE
metricName<-"SD"
  
terra_ID_start<-3          

rm_ID_list<-c("")

#distance decay function used in C(r) matrix calculation
decay_func<-"exp"

#distance list
dist_max<-400
dist_min<-1
num_bins<-50
dist_list1<-seq(0.1,0.9,length.out=9)
dist_list2<-exp(seq(log(dist_min),log(dist_max),length.out=num_bins))
dist_list<-c(dist_list1,dist_list2)
num_dist_sim<-length(dist_list)

if(bool_rm_zero_rec){
  aquaDF[aquaDF==-1]<-0
  terraDF[terraDF==-1]<-0
  sum_aqua_zeros<-apply(aquaDF[,aqua_biodiv_headers],1,sum)
  sum_terra_zeros<-apply(terraDF[,terra_ID_start:terra_dim[2]],1,sum)
  bool_aqua_zeros<-sum_aqua_zeros==0
  bool_terra_zeros<-sum_terra_zeros==0
  bool_non_zeros<-!(bool_aqua_zeros|bool_terra_zeros)

  aquaDF<-aquaDF[bool_non_zeros,]
  terraDF<-terraDF[bool_non_zeros,]
}
if(bool_biodiv_sqrt){aquaDF[,aqua_biodiv_headers]<-sqrt(aquaDF[,aqua_biodiv_headers])}

num_sites<-dim(aquaDF)[1]

#random sampling for k fold TerrAqua model
rand_sam_ID<-sample(x=1:num_sites,replace=FALSE)
sam_ID_seqs<-split(rand_sam_ID, sort(rank(rand_sam_ID)%%num_k_fold))

#---------------------------------- TerrAqua model with null model I ----------------------------------
model_metrics<-array(NA,dim=c(num_k_fold,num_dist_sim))
model_params<-array(NA,dim=c(num_k_fold,num_dist_sim))
null_metrics<-array(NA,dim=c(num_k_fold,num_dist_sim))
null_params<-array(NA,dim=c(num_k_fold,num_dist_sim))
idx_sample<-1:num_sites
for(i_dist in 1:num_dist_sim){
  dist_value<-dist_list[i_dist]
  print(paste0("assessing biodiv linkage: ",aqua_biodiv_header," - ",terra_biodiv_header,"; dist: ",sprintf("%.2f",dist_value),"    ",sprintf("(%d / %d)",i_dist,num_dist_sim)))
  for(i_bsp in 1:num_k_fold){
    #TerrAqua model
    aqua_idx<-sam_ID_seqs[i_bsp][[1]]
    terra_idx<-aqua_idx
    C_sample<-data.frame(terraDF[terra_idx,terra_ID_start+i_dist-1])
    modelDF<-data.frame(aquaDF[aqua_idx,aqua_biodiv_header])
    modelDF<-cbind(modelDF,C_sample)
    names(modelDF)<-c("biodiv","dist")
    
    m_lmer_model <- lm(biodiv ~ dist, data=modelDF)
    m_lm_model_pred<-summary(m_lmer_model)
    model_params[i_bsp,i_dist]<-as.numeric(m_lmer_model$coefficients[2])
    model_metrics[i_bsp,i_dist]<-m_lm_model_pred$adj.r.squared
    
    #null model
    aqua_idx<-sam_ID_seqs[i_bsp][[1]]
    terra_idx<-aqua_idx[sample(x=1:length(aqua_idx),replace=FALSE)]
    
    C_sample<-data.frame(terraDF[terra_idx,terra_ID_start+i_dist-1])
    modelDF<-data.frame(aquaDF[aqua_idx,aqua_biodiv_header])
    modelDF<-cbind(modelDF,C_sample)
    names(modelDF)<-c("biodiv","dist")    
    n_lmer_model <- lm(biodiv ~ dist, data=modelDF)
    n_lm_model_pred<-summary(n_lmer_model)
    null_params[i_bsp,i_dist]<-as.numeric(n_lmer_model$coefficients[2])
    null_metrics[i_bsp,i_dist]<-n_lm_model_pred$adj.r.squared
  }
  print(paste0("average model adj.r2: ",sprintf("%.3f",mean(model_metrics[,i_dist])),
               ", null adj.r2: ",sprintf("%.3f",mean(null_metrics[,i_dist]))))
}

#format k fold results for model metric
num_rec_total<-num_k_fold*length(dist_list)
model_data_plot<-array(0,dim=c(num_rec_total,2))
null_data_plot<-array(0,dim=c(num_rec_total,2))
i_pd<-1
for(j in 1:length(dist_list)){
  dist<-dist_list[j]
  rec_col<-model_metrics[,j]
  rec_col<-na.omit(rec_col)
  num_rec<-length(rec_col)
  model_data_plot[i_pd:(i_pd+num_rec-1),1]<-rep(dist,num_rec)
  model_data_plot[i_pd:(i_pd+num_rec-1),2]<-rec_col
  
  rec_col<-null_metrics[,j]
  rec_col<-na.omit(rec_col)
  num_rec<-length(rec_col)
  null_data_plot[i_pd:(i_pd+num_rec-1),1]<-rep(dist,num_rec)
  null_data_plot[i_pd:(i_pd+num_rec-1),2]<-rec_col
  i_pd<-i_pd+num_rec
}
model_data_plot<-model_data_plot[1:(i_pd-1),]
model_data_plot<-data.frame(model_data_plot)
names(model_data_plot)<-c("distance","r2")
plotstats_m<-summarySE(model_data_plot, conf.interval = 0.95, measurevar="r2", groupvars=c("distance"))

null_data_plot<-null_data_plot[1:(i_pd-1),]
null_data_plot<-data.frame(null_data_plot)
names(null_data_plot)<-c("distance","r2")
plotstats_n<-summarySE(null_data_plot, conf.interval = 0.95, measurevar="r2", groupvars=c("distance"))


y_m<-plotstats_m$r2
if(metricName=="SD"){
  y1_m<-plotstats_m$r2+plotstats_m$sd
  y2_m<-plotstats_m$r2-plotstats_m$sd
}else if(metricName=="CI"){
  y1_m<-plotstats_m$r2+plotstats_m$ci
  y2_m<-plotstats_m$r2-plotstats_m$ci
}
max_id<-which.max(y_m)
max_dist<-dist_list[max_id]

y_n<-plotstats_n$r2
if(metricName=="SD"){
  y1_n<-plotstats_n$r2+plotstats_n$sd
  y2_n<-plotstats_n$r2-plotstats_n$sd
}else if(metricName=="CI"){
  y1_n<-plotstats_n$r2+plotstats_n$ci
  y2_n<-plotstats_n$r2-plotstats_n$ci
}

print(paste0("max R2 distance: ", as.character(max_dist),", dist ID: ",as.character(max_id),
             ", R2 = ",as.character(y_m[max_id])))


#---------------------------------- Plot the regression line ----------------------------------   
#step 4: plot R2~distance relations
x<-dist_list
plot_DF<-data.frame(x,y_m,y_n)
p<-ggplot(plot_DF,aes(x=x)) + theme_classic()

for(i in 1:num_k_fold){
  y_m_line<-as.numeric(model_metrics[i,])
  y_n_line<-as.numeric(null_metrics[i,])
  loop_data<-data.frame(x,y_m_line,y_n_line)
  p<-p + geom_line(data=loop_data,aes(y=y_m_line),linewidth=0.4, colour="RoyalBlue1", linetype="solid",alpha=0.1)
  p<-p + geom_line(data=loop_data,aes(y=y_n_line),linewidth=0.4, colour="grey51", linetype="dashed",alpha=0.1)
}
p<-p + geom_line(data=plot_DF,aes(x=x,y=y_m),linewidth=0.8, colour="RoyalBlue4", linetype="solid",alpha=1)
p<-p + geom_line(data=plot_DF,aes(x=x,y=y_n),linewidth=0.8, colour="grey51", linetype="dashed",alpha=1)

# 
# y_m_line<-y_m
# y1_m_line<-y1_m
# y2_m_line<-y2_m
# y_n_line<-y_n
# y1_n_line<-y1_n
# y2_n_line<-y2_n
# loop_data<-data.frame(x,y1_m_line,y1_m_line,y2_m_line,y_n_line,y1_n_line,y2_n_line)
# p<-p + geom_ribbon(data=loop_data,aes(ymin = y1_m_line, ymax = y2_m_line),fill="grey61",alpha=0.2)
# p<-p + geom_line(data=loop_data,aes(y=y_m_line),linewidth=1, colour="RoyalBlue4", linetype="solid",alpha=1)
# p<-p + geom_ribbon(data=loop_data,aes(ymin = y1_n_line, ymax = y2_n_line),fill="Wheat1",alpha=0.2)
# p<-p + geom_line(data=loop_data,aes(y=y_n_line),linewidth=1, colour="Wheat4", linetype="solid",alpha=1)

p<-p + geom_text(x=log10(dist_min)+0.45*(log10(dist_max)-log10(dist_min)), y=y_min+0.85*(plot_y_max-plot_y_min),
                 label = plot_legend_pair,
                 colour="RoyalBlue4",size=6.5)
p<-p + 
  scale_y_continuous(breaks=c(0.0,0.2,0.4,0.6),limits = c(-0.03,0.7)) +
  # scale_y_continuous(breaks=c(0.0,0.1,0.2,0.3),limits = c(-0.02,0.36)) +
  scale_x_continuous(trans = "log10", breaks=c(1,10,100),limits = c(0.2,400)) +
  xlab("Flow Distance (km)") + ylab(expression(adj.R^2))+
  # geom_vline(aes(xintercept=max_dist), linewidth=2.5, colour="PaleVioletRed", linetype="dashed",alpha=0.5) +
  # geom_hline(aes(yintercept=0.0), linewidth=0.88, colour="grey51", linetype="dashed",alpha=0.7) +
  theme(axis.title.y=element_text(angle=90, colour="black",size=24)) +
  theme(axis.title.x=element_text(angle=0, colour="black",size=24)) +
  theme(axis.text.y=element_text(angle=0, colour="grey21",size=20)) +
  theme(axis.text.x=element_text(angle=0, colour="grey21",size=20)) +
  theme(plot.title = element_text(face = "bold",colour="grey21",size=20))
p    
    
plotname1<-paste0("curve_terra_aqua_linkage_",all_taxa_label,"_taxa",".pdf")
ggsave(paste0(dirto,"/",plotname1),p,width=5.5, height=6.5,unit="in")

