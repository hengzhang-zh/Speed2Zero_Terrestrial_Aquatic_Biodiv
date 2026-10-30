#Author: Heng Zhang (Hank) 
#Email: heng.zhang@eawag.ch
#Date: 07.08.2026
#Aim: to plot the spatial variogram for .

library(spam)
library(fields)
library(sp)
library(gstat)
library(ggplot2)

root<-"XXX"
rootsemivariofolderdir<-paste0(root,"/","result/table/env_var/spatial_autocorrelation_semivario")
rootresultfolderdir<-paste0(root,"/","result/figure","/","terra_spatial_semivariogram")

all_taxa_label<-"all"       #terrestrial    all
max_semivario_dist<-50

if(all_taxa_label=="all"){
  semivariofolderdir<-paste0(rootsemivariofolderdir,"/","sp_all")
  resultfolderdir<-paste0(rootresultfolderdir,"/","sp_all")
  terra_biodiv_headers<-c("all", "birds", "lepidoptera", "mammals", "reptiles", "vascular_plants")
  plot_legend_pairs<-c("terrestrial", "birds", "butterflies moths", "mammals", "reptiles", "vascular plants")
  
  num_terraqua_headers<-length(terra_biodiv_headers)
}else{
  semivariofolderdir<-paste0(rootsemivariofolderdir,"/","sp_terrestrial")
  resultfolderdir<-paste0(rootresultfolderdir,"/","sp_terrestrial")
  terra_biodiv_headers<-c("terrestrial", "birds_terrestrial", "lepidoptera_terrestrial", 
                          "mammals_terrestrial", "reptiles_terrestrial", "vascular_plants_terrestrial")
  plot_legend_pairs<-c("terrestrial (only)", "birds (T)", "butterflies moths (T)",
                       "mammals (T)","reptiles (T)","vascular plants (T)")

  num_terraqua_headers<-length(terra_biodiv_headers)
}


i_terrabiodiv<-1
for(i_terrabiodiv in 1:num_terraqua_headers){
  terra_header<-terra_biodiv_headers[i_terrabiodiv]
  simcsvfilename<-paste0("spatial_autocorrelation_simulation_records_",terra_header,".csv")
  SimData<-read.csv(paste0(semivariofolderdir,"/",simcsvfilename),header = T)
  num_sim<-dim(SimData)[1]
  
  svresultfilename<-paste0("spatial_autocorrelation_result_",terra_header,".csv")
  svresultDF<-read.csv(paste0(semivariofolderdir,"/",svresultfilename),header = T)
  
  dist_array<-SimData[,"dist"]
  vario_array<-SimData[,"vario"]
  
  #binarize simulation data
  bins<-seq(0,max_semivario_dist,by=max_semivario_dist/200)
  num_bins<-length(bins)-1
  dist_bins<-array(NA,dim=num_bins)
  vario_bins<-array(NA,dim=num_bins)
  for(j in 1:num_bins){
    bool_bin<-dist_array>bins[j]&dist_array<=bins[j+1]
    vario_bins[j]<-mean(vario_array[bool_bin])/2
    dist_bins[j]<-(bins[j]+bins[j+1])/2
  }
  
  x_fit<-dist_bins
  range<-svresultDF$range
  sill<-svresultDF$sill
  nugget<-svresultDF$nugget
  
  vario_sph_fit<-nugget+sill*(1-exp(-1*3*x_fit/range))
  rmse<-sqrt(sum((vario_bins-vario_sph_fit)**2)/num_bins)
  r_sq<-1-sum((vario_bins-vario_sph_fit)**2)/sum((vario_bins-mean(vario_bins))**2)
  
  eval_eq<-substitute(italic(RMSE) == a*","~~italic(R)^2~"="~r2, 
                      list(a = sprintf("%.2f",rmse), 
                           r2 = sprintf("%.2f",r_sq)))
  eval_eq<-as.expression(eval_eq)
  
  #plotting
  plotdata<-data.frame(dist_bins,vario_bins)
  p<-ggplot(plotdata,aes(x=dist_bins)) + theme_classic() +
    geom_point(aes(x = dist_bins, y = vario_bins), color="Wheat4", size=1.6, alpha = 0.6)+
    # geom_smooth(aes(x = dist_bins, y = vario_bins),method='loess',colour="RoyalBlue3",size=1.5,alpha=0.5) +
    geom_line(aes(x = dist_bins, y = vario_sph_fit),colour="RoyalBlue3",size=1.8,alpha=0.9) +
    scale_x_continuous(breaks=c(0,25,50),limits = c(0,52)) +
    scale_y_continuous(breaks=c(0,0.5,1),limits = c(0,1.2)) +
    xlab(expression(italic(h)~" (km)")) + ylab(expression(gamma(italic(h))))+
    # geom_vline(aes(xintercept=0.0), size=0.88, colour="grey51", linetype="dashed",alpha=0.5) +
    # geom_hline(aes(yintercept=0.0), size=0.88, colour="grey51", linetype="dashed",alpha=0.5) +
    # geom_text(x = 2.5, y = 1.0, label = lm_eqn(c(range,sill,nugget)), parse = TRUE,colour="grey21",size=6) + 
    
    #text position for original panel: 0.68, 0.55, 0.42, 0.2 
    geom_text(x = 2.6/4*max_semivario_dist, y = 0.50, label = paste0("range = ",sprintf("%.2f",range)), colour="grey21",size=7) +  
    geom_text(x = 2.6/4*max_semivario_dist, y = 0.40, label = paste0("sill = ",sprintf("%.2f",sill)), colour="grey21",size=7) + 
    geom_text(x = 2.6/4*max_semivario_dist, y = 0.30, label = paste0("nugget = ",sprintf("%.2f",nugget)), colour="grey21",size=7) +
    geom_text(x = 2.2/4*max_semivario_dist, y = 0.15, label = eval_eq, colour="grey21",size=7) +
    geom_text(x = 0.475*max_semivario_dist, y = 1.2, label = plot_legend_pairs[i_terrabiodiv], colour="grey21",size=9) +
    theme(axis.title.y=element_text(angle=90, colour="black",size=24)) +
    theme(axis.title.x=element_text(angle=0, colour="black",size=24)) +
    theme(axis.text.y=element_text(angle=0, colour="grey21",size=20)) +
    theme(axis.text.x=element_text(angle=0, colour="grey21",size=20))

  #write result
  if(dir.exists(resultfolderdir)==FALSE){dir.create(resultfolderdir,recursive = TRUE)}
  
  plotname1<-paste0("corr_plot_variogram_",terra_header,".png")
  ggsave(paste0(resultfolderdir,"/",plotname1),p,width=5, height=6,dpi=900,unit="in")
}
print("ALL DONE.\n")