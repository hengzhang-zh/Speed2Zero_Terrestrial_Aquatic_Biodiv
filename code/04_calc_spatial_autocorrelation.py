# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 08:13:37 2026

@author: Heng Zhang
"""

import os
import numpy as np

from skgstat import models
from scipy.optimize import curve_fit

import src.init as init
import src.processgeotiff4 as ptf

def standardizeRaster(raster,bool_mask,nodata):
    ras_lyr=raster
    lyr_mu=np.mean(ras_lyr[bool_mask])
    lyr_sigma=np.std(ras_lyr[bool_mask])
    ras_lyr_std=(ras_lyr-lyr_mu)/lyr_sigma
    ras_lyr_std[~bool_mask]=nodata
    return ras_lyr_std

#%%
if __name__=="__main__":
    root=r"XXX"
    ras1folderdir=root+os.sep+r"data\biodiv\raster\species_richness_taxa\sp_aquatic"
    ras2folderdir=root+os.sep+r"data\env_var\hydro_map"
    resultfolderdir=root+os.sep+r"result\table\env_var\spatial_autocorrelation_semivario\sp_aquatic"
        
    bool_aquatic=True
    
    # biodiv_headers=["terrestrial","birds_terrestrial","lepidoptera_terrestrial",
    #                 "mammals_terrestrial",
    #                 "reptiles_terrestrial","vascular_plants_terrestrial"]
    biodiv_headers=["all","birds","lepidoptera",
                      "mammals","reptiles","vascular_plants"]
    num_layers=len(biodiv_headers)
    
    pixel_size_scale=0.001
    pixel_area_proj="CH1903+"           #CH1903     WGS1984
    
    num_sim=10000           #10000 for aquatic; 300000 for terrestrial
    max_dist=100         #unit: km
    nodata=65535

#%%
    for raster_id in range(num_layers):
        biodiv_header=biodiv_headers[raster_id]
        
        csvfilename2="spatial_autocorrelation_result_"+biodiv_header+".csv"
        csvfiledir2=resultfolderdir+os.sep+csvfilename2
        if os.path.exists(csvfiledir2):
            print("spatial autocorrelation analysis of %s was finished. \n"%biodiv_header)
            continue
        
        #read base raster layer    
        ras1filedir=ras1folderdir+os.sep+"SDMapCHv1_3_richness_"+biodiv_header+".tif"   
        [ras_sp,driver_sp,geoTrans_sp,proj_sp,nrow_sp,ncol_sp]=ptf.readSingleTiffAsNumpy(ras1filedir,datatype='UInt16')
        
        #read standard layer to find accurate boundary
        ras1filedir=ras1folderdir+os.sep+"SDMapCHv1_3_richness_"+biodiv_headers[0]+".tif"   
        [ras_bound,driver_sp,geoTrans_sp,proj_sp,nrow_sp,ncol_sp]=ptf.readSingleTiffAsNumpy(ras1filedir,datatype='UInt16')
        
        #read river mask layer
        ras2filename="river_pixel_template_fish_sr.tif"
        ras2filedir=ras2folderdir+os.sep+ras2filename
        [ras_mask,driver_mask,geoTrans_mask,proj_mask,nrow_mask,ncol_mask]=ptf.readSingleTiffAsNumpy(ras2filedir,datatype='UInt16')    
        bool_mask=np.zeros([nrow_mask,ncol_mask],dtype=np.bool_)
        bool_mask[ras_mask>0]=True
        
        
        if not bool_aquatic:
            #mask out water pixels
            bool_bound=ras_bound==ras_bound[0,0]
            bool_mask=bool_mask|bool_bound
            bool_mask=~bool_mask
            del bool_bound
        del ras_mask,ras_bound
        
    #%%
        print("scaling raster layer %s... \n"%biodiv_header)
        ras_std=standardizeRaster(ras_sp,bool_mask,nodata)            
        
    #%%
        pix_len=max(abs(geoTrans_sp[1]),abs(geoTrans_sp[5]))
        #simulation starts
        sim_mat=np.zeros([num_sim,6],dtype=np.float32)
        idx=0
        [nrow,ncol]=[nrow_sp,ncol_sp]
        while idx<num_sim:
            p1_x=np.int32(np.random.uniform(0,ncol,1))
            p1_y=np.int32(np.random.uniform(0,nrow,1))
            if bool_mask[p1_y,p1_x]:
                theta=np.random.uniform(0,2*np.pi,1)
                rho=np.random.uniform(0,max_dist/pixel_size_scale/pix_len,1)
                p2_x=np.int32(p1_x+rho*np.cos(theta))
                p2_y=np.int32(p1_y+rho*np.sin(theta))
                if p2_x<0 or p2_x>=ncol or p2_y<0 or p2_y>=nrow or not bool_mask[p2_y,p2_x]:
                    continue
                else:
                    vario=np.square(ras_std[p2_y,p2_x]-ras_std[p1_y,p1_x])
                    dist=np.sqrt(np.square(p2_y-p1_y)+np.square(p2_x-p1_x))*pix_len*pixel_size_scale
                    sim_mat[idx,:]=np.array((p1_y,p1_x,p2_y,p2_x,dist,vario)).ravel()
                    # print("iteration %d / %d"%(idx,num_sim))
                    idx=idx+1
    #%%
        #fitting a variogram
        dist_sep=0.25
        max_dist_fit=50
        num_bins=int(max_dist_fit/dist_sep)
        bins=np.linspace(0,max_dist_fit,num_bins+1)
        dist_bins=np.zeros(num_bins,dtype=np.float32)
        vario_bins=np.zeros(num_bins,dtype=np.float32)
        for i in range(num_bins):
            bool_bin=(sim_mat[:,4]>bins[i])&(sim_mat[:,4]<=bins[i+1])
            vario_bins[i]=np.mean(sim_mat[bool_bin,5])/2
            # dist_bins[i]=(bins[i]+bins[i+1])/2
            dist_bins[i]=np.mean(sim_mat[bool_bin,4])
    
        p0=[np.mean(dist_bins), np.mean(vario_bins), 1]
        cof, cov =curve_fit(models.exponential, dist_bins, vario_bins, p0=p0)
        v_pred=models.exponential(dist_bins, cof[0], cof[1], cof[2])
        rmse=np.sqrt(np.sum(np.square(vario_bins-v_pred))/num_bins)
        r_sq=1-(np.sum(np.square(vario_bins-v_pred)))/(np.sum(np.square(vario_bins-np.mean(vario_bins))))
        spatial_result=np.array([cof[0], cof[1], cof[2],rmse,r_sq])
        spatial_result=spatial_result.reshape([1,5])
        print("%s spatial autocorrelation estimation finished. \n"%biodiv_header)
        print("range: %.3f   sill: %.3f   nugget: %.3f  |  rmse = %.2f    r-squared = %.2f" % (cof[0], cof[1], cof[2],rmse,r_sq))
             
    #%%       
        #write the results
        if not os.path.exists(resultfolderdir):
            os.makedirs(resultfolderdir)
        csvfilename1="spatial_autocorrelation_simulation_records_"+biodiv_header+".csv"
        csvfiledir1=resultfolderdir+os.sep+csvfilename1
        init.writeArrayToCSV(sim_mat,["p1_y","p1_x","p2_y","p2_x","dist","vario"],csvfiledir1)
        
        csvfilename2="spatial_autocorrelation_result_"+biodiv_header+".csv"
        csvfiledir2=resultfolderdir+os.sep+csvfilename2
        init.writeArrayToCSV(spatial_result,["range","sill","nugget","rmse","rsquared"],csvfiledir2)    
    print("ALL DONE.\n")