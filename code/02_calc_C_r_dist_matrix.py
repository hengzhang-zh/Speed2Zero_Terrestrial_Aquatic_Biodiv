# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:09:20 2025

@author: Heng Zhang
"""

import os
import time
import numpy as np
import pandas as pd

import src.init as init
import src.catch_funcs as catch
import src.processgeotiff4 as ptf
import src.model_funcs_terra_aqua as model

#%%
if __name__=="__main__":
    root=r"XXX"
    csvfolderdir=root+os.sep+r"data\env_var\pixel_coords"
    ras1folderdir=r"XXX\data\biodiv\raster\species_richness_taxa\sp_terrestrial"    
    ras2folderdir=root+os.sep+r"data\env_var\hydro_map"
    catchfolderdir=r"F:\S2Z_TerrAqua_Biodiv\data\env_var\catchmask_pix_v01"
    resultfolderdir=root+os.sep+r"result\env_var\catchmask_pix_v01_C_Mat"
    
    csvfilename="riv_pix_CH_example.csv"
    csvfiledir=csvfolderdir+os.sep+csvfilename
    sitePD=init.readCSVasPandas(csvfiledir)
    
    sitePD_Index_start=0                 #0        87823     175646     263469
    sitePD_Index_end=351292               #87823    175646    263469     351292
    
    sitePD=sitePD.iloc[sitePD_Index_start:sitePD_Index_end]
 
    lon_ori_header="CHX_ori"
    lat_ori_header="CHY_ori"
    lon_calib_header="CHX_calib"
    lat_calib_header="CHY_calib"
    site_header="SiteID"
    
    biodiv_headers=["terrestrial","amphibians_terrestrial","bees_terrestrial","birds_terrestrial","bryophytes_terrestrial","coleoptera_terrestrial",
                    "fungi_terrestrial","lepidoptera_terrestrial","lichens_terrestrial","mammals_terrestrial","molluscs_terrestrial","orthoptera_terrestrial",
                    "reptiles_terrestrial","spiders_terrestrial","vascular_plants_terrestrial"]
    biodiv_headers=["all","amphibians","bees","birds","bryophytes","coleoptera","fishes","fungi","lepidoptera","lichens",
                      "mammals","may_stone_caddis_flies","molluscs","odonata","orthoptera","reptiles","spiders","vascular_plants"]
    num_layers=len(biodiv_headers)
    
    siteIDList=list(sitePD[site_header])
    num_sites=len(siteIDList)
       
    decay_func="exp" #exp, sph, matern15, matern25, gaussian
    pixel_size_scale=0.001
    pixel_area_proj="CH1903+"           #CH1903     WGS1984
    num_time_lags=20

#%%
    #read base raster layer
    
    ras1filedirlist=[ras1folderdir+os.sep+"SDMapCHv1_3_richness_"+biodiv_header+".tif" for biodiv_header in biodiv_headers]    
    [ras_sp,driver_sp,geoTrans_sp,proj_sp,nrow_sp,ncol_sp]=ptf.readTiffAsNumpy(ras1filedirlist,datatype='UInt16')
    
    #read river mask layer
    ras2filename="river_pixel_template_fish_sr.tif"
    ras2filedir=ras2folderdir+os.sep+ras2filename
    [ras_mask,driver_mask,geoTrans_mask,proj_mask,nrow_mask,ncol_mask]=ptf.readSingleTiffAsNumpy(ras2filedir,datatype='UInt16')    
    bool_mask=np.zeros([nrow_mask,ncol_mask],dtype=np.bool_)
    bool_mask[ras_mask>0]=True
    
    #mask out water pixels
    for i_lyr in range(num_layers):
        ras_sp[bool_mask,i_lyr]=0
    del ras_mask,bool_mask
    
#%%
    #set up distance list for estimation
    dist_max=400
    dist_min=1
    num_bins=50
    dist_list1=np.linspace(0.1,0.9,9).astype(np.float32)
    dist_list2=np.exp(np.linspace(np.log(dist_min),np.log(dist_max),num_bins)).astype(np.float32)
    dist_list=np.concatenate((dist_list1,dist_list2),axis=0)
    num_dist_sim=len(dist_list)

#%%    
    # apply site location shifting for sites with large manual changes.
    bool_sitelocshift=True
    siteloc_thres=1
    catch_area_thres=5
    
    #shift flow direction map location
    if bool_sitelocshift:
        siteLocShifts=catch.calcShiftVector(sitePD[lon_ori_header],sitePD[lat_ori_header],\
                                            sitePD[lon_calib_header],sitePD[lat_calib_header],\
                                            geoTrans_sp[1])
#%%
    C_mat=np.zeros([num_sites,num_dist_sim,num_layers],dtype=np.float32)
    time_array=np.zeros(num_time_lags,dtype=np.float32)
    for i_site in range(0,num_sites):  

        siteID=siteIDList[i_site]
        if i_site%num_time_lags==0:
            print("computing catchment matrix C(r)...         site no. (%d / %d)"%((i_site+1),num_sites))  

        t1=time.time()        
        catchdistfilename=siteID+"_catchdist.tif"     
        catchdistfiledir=catchfolderdir+os.sep+catchdistfilename
        if not os.path.exists(catchdistfiledir):
            print(catchdistfilename+" does not exist.     continue...")
            C_mat[i_site,:]=-1
            continue      
  
        #read catchment raster layer
        [ras_catch,driver_catch,geoTrans_catch,proj_catch,nrow_catch,ncol_catch]=ptf.readSingleTiffAsNumpy(catchdistfiledir,datatype='Float32')
        ras_catch=ras_catch*pixel_size_scale
        if bool_sitelocshift:
            [shift_x,shift_y,num_pixel_shift]=siteLocShifts[i_site,:]
            if num_pixel_shift>siteloc_thres:
                geoTrans_catch=catch.shiftRasterGeoTrans(shift_x, shift_y, geoTrans_catch)
        
        #iteration through raster layers
        for i_lyr in range(num_layers):
            C_rec=model.calc_C_row(ras_sp[:,:,i_lyr],ras_catch,geoTrans_sp,geoTrans_catch,nrow_catch,ncol_catch,\
                                   dist_list,num_dist_sim,decay_func,pixel_size_scale,pixel_area_proj)
            C_mat[i_site,:,i_lyr]=C_rec

        #calculate remaining time for computation
        t2=time.time() 
        if i_site%num_time_lags==0:
            time_array[i_site%num_time_lags]=t2-t1
            proc_time_avg=np.sum(time_array)/np.sum(time_array!=0)
            proc_time_remain=(num_sites-i_site-1)*proc_time_avg
            num_days=np.floor(proc_time_remain/86400)
            proc_time_res=proc_time_remain-num_days*86400
            num_hours=np.floor(proc_time_res/3600)
            proc_time_res=proc_time_res-num_hours*3600
            num_mins=np.floor(proc_time_res/60)
            print("finished. remaining computational time:  %d day, %d hour, %d min  "%(num_days,num_hours,num_mins))
        
#%%         
    #write C matrix
    dirto=resultfolderdir
    if not os.path.exists(dirto):
        os.makedirs(dirto) 
    for i_lyr in range(num_layers):
        C_mat_layer=C_mat[:,:,i_lyr]
        csv_headers=["D_ind_"+str(i_dist+1).zfill(3) for i_dist in range(num_dist_sim)]    
        csvfilename1="terra_effect_matrix_"+biodiv_headers[i_lyr]+"_decay_func_"+decay_func+".csv"
        csvfiledir1=dirto+os.sep+csvfilename1
        
        Array=C_mat_layer
        save=sitePD
        for i in range(Array.shape[1]):
            pdtmp=pd.DataFrame({csv_headers[i]:Array[:,i]})
            save=pd.concat([save,pdtmp],axis=1)
        save.to_csv(csvfiledir1,index=False,header=True)
        
        init.writeArrayToCSV(C_mat_layer,csv_headers,csvfiledir1)
    print("ALL DONE.")    
    