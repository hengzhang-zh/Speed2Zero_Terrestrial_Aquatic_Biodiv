# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:35:00 2023

@author: Heng Zhang (heng.zhang@eawag.ch; hengzhang.zhh@gmail.com) at Prof. Florian Altermatt Lab (https://www.altermattlab.ch/)
@Swiss Federal Institute of Aquatic Science and Technology (EAWAG / ETH Domain) & University of Zurich (UZH). 
@Please cite the paper below when using any part of this code/project. 

Heng Zhang, et al., Terrestrial land cover shapes fish diversity across a major subtropical basin. 
BioRxiv link: 

"""

#required packages: gdal & pycuda
#please note that the catchment computing only runs on a NVIDIA GPU. 
#please translate the CUDA kernal code to OpenCL/C++ if you would like to use an AMD GPU or even CPUs

import os
import time
import copy
import numpy as np

#CUDA environment
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

#local functions
import src.init as init
import src.catch_funcs as catch
import src.processgeotiff4 as ptf


#%%
if __name__=="__main__":
    #please specify the local file directory
    root=r"XXX"                             
    hydshfolderdir=root+os.sep+r"data\env_var\hydro_map"        
    biodivfolderdir=r"data\env_var\pixel_coords"                        
    resultdirto=root+os.sep+r"data\catchmask_pix_v01"         
    cufile=root+os.sep+r"code\src\calcFlowDist_HS.cu"

    #read biodiversity data with coordinates
    csvfilename="riv_pix_CH_example.csv"
    csvfiledir=biodivfolderdir+os.sep+csvfilename
    sitePD=init.readCSVasPandas(csvfiledir)
    
    sitePD_Index_start=0                 #0        87823     175646     263469
    sitePD_Index_end=351292               #87823    175646    263469     351292
    
    sitePD=sitePD.iloc[sitePD_Index_start:sitePD_Index_end]
    
    #read flow direction (FD) raster 
    FD_filename="topo_alti3d_dem_mean_flow_direct.tif"      
    FD_filedir=hydshfolderdir+os.sep+FD_filename

    #read the flow direction map layer: please convert the FD raster to np.uint8 format
    [FD,driver,geoTrans_FD,proj,nrow_FD,ncol_FD]=ptf.readTiffAsNumpy([FD_filedir],datatype='UInt8')
    FD=FD[:,:,0]

    radius_pix=0.5                                                               #radius in pixel for flow outlets (sampling sites)
    default_dist_value=geoTrans_FD[1]*radius_pix                           #default flow distance value for the outlet pixels
    max_trace_steps=int(400/0.002)                                            #maximal tracing steps in the catchment computation (see cufile)

    bool_clip_result=True
    max_clip_dist=int(300/0.001)                                                          #unit: km/0.001
    
    lon_header="CHX_calib"                                                #header for longitudes of calibrated sites
    lat_header="CHY_calib"                                                #header for latitudes of calibrated sites

    siteIDList=list(sitePD["SiteID"])
    num_sites=len(siteIDList)
    # siteIDList=np.linspace(0,num_sites-1,num_sites).astype(np.int32)
#%%
    #CUDA implementation: read cu file and compile the CUDA kernel code
    #please set the correct flow direction vector (var: dirvecs) in the cu file.
    with open(cufile,"rt") as f:
        cu_src = f.read()
    
    mod = SourceModule(cu_src)
    func = mod.get_function("calcFlowDist")
    #%%
    #iteration: calculate catchment for each site    
    DX=np.ones(nrow_FD,dtype=np.float32)*np.abs(geoTrans_FD[5])
    DY=np.ones(ncol_FD,dtype=np.float32)*np.abs(geoTrans_FD[1])    
    DX=gpuarray.to_gpu(DX)
    DY=gpuarray.to_gpu(DY)
    FD=gpuarray.to_gpu(FD)
    num_time_lags=25
    time_array=np.zeros(num_time_lags,dtype=np.float32)
    for i_site in range(0,num_sites):
        t1=time.time()
        siteID=siteIDList[i_site]
        print("computing catchment...         site no. (%d / %d)"%((i_site+1),num_sites))  
        
        catchdistfilename=siteID+"_catchdist.tif"
        catchdistfiledir=resultdirto+os.sep+catchdistfilename  
        if os.path.exists(catchdistfiledir):
            print("already finished. continue...    \n")
            continue
        
        #update the geoinfo of catchment map (the geoinfo may change in the clipping step below)
        [nrow,ncol]=[nrow_FD,ncol_FD]       
        geoTrans=np.float32(copy.deepcopy(geoTrans_FD))
        site_coord=[sitePD.loc[i_site+sitePD_Index_start,lon_header],sitePD.loc[i_site+sitePD_Index_start,lat_header]]        #get the coordinate of sampling site
        dst_points=catch.genDstPoints(site_coord,geoTrans,radius_dist=geoTrans[1]*radius_pix)
        if len(dst_points)==0 or np.min(dst_points)<0:
            print("site out of catchment boundary! computation for next site continues.\n")
            continue
        
        catchdist=catch.initCatchDist(dst_points,nrow,ncol,default_dist_value)
        catchdist=gpuarray.to_gpu(catchdist)
        
        #%%    
        #specify the block & grid sizes in CUDA computing
        BLOCKDIM=16
        blockSize=(BLOCKDIM,BLOCKDIM,1)
        bx=int((ncol+BLOCKDIM-1)/BLOCKDIM)
        by=int((nrow+BLOCKDIM-1)/BLOCKDIM)
        gridSize=(bx,by,1)
        
        #calculate catchment for sampling site on NVIDIA GPU
        #max_trace_steps, nrow, ncol, etc. need to be transferred to Int32 type to match CUDA kernel functions
        #please check "__global__ void calcFlowDist()" function for corresponding variable types.
        func(catchdist,FD,DX,DY,\
             np.float32(default_dist_value),np.int32(max_trace_steps),np.int32(nrow),np.int32(ncol),\
                 block=blockSize,grid=gridSize)
        cuda.Context.synchronize()      
        catchdist=catchdist.get()
          
        #%%
        #clip catchment map with flow distance with max_clip_dist
        if bool_clip_result:
            catchdist[catchdist>max_clip_dist]=0
            [catchdist,geoTrans,nrow,ncol]=ptf.clipRas2Rect(catchdist,geoTrans)
        
        #write catchment map to local directory
        catchdistfilename=siteID+"_catchdist.tif"
        if not os.path.exists(resultdirto):
            os.makedirs(resultdirto)
        catchdistfiledir=resultdirto+os.sep+catchdistfilename            
        # [driver,proj]=ptf.createTIFFDriverFromEPSG(4326)                       #generate the driver of TIFF raster and the projection of WGS_1984 (lon-lat format)
        ptf.writeNumpyToTiff(catchdist,driver,geoTrans,proj,nrow,ncol,-9999,catchdistfiledir,datatype='Float32')
        t2=time.time()  
        time_array[i_site%num_time_lags]=t2-t1
        print("finished!                      time = %.2f s"%(t2-t1))  
        
        #calculate remaining time for computation
        proc_time_avg=np.sum(time_array)/np.sum(time_array!=0)
        proc_time_remain=(num_sites-i_site-1)*proc_time_avg
        num_days=np.floor(proc_time_remain/86400)
        proc_time_res=proc_time_remain-num_days*86400
        num_hours=np.floor(proc_time_res/3600)
        proc_time_res=proc_time_res-num_hours*3600
        num_mins=np.floor(proc_time_res/60)
        print("remaining computational time:  %d day, %d hour, %d min  \n"%(num_days,num_hours,num_mins))
    print("ALL DONE.\n")