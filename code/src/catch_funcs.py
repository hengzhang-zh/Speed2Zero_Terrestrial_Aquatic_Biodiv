# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:33:10 2023

@author: Heng Zhang
"""

import os
import copy
import numpy as np
from PIL import Image, ImageDraw


import src.processgeotiff4 as ptf

#generate destined points (water may flow through) based on given sampling points
def genDstPoints(point_coord,GeoTransform,radius_dist=100):
    [pX, pY] = ptf.world2Pixel(GeoTransform, point_coord[0], point_coord[1])  
    p_size=max(abs(GeoTransform[1]),abs(GeoTransform[5]))
    radius_num=int(radius_dist/p_size)
    dst_points=[]
    for j in range(pY-radius_num,pY+radius_num+1):
        for i in range(pX-radius_num,pX+radius_num+1):
            dist=np.sqrt(np.square(j-pY)+np.square(i-pX))
            if dist<=radius_num:
                dst_points.append([j,i])
    return dst_points

#initialize catchment mask and catchment distance data
def initCatchDist(dst_points,nrow,ncol,default_dist_value):
    catchdist=np.zeros([nrow,ncol],dtype=np.float32)
    for dp in dst_points:
        catchdist[dp[0],dp[1]]=default_dist_value   #here -1 stands for the outlet of catchment
    return catchdist

def calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform):
 	#p1_lonlat_x = (GeoTransform[0] + p1_x*GeoTransform[1])*np.pi / 180
 	p1_lonlat_y = (GeoTransform[3] + p1_y*GeoTransform[5])*np.pi / 180
 	#p2_lonlat_x = (GeoTransform[0] + p2_x*GeoTransform[1])*np.pi / 180
 	p2_lonlat_y = (GeoTransform[3] + p2_y*GeoTransform[5])*np.pi / 180
 	
 	dlat = (p2_y-p1_y)*GeoTransform[1]*np.pi/180
 	dlon = (p2_x-p1_x)*GeoTransform[5]*np.pi/180
 	a = np.sin(dlat / 2)*np.sin(dlat / 2) + np.cos(p1_lonlat_y)*np.cos(p2_lonlat_y) * np.sin(dlon / 2)*np.sin(dlon / 2)
 	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
 	return 6371*c

def calcPixelAreaMat(nrow,ncol,GeoTransform):
    pixel_area_rows=np.zeros(nrow,dtype=np.float32)
    for j in range(nrow):
        [p1_x,p1_y,p2_x,p2_y]=[0,j,0,j+1]        
        dy=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)
        [p1_x,p1_y,p2_x,p2_y]=[0,j,1,j]        
        dx=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)
        pixel_area_rows[j]=dy*dx
    A=np.zeros([nrow,ncol],dtype=np.float32)
    for i in range(ncol):
        A[:,i]=pixel_area_rows[:]
    return A

def calcPixelDXDYArray(nrow,ncol,GeoTransform):
    DX=np.zeros(nrow,dtype=np.float32)
    DY=np.zeros(ncol,dtype=np.float32)    
    for j in range(nrow):     
        [p1_x,p1_y,p2_x,p2_y]=[0,j,1,j]        
        DX[j]=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)  
    for i in range(ncol):
        [p1_x,p1_y,p2_x,p2_y]=[i,0,i,1]        
        DY[i]=calcLonLatDist(p1_x,p1_y,p2_x,p2_y,GeoTransform)
    return [DX,DY]

def readFDBlocksFromCoord_HG(catch_rec,lon_splits,lat_splits,catch_bbox,\
                         FDfolderdir,FD_filename_predix,FD_filename_postfix):
    
    #find the row and column of blocks of global FD map
    catch_block_h_lower=np.where(lon_splits>catch_bbox[0])[0][0]-1
    catch_block_h_upper=np.where(lon_splits>catch_bbox[1])[0][0]-1
    catch_block_v_upper=np.where(lat_splits<catch_bbox[3])[0][0]-1
    catch_block_v_lower=np.where(lat_splits<catch_bbox[2])[0][0]-1

    FD_block_list=[]
    for i_lon in range(catch_block_h_lower,catch_block_h_upper+1):
        for i_lat in range(catch_block_v_upper,catch_block_v_lower+1):
            FD_filename=FD_filename_predix+"h"+str(i_lon*2).zfill(2)+"v"+str(i_lat*2).zfill(2)+FD_filename_postfix
            FD_filedir=FDfolderdir+os.sep+FD_filename
            if os.path.exists(FD_filedir):
                FD_block_list.append(FD_filedir)
    [geoTrans_ori,ras_bbox,nrow_ori,ncol_ori]=ptf.getBlockRasterExtent(FD_block_list)
   
    FD=np.zeros([nrow_ori,ncol_ori],dtype=np.int32)
    for i in range(len(FD_block_list)):
        blockmapfiledir=FD_block_list[i]
        [ras_tmp,driver_tmp,geoTrans_tmp,proj_tmp,nrow_tmp,ncol_tmp]=ptf.readTiffAsNumpy([blockmapfiledir],datatype='Int32')
        ras_tmp=ras_tmp[:,:,0]
        row_start=int(0.5+(geoTrans_tmp[3]-geoTrans_ori[3])/geoTrans_tmp[5])
        row_end=row_start+nrow_tmp
        col_start=int(0.5+(geoTrans_tmp[0]-geoTrans_ori[0])/geoTrans_tmp[1])
        col_end=col_start+ncol_tmp
        
        FD[row_start:row_end,col_start:col_end]=ras_tmp
        
    #clip FD to match buffer of sampling sites
    [i_col_start, i_row_start]=ptf.world2Pixel(geoTrans_ori, catch_bbox[0], catch_bbox[3])
    [i_col_end, i_row_end]=ptf.world2Pixel(geoTrans_ori, catch_bbox[1], catch_bbox[2])
    
    FD_clip=np.ascontiguousarray(FD[i_row_start:i_row_end,i_col_start:i_col_end])
    FD_clip[np.isnan(FD_clip)]=0
    FD_clip[FD_clip<0]=0
    
    
    geoTrans=copy.deepcopy(geoTrans_ori)
    geoTrans[0]=geoTrans_ori[0]+geoTrans_tmp[1]*i_col_start
    geoTrans[3]=geoTrans_ori[3]+geoTrans_tmp[5]*i_row_start
    
    nrow=i_row_end-i_row_start
    ncol=i_col_end-i_col_start
        
    driver=driver_tmp
    proj=proj_tmp
    return [FD_clip,driver,geoTrans,proj,nrow,ncol]

def readFDBlocksFromCoord_HS(catch_rec,lon_splits,lat_splits,catch_bbox,\
                         FDfolderdir,FD_filename_predix,FD_filename_postfix):
    
    #find the row and column of blocks of global FD map
    catch_block_h_lower=np.where(lon_splits>catch_bbox[0])[0][0]-1
    catch_block_h_upper=np.where(lon_splits>catch_bbox[1])[0][0]-1
    catch_block_v_upper=np.where(lat_splits<catch_bbox[3])[0][0]-1
    catch_block_v_lower=np.where(lat_splits<catch_bbox[2])[0][0]-1

    FD_block_list=[]
    for i_lon in range(catch_block_h_lower,catch_block_h_upper+1):
        for i_lat in range(catch_block_v_upper,catch_block_v_lower+1):
            lon_str="e"+str(int(lon_splits[i_lon])).zfill(3) if lon_splits[i_lon] >=0 else "w"+str(-1*int(lon_splits[i_lon])).zfill(3)
            lat_str="n"+str(int(lat_splits[i_lat+1])).zfill(2) if lat_splits[i_lat+1] >=0 else "s"+str(-1*int(lat_splits[i_lat+1])).zfill(2)
            FD_filename=FD_filename_predix+lat_str+lon_str+FD_filename_postfix
            FD_filedir=FDfolderdir+os.sep+FD_filename
            if os.path.exists(FD_filedir):
                FD_block_list.append(FD_filedir)
    [geoTrans_ori,ras_bbox,nrow_ori,ncol_ori]=ptf.getBlockRasterExtent(FD_block_list)
   
    FD=np.zeros([nrow_ori,ncol_ori],dtype=np.uint8)
    for i in range(len(FD_block_list)):
        blockmapfiledir=FD_block_list[i]
        [ras_tmp,driver_tmp,geoTrans_tmp,proj_tmp,nrow_tmp,ncol_tmp]=ptf.readTiffAsNumpy([blockmapfiledir],datatype='UInt8')
        ras_tmp=ras_tmp[:,:,0]
        row_start=int(0.5+(geoTrans_tmp[3]-geoTrans_ori[3])/geoTrans_tmp[5])
        row_end=row_start+nrow_tmp
        col_start=int(0.5+(geoTrans_tmp[0]-geoTrans_ori[0])/geoTrans_tmp[1])
        col_end=col_start+ncol_tmp
        
        FD[row_start:row_end,col_start:col_end]=ras_tmp
        
    #clip FD to match buffer of sampling sites
    [i_col_start, i_row_start]=ptf.world2Pixel(geoTrans_ori, catch_bbox[0], catch_bbox[3])
    [i_col_end, i_row_end]=ptf.world2Pixel(geoTrans_ori, catch_bbox[1], catch_bbox[2])
    
    FD_clip=np.ascontiguousarray(FD[i_row_start:i_row_end,i_col_start:i_col_end])
    FD_clip[np.isnan(FD_clip)]=0
    FD_clip[FD_clip<0]=0
    
    
    geoTrans=copy.deepcopy(geoTrans_ori)
    geoTrans[0]=geoTrans_ori[0]+geoTrans_tmp[1]*i_col_start
    geoTrans[3]=geoTrans_ori[3]+geoTrans_tmp[5]*i_row_start
    
    nrow=i_row_end-i_row_start
    ncol=i_col_end-i_col_start
        
    driver=driver_tmp
    proj=proj_tmp
    return [FD_clip,driver,geoTrans,proj,nrow,ncol]

def clipRasterBBOX(ras1,ras2,geoTrans):
    cd_row_sum=np.sum(ras2,axis=1)
    cd_col_sum=np.sum(ras2,axis=0)
    row_start=np.where(cd_row_sum>0)[0][0]
    row_end=np.where(cd_row_sum>0)[0][-1]
    col_start=np.where(cd_col_sum>0)[0][0]
    col_end=np.where(cd_col_sum>0)[0][-1]
    
    ras1=np.ascontiguousarray(ras1[row_start:row_end,col_start:col_end])
    ras2=np.ascontiguousarray(ras2[row_start:row_end,col_start:col_end])
    
    geoTrans[0]=geoTrans[0]+geoTrans[1]*col_start
    geoTrans[3]=geoTrans[3]+geoTrans[5]*row_start
    
    nrow=int(row_end-row_start)
    ncol=int(col_end-col_start)  
    return [ras1,ras2,geoTrans,nrow,ncol]

def calcShiftVector(ref_x,ref_y,tar_x,tar_y,pixel_reso):
    shift_x=tar_x-ref_x
    shift_y=tar_y-ref_y
    coord_shift=np.sqrt(shift_x*shift_x+shift_y*shift_y)
    num_pixel_shift=coord_shift/pixel_reso
    siteShifts=np.array([shift_x,shift_y,num_pixel_shift]).transpose()
    return siteShifts

def shiftRasterGeoTrans(shift_x,shift_y,geoTrans):
    geoTrans_shift=copy.deepcopy(geoTrans)
    pixel_shift_x=round(shift_x/geoTrans[1])
    pixel_shift_y=round(shift_y/geoTrans[5])
    
    geoTrans_shift[0]=geoTrans[0]+geoTrans[1]*pixel_shift_x
    geoTrans_shift[3]=geoTrans[3]+geoTrans[5]*pixel_shift_y
    return geoTrans_shift

def clipRaster(ras,geoTrans_ras,geoTrans_clip,nrow,ncol):
    #clip FD to match buffer of sampling sites
    [i_col_start, i_row_start]=ptf.world2Pixel(geoTrans_ras, geoTrans_clip[0], geoTrans_clip[3])
    [i_col_end, i_row_end]=ptf.world2Pixel(geoTrans_ras, geoTrans_clip[0]+ncol*geoTrans_clip[1], geoTrans_clip[3]+nrow*geoTrans_clip[5])
    
    geoTrans_out=copy.deepcopy(geoTrans_ras)
    geoTrans_out[0]=geoTrans_ras[0]+geoTrans_ras[1]*i_col_start
    geoTrans_out[3]=geoTrans_ras[3]+geoTrans_ras[5]*i_row_start
    
    ras_out=np.ascontiguousarray(ras[i_row_start:i_row_end,i_col_start:i_col_end])

    return [ras_out,geoTrans_out]

def resizeImage(srcImage,dst_shape):
    img=Image.fromarray(srcImage)
    # dstImage=np.array(img.resize(dst_shape,Image.Resampling.NEAREST),dtype=np.float32)
    dstImage=np.array(img.resize(dst_shape,Image.NEAREST),dtype=np.float32)
    return dstImage

def cvtLCTypes(ras_LC,LC_CVT,LC_header_in,LC_header_out,rm_list_out=[]):
    LC_types_in=list(LC_CVT[LC_header_in])
    LC_types_out=list(LC_CVT[LC_header_out])
    ras_LC_cvt=np.zeros_like(ras_LC,dtype=np.int16)+999  #zero is default (No Data)
    for i_type in range(len(LC_types_in)):
        lc_type_in=LC_types_in[i_type]
        lc_type_out=LC_types_out[i_type]
        if lc_type_out in rm_list_out:
            continue
        ras_LC_cvt[ras_LC==lc_type_in]=lc_type_out
    return ras_LC_cvt

def extRiverPixels(FA,FA_thres):
    bool_RivPix=np.zeros_like(FA,dtype=np.bool8)
    bool_RivPix[FA>=FA_thres]=True
    river_pixels=np.matrix(np.int32(np.argwhere(bool_RivPix==True)))
    return river_pixels

def calcDistCircleMask_HVS(p_lon,p_lat,max_dist,geoTrans):
    radius_pixel=int(max_dist/(geoTrans[1]*111))
    [nrow,ncol]=[2*radius_pixel+1,2*radius_pixel+1]
    geoTrans_catch_disc=copy.deepcopy(geoTrans)
    geoTrans_catch_disc[0]=p_lon-geoTrans_catch_disc[1]*(radius_pixel+1)
    geoTrans_catch_disc[3]=p_lat-geoTrans_catch_disc[5]*(radius_pixel+1)
        
    [DX,DY]=calcPixelDXDYArray(nrow,ncol,geoTrans_catch_disc)
    dROW=np.zeros([nrow,ncol],dtype=np.float32)
    dCOL=np.zeros([nrow,ncol],dtype=np.float32)
    for i_row in range(nrow):
        i_start=min(i_row,radius_pixel)
        i_end=max(i_row,radius_pixel)
        dROW[i_row,:]=np.sum(DY[i_start:i_end])
    for i_col in range(ncol):
        i_start=min(i_col,radius_pixel)
        i_end=max(i_col,radius_pixel)
        dCOL[:,i_col]=np.sum(DX[i_start:i_end])
    
    catch_FD_disc=np.sqrt(dROW*dROW+dCOL*dCOL)
    catch_FD_disc[catch_FD_disc>max_dist]=0    

    catch_FD_disc_bbox=np.zeros(4, dtype=np.float32)
    catch_FD_disc_bbox[0]=max(-179.9,geoTrans_catch_disc[0])
    catch_FD_disc_bbox[1]=min(179.9,geoTrans_catch_disc[0]+(2*radius_pixel+1)*geoTrans_catch_disc[1])
    catch_FD_disc_bbox[2]=max(-89.9,geoTrans_catch_disc[3]+(2*radius_pixel+1)*geoTrans_catch_disc[5])
    catch_FD_disc_bbox[3]=min(89.9,geoTrans_catch_disc[3])
    
    [catch_FD,geoTrans_FD,nrow_FD,ncol_FD]=ptf.clipRasBBOX(catch_FD_disc,geoTrans_catch_disc,catch_FD_disc_bbox)
    return [catch_FD,geoTrans_FD,nrow_FD,ncol_FD]