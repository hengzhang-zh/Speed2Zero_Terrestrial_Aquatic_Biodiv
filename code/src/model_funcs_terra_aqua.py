# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:58:24 2025

@author: Heng Zhang
"""

import os
import copy
import numpy as np
import pandas as pd
import scipy.optimize as opt

#the following three modules are only used in species-level modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

import statsmodels.api as sm

from imblearn.over_sampling import SMOTE 

import src.catch_funcs as catch
import src.processgeotiff4 as ptf


def S_LC_exp(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*np.exp(-1*alpha*FD_vector)
    return np.sum(s)

def S_LC_sph(FD_vector,theta1,theta2,pixel_areas):
    bool_range=FD_vector<=theta1
    s=np.zeros_like(FD_vector)
    s[bool_range]=pixel_areas[bool_range]*theta2*(1-1.5*(FD_vector[bool_range]/theta1)+0.5*(FD_vector[bool_range]/theta1)**3)
    return np.sum(s)

def S_LC_Gaussian(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*np.exp(-1*alpha*alpha*FD_vector*FD_vector)
    return np.sum(s)

def S_LC_Matern15(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*(1+np.sqrt(3)*alpha*FD_vector)*np.exp(-1*np.sqrt(3)*alpha*FD_vector)
    return np.sum(s)

def S_LC_Matern25(FD_vector,alpha,beta,pixel_areas):
    s=pixel_areas*beta*(1+np.sqrt(5)*alpha*FD_vector+5/3*alpha*alpha*FD_vector*FD_vector)*np.exp(-1*np.sqrt(5)*alpha*FD_vector)
    return np.sum(s)

def calcOptTSSDiv(p_a,p_a_prob,intv=0.01):
    divs=np.linspace(intv,1,int(1/intv),dtype=np.float32)
    TSSs=np.zeros_like(divs,dtype=np.float32)
    p_prob=p_a_prob[:,1]
    for i in range(int(1/intv)):
        div=divs[i]
        bool_p=1*(p_prob>div)
        a=np.sum((p_a==1)&(bool_p==1))
        b=np.sum((p_a==0)&(bool_p==1))
        c=np.sum((p_a==1)&(bool_p==0))
        d=np.sum((p_a==0)&(bool_p==0))
        sensi=a/(a+c)
        speci=d/(b+d)
        TSSs[i]=sensi+speci-1
    idx_max=np.argmax(TSSs)
    return [divs[idx_max],TSSs[idx_max]]

def calc_neg2LogLik(params,X,y,num_sites,num_LULC_types):
    a=params[0]
    b=params[1+num_LULC_types]
    sigma2=params[2+num_LULC_types]
    V=params[1:(1+num_LULC_types)]

    mu=a*X[:,0]+b*X[:,1+num_LULC_types]+np.dot(X[:,1:(1+num_LULC_types)],V)
    resid=y-mu
    LL=-0.5*num_sites*(np.log(sigma2)+np.log(2*np.pi))-0.5*np.sum(resid*resid/sigma2)
    return -2*LL

def calc_C_rec(ras_value_catch,ras_catch,A_catch,dist_list,num_dist_sim,decay_func):   
    C_rec=np.zeros(num_dist_sim,dtype=np.float32)
    
    FD_mask=np.zeros_like(ras_catch,dtype=np.bool_)    
    FD_mask[ras_catch>0]=True
    
    ras_value_vector=ras_value_catch[FD_mask]
    FD_vector=ras_catch[FD_mask]
    pixel_areas=A_catch[FD_mask]
    
    for i_dist in range(num_dist_sim):
        dist=dist_list[i_dist]            
        if decay_func=="exp":
            alpha=3/dist
            beta=alpha
            S_lc=S_LC_exp(FD_vector,alpha,beta,pixel_areas*ras_value_vector)
        elif decay_func=="sph":
            theta1=dist
            theta2=(8/3/theta1)
            S_lc=S_LC_sph(FD_vector,theta1,theta2,pixel_areas*ras_value_vector)
        elif decay_func=="gaussian":
            alpha=3/dist
            beta=2*np.sqrt(alpha*alpha/np.pi/3)
            S_lc=S_LC_Gaussian(FD_vector,alpha/np.sqrt(3),beta,pixel_areas*ras_value_vector)
        elif decay_func=="matern15":
            alpha=3/dist
            beta=(np.sqrt(3)*alpha)/2
            S_lc=S_LC_Matern15(FD_vector,alpha,beta,pixel_areas*ras_value_vector)
        elif decay_func=="matern25":
            alpha=3/dist
            beta=(3*np.sqrt(5)*alpha)/8
            S_lc=S_LC_Matern15(FD_vector,alpha,beta,pixel_areas*ras_value_vector)
        else:
            print("decay function method error!")
            
        C_rec[i_dist]=S_lc
    return C_rec        

def calc_C_row(ras_value,ras_catch,geoTrans_value,geoTrans_catch,nrow_catch,ncol_catch,\
               dist_list,num_dist_sim,decay_func,pixel_size_scale,pixel_area_proj):    
    #clip raster layer map
    [ras_value_catch,geoTrans_value_catch]=catch.clipRaster(ras_value,geoTrans_value,geoTrans_catch,nrow_catch,ncol_catch)         
    ras_catch_dim=ras_value_catch.shape
    if not np.sum(ras_value_catch):
        C_rec=np.zeros([1,num_dist_sim],dtype=np.float32)-1
    else:
        if pixel_area_proj=="WGS1984":
            A_catch=catch.calcPixelAreaMat(ras_catch_dim[0],ras_catch_dim[1],geoTrans_value_catch)*pixel_size_scale*pixel_size_scale
        elif pixel_area_proj=="CH1903+":
            A_catch=np.ones([ras_catch_dim[0],ras_catch_dim[1]],dtype=np.float32)*\
                np.abs(geoTrans_value_catch[1]*geoTrans_value_catch[5])*pixel_size_scale*pixel_size_scale
        ras_catch=catch.resizeImage(ras_catch,[ras_catch_dim[1],ras_catch_dim[0]])
        #calculate C(r) row vector for one record
        C_rec=calc_C_rec(ras_value_catch,ras_catch,A_catch,dist_list,num_dist_sim,decay_func)
    return C_rec

def calc_opt_params(C_mat,biodiv,env_var,dist_list,num_dist_sim,num_LULC_types,bool_B0_sep):
    LULC_values=np.zeros([num_dist_sim,num_LULC_types+5])    
    #remove all-zero records
    bool_site=C_mat[0,:,0]!=-1
    bool_biodiv=~np.isnan(biodiv).ravel()
    bool_site=bool_site&bool_biodiv
    #if no value in biodiv & LULC_site vector
    if np.sum(bool_site)==0:
        print("no value in biodiv vector...  quit computation.\n")
        return LULC_values
    
    C_mat=np.ascontiguousarray(C_mat[:,bool_site,:])
    biodiv=biodiv[bool_site,:]
    env_var=env_var[bool_site,:]
    num_sites=np.sum(bool_site)
    
    #calculate num_LULC_exist
    bool_LULC=np.sum(C_mat[num_dist_sim-1,:,1:(num_LULC_types+1)],axis=0)>0
    num_LULC_exist=np.sum(bool_LULC)
    
    biodiv_mean=np.mean(biodiv)
    TSS=np.sum((biodiv-biodiv_mean)**2)
    
    biodiv_B0=np.zeros([num_sites,1],dtype=np.float32)   
        
    # B0 separation: to estimate B0 before other parameter estimations
    if bool_B0_sep:        
        C_B0=np.zeros([num_sites,2])
        C_B0[:,0]=1
        C_B0[:,1]=env_var.flatten()
        lm_model_B0=np.linalg.lstsq(C_B0, biodiv, rcond=None)
        lm_coefs_B0=lm_model_B0[0]       
        biodiv_B0=np.dot(C_B0,lm_coefs_B0)
    else:
        biodiv_B0=np.zeros([num_sites,1])     

    #optimization
    for i_dist in range(num_dist_sim):#%%
        dist=dist_list[i_dist]              
        C=np.ascontiguousarray(C_mat[i_dist,:,:])
        bool_LULC_dist=np.sum(C[:,1:(num_LULC_types+1)],axis=0)>0
        num_LULC_dist_exist=np.sum(bool_LULC_dist)
        
        #step 6: regress C and biodiv to estimate LC_Values (OLS)
        lm_model=np.linalg.lstsq(C, biodiv-biodiv_B0, rcond=None)
        lm_coefs=lm_model[0]
        biodiv_pred=np.dot(C,lm_coefs)+biodiv_B0
        resid=biodiv-biodiv_pred
        RSS=np.sum(resid*resid)  
        sigma2=np.sum(resid*resid)/num_sites
        coefs=lm_coefs[:,0]
        neg2LL=num_sites*(np.log(sigma2)+np.log(2*np.pi))+np.sum(resid*resid/sigma2)

        # #step 6: estimate parameters using maximum likelihood estimation (MLE)
        # params=np.concatenate((np.array([30,3,-2]),np.repeat(2, num_LULC_types)),axis=0)
        # # params=params+np.random.uniform(low=-0.1,high=0.1,size=num_LULC_types+3)
        # # params[0:(num_LULC_types+2)]=coefs 
        # # params[num_LULC_types+2]=sigma2
        # y=biodiv-biodiv_B0
        # y=np.float32(y.flatten())     
        # opt_result=opt.minimize(calc_neg2LogLik, params, args=(C,y,num_sites,num_LULC_types), method='L-BFGS-B')             
        # mle_coefs=opt_result.x[0:(2+num_LULC_types)]
        # mle_fvalue=opt_result.fun
        # neg2LL=mle_fvalue
        # biodiv_pred=np.dot(C,mle_coefs)+biodiv_B0.flatten()
        # RSS=np.sum((biodiv_pred-biodiv.flatten())**2)
        # coefs=mle_coefs
        
        if bool_B0_sep:
            coefs[0]=lm_coefs_B0[0,0]
            coefs[num_LULC_types+1]=lm_coefs_B0[1,0]     
            
        adj_r2=1-(RSS/(num_sites-num_LULC_dist_exist-2-1))/(TSS/(num_sites-1))
        LULC_values[i_dist,(num_LULC_types+2-len(coefs)):(num_LULC_types+2)]=coefs      
        LULC_values[i_dist,num_LULC_types+2]=dist
        LULC_values[i_dist,num_LULC_types+3]=adj_r2
        LULC_values[i_dist,num_LULC_types+4]=neg2LL
        print("MLE estimation no. %d / %d (dist = %.1f km),  r-sq = %.4f, -2LL = %.2f\n"%(i_dist,num_dist_sim,dist,adj_r2,neg2LL))
    return LULC_values

#fEnv is the equation proposed by Pascal Niklaus
def calc_opt_params_fEnv(C_mat,biodiv,env_var,dist_list,num_dist_sim,num_LULC_types,bool_B0_sep):
    LULC_values=np.zeros([num_dist_sim,num_LULC_types+5])    
    #remove all-zero records
    bool_site=C_mat[0,:,0]!=-1
    bool_biodiv=~np.isnan(biodiv).ravel()
    bool_site=bool_site&bool_biodiv
    #if no value in biodiv & LULC_site vector
    if np.sum(bool_site)==0:
        print("no value in biodiv vector...  quit computation.\n")
        return LULC_values
    
    C_mat=np.ascontiguousarray(C_mat[:,bool_site,:])
    biodiv=biodiv[bool_site,:]
    env_var=env_var[bool_site,:]
    num_sites=np.sum(bool_site)
    
    #calculate num_LULC_exist
    bool_LULC=np.sum(C_mat[num_dist_sim-1,:,1:(num_LULC_types+1)],axis=0)>0
    num_LULC_exist=np.sum(bool_LULC)
    
    biodiv_mean=np.mean(biodiv)
    TSS=np.sum((biodiv-biodiv_mean)**2)
    
    biodiv_B0=np.zeros([num_sites,1],dtype=np.float32)   
        
    # B0 separation: to estimate B0 before other parameter estimations
    if bool_B0_sep:        
        C_B0=np.zeros([num_sites,2])
        C_B0[:,0]=1
        C_B0[:,1]=env_var.flatten()
        lm_model_B0=np.linalg.lstsq(C_B0, biodiv, rcond=None)
        lm_coefs_B0=lm_model_B0[0]       
        biodiv_B0=np.dot(C_B0,lm_coefs_B0)
    else:
        biodiv_B0=np.zeros([num_sites,1])     

    #optimization
    for i_dist in range(num_dist_sim):#%%
        dist=dist_list[i_dist]              
        C=np.ascontiguousarray(C_mat[i_dist,:,:])
        bool_LULC_dist=np.sum(C[:,1:(num_LULC_types+1)],axis=0)>0
        num_LULC_dist_exist=np.sum(bool_LULC_dist)
        
        #step 6: regress C and biodiv to estimate LC_Values (OLS)        
        biodiv_R=biodiv/biodiv_B0
        lm_model=np.linalg.lstsq(C, biodiv_R, rcond=None)
        lm_coefs=lm_model[0]
        biodiv_pred=np.dot(C,lm_coefs)*biodiv_B0
        resid=biodiv-biodiv_pred
        RSS=np.sum(resid*resid)  
        sigma2=np.sum(resid*resid)/num_sites
        coefs=lm_coefs[:,0]
        neg2LL=num_sites*(np.log(sigma2)+np.log(2*np.pi))+np.sum(resid*resid/sigma2)

        # #step 6: estimate parameters using maximum likelihood estimation (MLE)
        # params=np.concatenate((np.array([30,3,-2]),np.repeat(2, num_LULC_types)),axis=0)
        # # params=params+np.random.uniform(low=-0.1,high=0.1,size=num_LULC_types+3)
        # # params[0:(num_LULC_types+2)]=coefs 
        # # params[num_LULC_types+2]=sigma2
        # y=biodiv/biodiv_B0
        # y=np.float32(y.flatten())     
        # opt_result=opt.minimize(calc_neg2LogLik, params, args=(C,y,num_sites,num_LULC_types), method='L-BFGS-B')             
        # mle_coefs=opt_result.x[0:(2+num_LULC_types)]
        # mle_fvalue=opt_result.fun
        # neg2LL=mle_fvalue
        # biodiv_pred=np.dot(C,mle_coefs)*biodiv_B0.flatten()
        # RSS=np.sum((biodiv_pred-biodiv.flatten())**2)
        # coefs=mle_coefs
        
        if bool_B0_sep:
            coefs[0]=lm_coefs_B0[0,0]
            coefs[num_LULC_types+1]=lm_coefs_B0[1,0]     
            
        adj_r2=1-(RSS/(num_sites-num_LULC_dist_exist-2-1))/(TSS/(num_sites-1))
        LULC_values[i_dist,(num_LULC_types+2-len(coefs)):(num_LULC_types+2)]=coefs      
        LULC_values[i_dist,num_LULC_types+2]=dist
        LULC_values[i_dist,num_LULC_types+3]=adj_r2
        LULC_values[i_dist,num_LULC_types+4]=neg2LL
        print("simulation no. %d / %d (dist = %.1f km),  r-sq = %.4f, -2LL = %.2f\n"%(i_dist,num_dist_sim,dist,adj_r2,neg2LL))
    return LULC_values

def calc_opt_params_Poisson(C_mat,biodiv,env_var,dist_list,num_dist_sim,num_LULC_types,bool_B0_sep):
    LULC_values=np.zeros([num_dist_sim,num_LULC_types+5])    
    #remove all-zero records
    bool_site=C_mat[0,:,0]!=-1
    bool_biodiv=~np.isnan(biodiv).ravel()
    bool_site=bool_site&bool_biodiv
    #if no value in biodiv & LULC_site vector
    if np.sum(bool_site)==0:
        print("no value in biodiv vector...  quit computation.\n")
        return LULC_values
    
    C_mat=np.ascontiguousarray(C_mat[:,bool_site,:])
    biodiv=biodiv[bool_site,:]
    env_var=env_var[bool_site,:]
    num_sites=np.sum(bool_site)
    
    #calculate num_LULC_exist
    bool_LULC=np.sum(C_mat[num_dist_sim-1,:,1:(num_LULC_types+1)],axis=0)>0
    num_LULC_exist=np.sum(bool_LULC)
    
    biodiv_mean=np.mean(biodiv)
    TSS=np.sum((biodiv-biodiv_mean)**2)
    
    biodiv_B0=np.zeros([num_sites,1],dtype=np.float32)   
        
    # # B0 separation: to estimate B0 before other parameter estimations
    # if bool_B0_sep:        
    #     C_B0=np.zeros([num_sites,2])
    #     C_B0[:,0]=1
    #     C_B0[:,1]=env_var.flatten()
    #     lm_model_B0=np.linalg.lstsq(C_B0, biodiv, rcond=None)
    #     lm_coefs_B0=lm_model_B0[0]       
    #     biodiv_B0=np.dot(C_B0,lm_coefs_B0)
    # else:
    #     biodiv_B0=np.zeros([num_sites,1])

    #optimization
    for i_dist in range(num_dist_sim):#%%
        dist=dist_list[i_dist]              
        C=np.ascontiguousarray(C_mat[i_dist,:,:])
        bool_LULC_dist=np.sum(C[:,1:(num_LULC_types+1)],axis=0)>0
        num_LULC_dist_exist=np.sum(bool_LULC_dist)

        #step 6: Poisson regression for C and biodiv         
        y=biodiv-biodiv_B0
        y=y.flatten()
        try:
            model_poisson = sm.GLM(y,C, family = sm.families.Poisson()).fit()
            df = pd.read_html(model_poisson.summary().tables[1].as_html(),header=0,index_col=0)[0]
            coefs=df["coef"].values
            biodiv_pred=np.exp(np.dot(C,coefs))+biodiv_B0.flatten()
            RSS=np.sum((biodiv_pred-biodiv.flatten())**2)   
            
            # if bool_B0_sep:
            #     coefs[0]=lm_coefs_B0[0,0]
            #     coefs[num_LULC_types+1]=lm_coefs_B0[1,0]     
                
            adj_r2=1-(RSS/(num_sites-num_LULC_dist_exist-2-1))/(TSS/(num_sites-1))
            pseudo_r2=model_poisson.pseudo_rsquared(kind='mcf')
            LULC_values[i_dist,(num_LULC_types+2-len(coefs)):(num_LULC_types+2)]=coefs      
            LULC_values[i_dist,num_LULC_types+2]=dist
            LULC_values[i_dist,num_LULC_types+3]=adj_r2
            LULC_values[i_dist,num_LULC_types+4]=pseudo_r2
            print("Poisson regression no. %d / %d (dist = %.1f km),  adj r2 = %.3f, pseudo r2 = %.3f\n"\
                  %(i_dist,num_dist_sim,dist,adj_r2,pseudo_r2))
        except:
            print("Poisson regression no. %d / %d (dist = %.1f km): computation error.\n"%(i_dist,num_dist_sim,dist))
    return LULC_values

def calc_opt_params_NegativeBinomial(C_mat,biodiv,env_var,dist_list,num_dist_sim,num_LULC_types,bool_B0_sep):
    LULC_values=np.zeros([num_dist_sim,num_LULC_types+5])    
    #remove all-zero records
    bool_site=C_mat[0,:,0]!=-1
    bool_biodiv=~np.isnan(biodiv).ravel()
    bool_site=bool_site&bool_biodiv
    #if no value in biodiv & LULC_site vector
    if np.sum(bool_site)==0:
        print("no value in biodiv vector...  quit computation.\n")
        return LULC_values
    
    C_mat=np.ascontiguousarray(C_mat[:,bool_site,:])
    biodiv=biodiv[bool_site,:]
    env_var=env_var[bool_site,:]
    num_sites=np.sum(bool_site)
    
    #calculate num_LULC_exist
    bool_LULC=np.sum(C_mat[num_dist_sim-1,:,1:(num_LULC_types+1)],axis=0)>0
    num_LULC_exist=np.sum(bool_LULC)
    
    biodiv_mean=np.mean(biodiv)
    TSS=np.sum((biodiv-biodiv_mean)**2)
    
    biodiv_B0=np.zeros([num_sites,1],dtype=np.float32)   
        
    # # B0 separation: to estimate B0 before other parameter estimations
    # if bool_B0_sep:        
    #     C_B0=np.zeros([num_sites,2])
    #     C_B0[:,0]=1
    #     C_B0[:,1]=env_var.flatten()
    #     lm_model_B0=np.linalg.lstsq(C_B0, biodiv, rcond=None)
    #     lm_coefs_B0=lm_model_B0[0]       
    #     biodiv_B0=np.dot(C_B0,lm_coefs_B0)
    # else:
    #     biodiv_B0=np.zeros([num_sites,1])

    #optimization
    for i_dist in range(num_dist_sim):#%%
        dist=dist_list[i_dist]              
        C=np.ascontiguousarray(C_mat[i_dist,:,:])
        bool_LULC_dist=np.sum(C[:,1:(num_LULC_types+1)],axis=0)>0
        num_LULC_dist_exist=np.sum(bool_LULC_dist)

        #step 6: Poisson regression for C and biodiv         
        y=biodiv-biodiv_B0
        y=y.flatten()
        try:
            model_nb = sm.GLM(y,C, family = sm.families.NegativeBinomial()).fit()
            df = pd.read_html(model_nb.summary().tables[1].as_html(),header=0,index_col=0)[0]
            coefs=df["coef"].values
            biodiv_pred=np.exp(np.dot(C,coefs))+biodiv_B0.flatten()
            RSS=np.sum((biodiv_pred-biodiv.flatten())**2)   
            
            # if bool_B0_sep:
            #     coefs[0]=lm_coefs_B0[0,0]
            #     coefs[num_LULC_types+1]=lm_coefs_B0[1,0]     
                
            adj_r2=1-(RSS/(num_sites-num_LULC_dist_exist-2-1))/(TSS/(num_sites-1))
            pseudo_r2=model_nb.pseudo_rsquared(kind='mcf')
            LULC_values[i_dist,(num_LULC_types+2-len(coefs)):(num_LULC_types+2)]=coefs      
            LULC_values[i_dist,num_LULC_types+2]=dist
            LULC_values[i_dist,num_LULC_types+3]=adj_r2
            LULC_values[i_dist,num_LULC_types+4]=pseudo_r2
            print("Poisson regression no. %d / %d (dist = %.1f km),  adj r2 = %.3f, pseudo r2 = %.3f\n"\
                  %(i_dist,num_dist_sim,dist,adj_r2,pseudo_r2))
        except:
            print("Poisson regression no. %d / %d (dist = %.1f km): computation error.\n"%(i_dist,num_dist_sim,dist))
    return LULC_values


def calc_opt_params_species_level(C_mat,biodiv,dist_list,num_dist_sim,num_LULC_types):
    LULC_values=np.zeros([num_dist_sim,num_LULC_types+6])    
    #remove all-zero records
    bool_site=C_mat[0,:,0]!=-1
    C_mat=np.ascontiguousarray(C_mat[:,bool_site,:])
    biodiv=biodiv[bool_site,:].ravel()
    num_sites=np.sum(bool_site)
    if np.sum(biodiv)==num_sites or np.sum(biodiv)==0:
        print("model for this species cannot be built up. \n")
        return    
    for i_dist in range(num_dist_sim):
        dist=dist_list[i_dist]    
        C=np.ascontiguousarray(C_mat[i_dist,:,:])
        #SMOTE to balance classes
        k_neighbors=min(np.min(np.bincount(biodiv))-1,5)
        if k_neighbors>0:
            sm=SMOTE(k_neighbors=k_neighbors)
            [C_res,biodiv_res]=sm.fit_resample(C, biodiv)        
        else:                
            idx=np.where(biodiv==1)[0][0]
            C_app=np.zeros([num_sites+1,num_LULC_types+2])
            biodiv_app=np.zeros(num_sites+1)
            C_app[0:num_sites,:]=C
            biodiv_app[0:num_sites]=biodiv
            C_app[num_sites,:]=C[idx,:]
            biodiv_app[num_sites]=biodiv[idx]
            k_neighbors=1
            sm=SMOTE(k_neighbors=k_neighbors)
            [C_res,biodiv_res]=sm.fit_resample(C_app, biodiv_app)
            
        clf = LogisticRegression(fit_intercept=True).fit(C_res, biodiv_res)
        coefs=clf.coef_.ravel()
        coef_intcp=clf.intercept_
        biodiv_pred_prob=clf.predict_proba(C)
        [p_a_thres,p_a_TSS]=calcOptTSSDiv(biodiv,biodiv_pred_prob,intv=0.01)
        
        biodiv_pred=1*(biodiv_pred_prob[:,1]>=p_a_thres)
        # ACC=np.sum(biodiv_pred==biodiv.ravel())/num_sites
        KAPPA=cohen_kappa_score(biodiv_pred,biodiv)
        
        LULC_values[i_dist,0]=coef_intcp
        LULC_values[i_dist,(num_LULC_types+2-len(coefs)):(num_LULC_types+2)]=coefs      
        LULC_values[i_dist,num_LULC_types+2]=dist
        LULC_values[i_dist,num_LULC_types+3]=p_a_thres
        LULC_values[i_dist,num_LULC_types+4]=p_a_TSS
        LULC_values[i_dist,num_LULC_types+5]=KAPPA
        print("simulation no. %d / %d (dist = %.1f), thres = %.2f, metric = %.3f\n"%(i_dist,num_dist_sim,dist,p_a_thres,p_a_TSS))
    return LULC_values