
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
"""

from random import *
from Izhikevich_class import * # Where definition of single cell models are found based on Izhikevich models
from Analysis_tools import *
import math

# This function sets up the connectome and runs a simulation for the time tmax.
# rand is a seed for a random function
# stim0 is a constant for scaling the drive to V2a
# sigma is a variance for gaussian randomization of the gap junction coupling
# dt is the time step size
# E_glu and E_gly are the reversal potential of glutamate and glycine respectively
# c is the transmission speed
# nMN, nV2a, ndI6, nV0v, nV1, and nMuscle is the number of MN, V2a, dI6, V0v, V1 and Muscle cells
# R_str is an indication of whether glycinergic synapses are present or blocked by strychnine (str). Ranges from 0 to 1. 
#    1: they are present; 0: they are all blocked
def connectome_beat_glide(rand=0, stim0=2.88, sigma=0, sigma_LR = 0.1,
                          tmax=1000, dt=0.1, E_glu=0, E_gly=-70, cv=0.80,
                          nMN=15, ndI6=15, nV0v=15, nV2a=15, nV1=15, nMuscle=15, 
                          R_str=1.0):
    seed(rand)
    ## Declare constants

    tmax += 200 # add 200 ms to give time for neurons to settle.  This initial 200 ms is not used in the data analysis
    nmax = int(tmax/dt) 
    
    ## Declare Neuron Types

    L_MN = [ Izhikevich_9P(a=0.5,b=0.01,c=-55, d=100, vmax=10, vr=-65, vt=-58, k=0.5, Cm = 20, dt=dt, x=5.0+1.6*i*gauss(1, sigma),y=-1) for i in range(nMN)]
    R_MN = [ Izhikevich_9P(a=0.5,b=0.01,c=-55, d=100, vmax=10, vr=-65, vt=-58, k=0.5, Cm = 20, dt=dt, x=5.0+1.6*i*gauss(1, sigma),y=1) for i in range(nMN)]

    L_dI6 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=5.1+1.6*i*gauss(1, sigma),y=-1) for i in range(ndI6)]
    R_dI6 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=5.1+1.6*i*gauss(1, sigma),y=1) for i in range(ndI6)]

    L_V0v = [ Izhikevich_9P(a=0.01,b=0.002,c=-55, d=2, vmax=8, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=5.1+1.6*i*gauss(1, sigma),y=-1) for i in range(nV0v)]
    R_V0v = [ Izhikevich_9P(a=0.01,b=0.002,c=-55, d=2, vmax=8, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=5.1+1.6*i*gauss(1, sigma),y=1) for i in range(nV0v)]

    L_V2a = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=5.1+1.6*i*gauss(1, sigma),y=-1) for i in range(nV2a)]
    R_V2a = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=5.1+1.6*i*gauss(1, sigma),y=1) for i in range(nV2a)]
    
    L_V1 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=7.1+1.6*i*gauss(1, sigma),y=-1) for i in range(nV1)]
    R_V1 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=dt, x=7.1+1.6*i*gauss(1, sigma),y=1) for i in range(nV1)]
    
    L_Muscle = [ Leaky_Integrator(1.0, 3.0, dt, 5.0+1.6*i,-1) for i in range(nMuscle)]
    R_Muscle = [ Leaky_Integrator(1.0, 3.0, dt, 5.0+1.6*i,-1) for i in range(nMuscle)]
    
    ## Declare Synapses
       
    L_glusyn_MN_Muscle = [TwoExp_syn(0.5, 1.0, -15, dt, 120) for i in range (nMN*nMuscle)] 
    R_glusyn_MN_Muscle = [TwoExp_syn(0.5, 1.0, -15, dt, 120) for i in range (nMN*nMuscle)]
    
    L_glysyn_dI6_MN = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (ndI6*nMN)]
    R_glysyn_dI6_MN = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (ndI6*nMN)] 
    L_glysyn_dI6_V2a = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (ndI6*nV2a)]
    R_glysyn_dI6_V2a = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (ndI6*nV2a)]
    L_glysyn_dI6_dI6 = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (ndI6*ndI6)]
    R_glysyn_dI6_dI6 = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (ndI6*ndI6)]
    
    L_glysyn_V1_MN = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*nMN)] 
    R_glysyn_V1_MN = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*nMN)]
    L_glysyn_V1_V2a = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*nV2a)] 
    R_glysyn_V1_V2a = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*nV2a)] 
    L_glysyn_V1_V0v = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*nV0v)]
    R_glysyn_V1_V0v = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*nV0v)]
    L_glysyn_V1_dI6 = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*ndI6)]
    R_glysyn_V1_dI6 = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV1*ndI6)]
    
    L_glusyn_V0v_V2a = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV0v*nV2a)]
    R_glusyn_V0v_V2a = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV0v*nV2a)]
    
    L_glusyn_V2a_V2a = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV2a)]
    R_glusyn_V2a_V2a = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV2a)]
    L_glusyn_V2a_MN = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nMN)]
    R_glusyn_V2a_MN = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nMN)]
    L_glusyn_V2a_dI6 = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*ndI6)]
    R_glusyn_V2a_dI6 = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*ndI6)]
    L_glusyn_V2a_V1 = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV1)]
    R_glusyn_V2a_V1 = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV1)]
    L_glusyn_V2a_V0v = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV0v)]
    R_glusyn_V2a_V0v = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV0v)]
    
    ## Declare Storage tables
    
    Time =zeros(nmax)
    
    VLMN =zeros((nMN, nmax))
    VRMN =zeros((nMN, nmax))
    VLdI6 = zeros ((ndI6,nmax))
    VRdI6 = zeros ((ndI6,nmax))
    VLV0v = zeros ((nV0v,nmax))
    VRV0v = zeros ((nV0v,nmax))
    VLV2a = zeros ((nV2a,nmax))
    VRV2a = zeros ((nV2a,nmax))
    VLV1 = zeros ((nV1,nmax))
    VRV1 = zeros ((nV1,nmax))
    VLMuscle = zeros((nMuscle, nmax))
    VRMuscle = zeros((nMuscle, nmax))
    
    #Lists to store synaptic currents
    
    #gly
    LSyn_dI6_MN = zeros((ndI6*nMN,3))
    RSyn_dI6_MN = zeros((ndI6*nMN,3))
    LSyn_dI6_V2a = zeros((ndI6*nV2a,3))
    RSyn_dI6_V2a = zeros((ndI6*nV2a,3))
    LSyn_dI6_dI6 = zeros((ndI6*ndI6,3))
    RSyn_dI6_dI6 = zeros((ndI6*ndI6,3))
    LSyn_V1_MN = zeros((nV1*nMN,3))
    RSyn_V1_MN = zeros((nV1*nMN,3))
    LSyn_V1_V2a = zeros((nV1*nV2a,3))
    RSyn_V1_V2a = zeros((nV1*nV2a,3))
    LSyn_V1_V0v = zeros((nV1*nV0v,3))
    RSyn_V1_V0v = zeros((nV1*nV0v,3))
    LSyn_V1_dI6 = zeros((nV1*ndI6,3))
    RSyn_V1_dI6 = zeros((nV1*ndI6,3))
    
    #glu
    LSyn_V0v_V2a = zeros((nV0v*nV2a,3))
    RSyn_V0v_V2a = zeros((nV0v*nV2a,3))
    LSyn_V2a_V2a = zeros((nV2a*nV2a,3))
    RSyn_V2a_V2a = zeros((nV2a*nV2a,3))
    LSyn_V2a_MN = zeros((nV2a*nMN,3))
    RSyn_V2a_MN = zeros((nV2a*nMN,3))
    LSyn_V2a_dI6 = zeros((nV2a*ndI6,3))
    RSyn_V2a_dI6 = zeros((nV2a*ndI6,3))
    LSyn_V2a_V1 = zeros((nV2a*nV1,3))
    RSyn_V2a_V1 = zeros((nV2a*nV1,3))
    LSyn_V2a_V0v = zeros((nV2a*nV0v,3))
    RSyn_V2a_V0v = zeros((nV2a*nV0v,3))
    
    #Ach
    LSyn_MN_Muscle = zeros((nMN*nMuscle,3))
    RSyn_MN_Muscle = zeros((nMN*nMuscle,3))
    #Gap
    LSGap_MN_MN = zeros((nMN,nMN))
    LSGap_MN_dI6 = zeros((nMN,ndI6))
    LSGap_MN_V0v = zeros((nMN,nV0v))
    LSGap_MN_V2a = zeros((nMN,nV2a))
    LSGap_dI6_MN = zeros((ndI6,nMN))
    LSGap_dI6_dI6 = zeros((ndI6,ndI6))
    LSGap_V0v_MN = zeros((nV0v,nMN))
    LSGap_V0v_V0v = zeros((nV0v,nV0v))
    LSGap_V2a_V2a = zeros((nV2a,nV2a))
    LSGap_V2a_MN = zeros((nV2a,nMN))
    
    RSGap_MN_MN = zeros((nMN,nMN))
    RSGap_MN_dI6 = zeros((nMN,ndI6))
    RSGap_MN_V0v = zeros((nMN,nV0v))
    RSGap_MN_V2a = zeros((nMN,nV2a))
    RSGap_dI6_MN = zeros((ndI6,nMN))
    RSGap_dI6_dI6 = zeros((ndI6,ndI6))
    RSGap_V0v_MN = zeros((nV0v,nMN))
    RSGap_V0v_V0v = zeros((nV0v,nV0v))
    RSGap_V2a_V2a = zeros((nV2a,nV2a))
    RSGap_V2a_MN = zeros((nV2a,nMN))
    
    
    ### List of synaptic weights
        
    #gly
    LW_dI6_MN = zeros((ndI6,nMN))      
    RW_dI6_MN = zeros((ndI6,nMN))
    LW_dI6_V2a = zeros((ndI6,nV2a))      
    RW_dI6_V2a = zeros((ndI6,nV2a))
    LW_dI6_dI6 = zeros((ndI6,ndI6))      
    RW_dI6_dI6 = zeros((ndI6,ndI6))
    LW_V1_MN = zeros((nV1,nMN))      
    RW_V1_MN = zeros((nV1,nMN))
    LW_V1_V2a = zeros((nV1,nV2a))      
    RW_V1_V2a = zeros((nV1,nV2a))
    LW_V1_V0v = zeros((nV1,nV0v))      
    RW_V1_V0v = zeros((nV1,nV0v))
    LW_V1_dI6 = zeros((nV1,ndI6))      
    RW_V1_dI6 = zeros((nV1,ndI6))
    
    #glu
    LW_V0v_V2a = zeros((nV0v,nV2a))
    RW_V0v_V2a = zeros((nV0v,nV2a))
    LW_V2a_V2a = zeros((nV2a,nV2a))
    RW_V2a_V2a = zeros((nV2a,nV2a))
    LW_V2a_MN = zeros((nV2a,nMN))
    RW_V2a_MN = zeros((nV2a,nMN))
    LW_V2a_dI6 = zeros((nV2a,ndI6))
    RW_V2a_dI6 = zeros((nV2a,ndI6))
    LW_V2a_V1 = zeros((nV2a,nV1))
    RW_V2a_V1 = zeros((nV2a,nV1))
    LW_V2a_V0v = zeros((nV2a,nV0v))
    RW_V2a_V0v = zeros((nV2a,nV0v))
    
    #Ach
    LW_MN_Muscle = zeros((nMN,nMuscle))
    RW_MN_Muscle = zeros((nMN,nMuscle))
    
    
    #List of Gap junctions
    LGap_MN_MN = zeros((nMN,nMN))
    LGap_MN_dI6 = zeros((nMN,ndI6))
    LGap_MN_V0v = zeros((nMN,nV0v))
    LGap_dI6_dI6 = zeros((ndI6,ndI6))
    LGap_V0v_V0v = zeros((nV0v,nV0v))
    LGap_V2a_V2a = zeros((nV2a,nV2a))
    LGap_V2a_MN = zeros((nV2a,nMN))
    
    RGap_MN_MN = zeros((nMN,nMN))
    RGap_MN_dI6 = zeros((nMN,ndI6))
    RGap_MN_V0v = zeros((nMN,nV0v))
    RGap_dI6_dI6 = zeros((ndI6,ndI6))
    RGap_V0v_V0v = zeros((nV0v,nV0v))
    RGap_V2a_V2a = zeros((nV2a,nV2a))
    RGap_V2a_MN = zeros((nV2a,nMN))
    
    #list to store membrane potential residuals 
    resLMN=zeros((nMN,3))
    resRMN=zeros((nMN,3))
    resLdI6 = zeros((ndI6,3))
    resRdI6 = zeros((ndI6,3))
    resLV0v = zeros((nV0v,3))
    resRV0v = zeros((nV0v,3))
    resLV2a = zeros((nV2a,3))
    resRV2a = zeros((nV2a,3))
    resLV1 = zeros((nV1,3))
    resRV1 = zeros((nV1,3))
    resLMuscle = zeros((nMuscle,2))
    resRMuscle = zeros((nMuscle,2))
    
    stim= zeros (nmax)
    stim2= zeros (nmax)
        
    #Distance Matrix 
    LD_MN_MN = zeros((nMN,nMN))
    LD_MN_dI6 = zeros((nMN,ndI6))
    LD_MN_V0v = zeros((nMN,nV0v))
    LD_dI6_dI6 = zeros((ndI6,ndI6))
    LD_V0v_V0v = zeros((nV0v,nV0v))
    LD_V0v_V2a = zeros((nV0v,nV2a))
    
    LD_V2a_V2a = zeros((nV2a,nV2a))     
    LD_V2a_MN = zeros((nV2a,nMN))           
    LD_V2a_dI6 = zeros((nV2a,ndI6))   
    LD_V2a_V1 = zeros((nV2a,nV1))   
    LD_V2a_V0v = zeros((nV2a,nV0v))   
    
    LD_V1_MN = zeros((nV1,nMN))
    LD_V1_V0v = zeros((nV1,nV0v))
    LD_V1_dI6 = zeros((nV1,ndI6))
    
    LD_dI6_MN = zeros((ndI6,nMN))     
    LD_dI6_V2a = zeros((ndI6,nV2a))   
    
    RD_MN_MN = zeros((nMN,nMN))
    RD_MN_dI6 = zeros((nMN,ndI6))
    RD_MN_V0v = zeros((nMN,nV0v))
    RD_dI6_dI6 = zeros((ndI6,ndI6))
    RD_V0v_V0v = zeros((nV0v,nV0v))
    RD_V0v_V2a = zeros((nV0v,nV2a))
    
    RD_V2a_V2a = zeros((nV2a,nV2a))     
    RD_V2a_MN = zeros((nV2a,nMN))           
    RD_V2a_dI6 = zeros((nV2a,ndI6))   
    RD_V2a_V1 = zeros((nV2a,nV1))   
    RD_V2a_V0v = zeros((nV2a,nV0v))
    
    RD_V1_MN = zeros((nV1,nMN))
    RD_V1_V0v = zeros((nV1,nV0v))
    RD_V1_dI6 = zeros((nV1,ndI6))
    
    RD_dI6_MN = zeros((ndI6,nMN))     
    RD_dI6_V2a = zeros((ndI6,nV2a))   
    
    ## Compute distance between Neurons
    
    #LEFT
            
    for k in range (0, nMN):
        for l in range (0, nMN):
            LD_MN_MN[k,l] = Distance(L_MN[k].x,L_MN[l].x,L_MN[k].y,L_MN[l].y)
            
    for k in range (0, nMN):
        for l in range (0, ndI6):
            LD_MN_dI6[k,l] = Distance(L_MN[k].x,L_dI6[l].x,L_MN[k].y,L_dI6[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nV0v):
            LD_MN_V0v[k,l] = Distance(L_MN[k].x,L_V0v[l].x,L_MN[k].y,L_V0v[l].y)
 
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            LD_V2a_V2a[k,l] = Distance(L_V2a[k].x,L_V2a[l].x,L_V2a[k].y,L_V2a[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nMN):
            LD_V2a_MN[k,l] = Distance(L_V2a[k].x,L_MN[l].x,L_V2a[k].y,L_MN[l].y)
                      
    for k in range (0, nV2a):
        for l in range (0, ndI6):
            LD_V2a_dI6[k,l] = Distance(L_V2a[k].x,L_dI6[l].x,L_V2a[k].y,L_dI6[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nV1):
            LD_V2a_V1[k,l] = Distance(L_V2a[k].x,L_V1[l].x,L_V2a[k].y,L_V1[l].y)

    for k in range (0, nV2a):
        for l in range (0, nV0v):
            LD_V2a_V0v[k,l] = Distance(L_V2a[k].x,L_V0v[l].x,L_V2a[k].y,L_V0v[l].y)

    for k in range (0, nV1):
        for l in range (0, nMN):
            LD_V1_MN[k,l] = Distance(L_V1[k].x,L_MN[l].x,L_V1[k].y,L_MN[l].y) 
            
    for k in range (0, nV1):
        for l in range (0, nV0v):
            LD_V1_V0v[k,l] = Distance(L_V1[k].x,L_V0v[l].x,L_V1[k].y,L_V0v[l].y)    
    
    for k in range (0, nV1):
        for l in range (0, ndI6):
            LD_V1_dI6[k,l] = Distance(L_V1[k].x,L_dI6[l].x,L_V1[k].y,L_dI6[l].y) 
                
    for k in range (0, ndI6):
        for l in range (0, nMN):
            LD_dI6_MN[k,l] = Distance(L_dI6[k].x,R_MN[l].x,L_dI6[k].y,R_MN[l].y) #Contalateral
                      
    for k in range (0, ndI6):
        for l in range (0, nV2a):
            LD_dI6_V2a[k,l] = Distance(L_dI6[k].x,R_V2a[l].x,L_dI6[k].y,R_V2a[l].y) #Contalateral
            
    for k in range (0, ndI6):
        for l in range (0, ndI6):
            LD_dI6_dI6[k,l] = Distance(L_dI6[k].x,L_dI6[l].x,L_dI6[k].y,L_dI6[l].y) #Contalateral
                    
    for k in range (0, nV0v):
        for l in range (0, nV2a):
            LD_V0v_V2a[k,l] = Distance(L_V0v[k].x,R_V2a[l].x,L_V0v[k].y,R_V2a[l].y) #Contalateral
            
    for k in range (0, nV0v):
        for l in range (0, nV0v):
            LD_V0v_V0v[k,l] = Distance(L_V0v[k].x,L_V0v[l].x,L_V0v[k].y,L_V0v[l].y) #Contalateral
            

    
    #RIGHT
    for k in range (0, nMN):
        for l in range (0, nMN):
            RD_MN_MN[k,l] = Distance(R_MN[k].x,R_MN[l].x,R_MN[k].y,R_MN[l].y)
            
    for k in range (0, nMN):
        for l in range (0, ndI6):
            RD_MN_dI6[k,l] = Distance(R_MN[k].x,R_dI6[l].x,R_MN[k].y,R_dI6[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nV0v):
            RD_MN_V0v[k,l] = Distance(R_MN[k].x,R_V0v[l].x,R_MN[k].y,R_V0v[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            RD_V2a_V2a[k,l] = Distance(R_V2a[k].x,R_V2a[l].x,R_V2a[k].y,R_V2a[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nMN):
            RD_V2a_MN[k,l] = Distance(R_V2a[k].x,R_MN[l].x,R_V2a[k].y,R_MN[l].y)
             
    for k in range (0, nV2a):
        for l in range (0, ndI6):
            RD_V2a_dI6[k,l] = Distance(R_V2a[k].x,R_dI6[l].x,R_V2a[k].y,R_dI6[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nV1):
            RD_V2a_V1[k,l] = Distance(R_V2a[k].x,R_V1[l].x,R_V2a[k].y,R_V1[l].y)

    for k in range (0, nV2a):
        for l in range (0, nV0v):
            RD_V2a_V0v[k,l] = Distance(R_V2a[k].x,R_V0v[l].x,R_V2a[k].y,R_V0v[l].y)

    for k in range (0, nV1):
        for l in range (0, nMN):
            RD_V1_MN[k,l] = Distance(R_V1[k].x,R_MN[l].x,R_V1[k].y,R_MN[l].y) 
            
    for k in range (0, nV1):
        for l in range (0, nV0v):
            RD_V1_V0v[k,l] = Distance(R_V1[k].x,R_V0v[l].x,R_V1[k].y,R_V0v[l].y)     
            
    for k in range (0, nV1):
        for l in range (0, ndI6):
            RD_V1_dI6[k,l] = Distance(R_V1[k].x,R_dI6[l].x,R_V1[k].y,R_dI6[l].y)
            
    for k in range (0, ndI6):
        for l in range (0, nMN):
            RD_dI6_MN[k,l] = Distance(R_dI6[k].x,L_MN[l].x,R_dI6[k].y,L_MN[l].y) #Contalateral
            
    for k in range (0, ndI6):
        for l in range (0, nV2a):
            RD_dI6_V2a[k,l] = Distance(R_dI6[k].x,L_V2a[l].x,R_dI6[k].y,L_V2a[l].y) #Contalateral
               
    for k in range (0, ndI6):
        for l in range (0, ndI6):
            RD_dI6_dI6[k,l] = Distance(R_dI6[k].x,R_dI6[l].x,R_dI6[k].y,R_dI6[l].y) 
            
    for k in range (0, nV0v):
        for l in range (0, nV0v):
            RD_V0v_V0v[k,l] = Distance(R_V0v[k].x,R_V0v[l].x,R_V0v[k].y,R_V0v[l].y)
    
    for k in range (0, nV0v):
        for l in range (0, nV2a):
            RD_V0v_V2a[k,l] = Distance(R_V0v[k].x,L_V2a[l].x,R_V0v[k].y,L_V2a[l].y) #Contalateral
    
    ## Compute synaptic weights and gap junction weights
    
   #LEFT
    MN_MN_gap_weight = 0.005 
    for k in range (0, nMN):
        for l in range (0, nMN):
            if (0.2<LD_MN_MN[k,l]< 4.5 ):    
                LGap_MN_MN[k,l] = MN_MN_gap_weight*gauss(1, sigma)
            else:
                LGap_MN_MN[k,l] = 0.0

    dI6_dI6_gap_weight = 0.04
    for k in range (0, ndI6):
        for l in range (0, ndI6):
            if (0.2<LD_dI6_dI6[k,l]< 3.5 ):    
                LGap_dI6_dI6[k,l] = dI6_dI6_gap_weight*gauss(1, sigma)
            else:
                LGap_dI6_dI6[k,l] = 0.0

    V0v_V0v_gap_weight = 0.05
    for k in range (0, nV0v):
        for l in range (0, nV0v):
            if (0.2<LD_V0v_V0v[k,l]< 3.5 ):    
                LGap_V0v_V0v[k,l] = V0v_V0v_gap_weight*gauss(1, sigma)
            else:
                LGap_V0v_V0v[k,l] = 0.0

    V2a_V2a_gap_weight = 0.005
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            if (0.2<LD_V2a_V2a[k,l]< 3.5 ):    
                LGap_V2a_V2a[k,l] = V2a_V2a_gap_weight*gauss(1, sigma)
            else:
                LGap_V2a_V2a[k,l] = 0.0

    V2a_MN_gap_weight = 0.005
    for k in range (0, nV2a):
        for l in range (0, nMN):
            if (0.2<LD_V2a_MN[k,l]< 3.5 ):    
                LGap_V2a_MN[k,l] = V2a_MN_gap_weight*gauss(1, sigma)
            else:
                LGap_V2a_MN[k,l] = 0.0
    
    MN_dI6_gap_weight = 0.0001
    for k in range (0, nMN):
        for l in range (0, ndI6):
            if (L_dI6[l].x-1.5 <L_MN[k].x< L_dI6[l].x + 1.5 ):    
                LGap_MN_dI6[k,l] = MN_dI6_gap_weight*gauss(1, sigma)
            else:
                LGap_MN_dI6[k,l] = 0.0

    MN_V0v_gap_weight = 0.005 
    for k in range (0, nMN):
        for l in range (0, nV0v):
            if (L_V0v[l].x-1.5 <L_MN[k].x< L_V0v[l].x + 1.5 ):    
                LGap_MN_V0v[k,l] = MN_V0v_gap_weight *gauss(1, sigma)
            else:
                LGap_MN_V0v[k,l] = 0.0
                
    #chem syn
    V2a_V2a_syn_weight = 0.3 
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            if (0.2<LD_V2a_V2a[k,l]<10 and L_V2a[k].x < L_V2a[l].x): #the second condition is because the connection is descending
                LW_V2a_V2a[k,l] = V2a_V2a_syn_weight*gauss(1, sigma)
            else:
                LW_V2a_V2a[k,l] = 0.0
    
    V2a_MN_syn_weight = 0.5
    for k in range (0, nV2a):
        for l in range (0, nMN):
            if (0<LD_V2a_MN[k,l]<10 and L_V2a[k].x < L_MN[l].x) or (0.2<LD_V2a_MN[k,l]<4 and L_V2a[k].x > L_MN[l].x): #the second condition is because the connection is descending and short ascending branch, used to be 0<LD_V2a_MN[k,l]<10
                LW_V2a_MN[k,l] = V2a_MN_syn_weight*gauss(1, sigma)
            else:
                LW_V2a_MN[k,l] = 0.0

    V2a_dI6_syn_weight = 0.3       
    for k in range (0, nV2a):
        for l in range (0, ndI6):
            if (0<LD_V2a_dI6[k,l]<10 and L_V2a[k].x < L_dI6[l].x): #the second condition is because the connection is descending
                LW_V2a_dI6[k,l] = V2a_dI6_syn_weight*gauss(1, sigma) 
            else:
                LW_V2a_dI6[k,l] = 0.0

    V2a_V1_syn_weight = 0.5           
    for k in range (0, nV2a):
        for l in range (0, nV1):
            if (0<LD_V2a_V1[k,l]<10 and L_V2a[k].x < L_V1[l].x):   #the second condition is because the connection is descending
                LW_V2a_V1[k,l] = V2a_V1_syn_weight *gauss(1, sigma) 
            else:
                LW_V2a_V1[k,l] = 0.0

    V2a_V0v_syn_weight = 0.414 # 0.25  
    for k in range (0, nV2a):
        for l in range (0, nV0v):
            if (0<LD_V2a_V0v[k,l]<10 and L_V2a[k].x < L_V0v[l].x) or (0.2<LD_V2a_V0v[k,l]<4 and L_V2a[k].x > L_V0v[l].x):    #the second condition is because the connection is descending and short ascending
                LW_V2a_V0v[k,l] = V2a_V0v_syn_weight*gauss(1, sigma) 
            else:
                LW_V2a_V0v[k,l] = 0.0                
 
    V1_MN_syn_weight = 1.0      
    for k in range (0, nV1):
        for l in range (0, nMN):
            if (0<LD_V1_MN[k,l]<4.0 and L_MN[l].x < L_V1[k].x):    #the second condition is because the connection is ascending
                LW_V1_MN[k,l] = V1_MN_syn_weight*gauss(1, sigma) 
            else:
                LW_V1_MN[k,l] = 0.0
                
    V1_V2a_syn_weight = 0.75    
    for k in range (0, nV1):
        for l in range (0, nV2a):
            if (0<LD_V2a_V1[l,k]<4.0 and L_V2a[l].x < L_V1[k].x):    #the second condition is because the connection is ascending
                LW_V1_V2a[k,l] = V1_V2a_syn_weight*gauss(1, sigma) 
            else:
                LW_V1_V2a[k,l] = 0.0

    V1_V0v_syn_weight = 0.1
    for k in range (0, nV1):
        for l in range (0, nV0v):
            if (0<LD_V1_V0v[k,l]<4.0 and L_V0v[l].x < L_V1[k].x):    #the second condition is because the connection is ascending
                LW_V1_V0v[k,l] = V1_V0v_syn_weight*gauss(1, sigma)  
            else:
                LW_V1_V0v[k,l] = 0.0
                
    V1_dI6_syn_weight = 0.2 
    for k in range (0, nV1):
        for l in range (0, ndI6):
            if (0<LD_V1_dI6[k,l]<4.0 and L_dI6[l].x < L_V1[k].x):    #the second condition is because the connection is ascending
                LW_V1_dI6[k,l] = V1_dI6_syn_weight*gauss(1, sigma)  
            else:
                LW_V1_dI6[k,l] = 0.0
    
    dI6_MN_syn_weight = 1.5  
    for k in range (0, ndI6):
        for l in range (0, nMN):
            if (0<LD_dI6_MN[k,l]<5.0 and L_dI6[k].x < R_MN[l].x) or (0.2<LD_dI6_MN[k,l]<2.0 and L_dI6[k].x > R_MN[l].x):   #because contralateral and bifurcating
                LW_dI6_MN[k,l] = dI6_MN_syn_weight*gauss(1, sigma)  
            else:
                LW_dI6_MN[k,l] = 0.0
    
    dI6_V2a_syn_weight = 1.5 
    for k in range (0, ndI6):
        for l in range (0, nV2a):
            if (0<LD_dI6_V2a[k,l]<5.0 and L_dI6[k].x < R_V2a[l].x) or (0.2<LD_dI6_V2a[k,l]<2.0 and L_dI6[k].x > R_V2a[l].x):     #because contralateral  and bifurcating
                LW_dI6_V2a[k,l] = dI6_V2a_syn_weight*gauss(1, sigma)
            else:
                LW_dI6_V2a[k,l] = 0.0
    
    dI6_dI6_syn_weight = 0.25
    for k in range (0, ndI6):
        for l in range (0, ndI6):
            if (0<LD_dI6_dI6[k,l]<5.0 and L_dI6[k].x < R_dI6[l].x) or (0.2<LD_dI6_dI6[k,l]<2.0 and L_dI6[k].x > R_dI6[l].x):     #because contralateral  and bifurcating
                LW_dI6_dI6[k,l] = dI6_dI6_syn_weight*gauss(1, sigma)*gauss(1, sigma_LR)
            else:
                LW_dI6_dI6[k,l] = 0.0
    
    V0v_V2a_syn_weight = 0.4   
    for k in range (0, nV0v):
        for l in range (0, nV2a):
            if (0<LD_V0v_V2a[k,l]<5.0 and L_V0v[k].x < R_V2a[l].x) or (0.2<LD_V0v_V2a[k,l]<2.0 and L_V0v[k].x > R_V2a[l].x): #because contralateral and bifurcating
                LW_V0v_V2a[k,l] = V0v_V2a_syn_weight*gauss(1, sigma) 
            else:
                LW_V0v_V2a[k,l] = 0.0
    
    MN_Muscle_syn_weight = 0.1 
    for k in range (0, nMN):
        for l in range (0, nMuscle):
            if (L_Muscle[l].x-1<L_MN[k].x< L_Muscle[l].x+1):         #this connection is segmental
                LW_MN_Muscle[k,l] = MN_Muscle_syn_weight *gauss(1, sigma) 
            else:
                LW_MN_Muscle[k,l] = 0.0
                
    #RIGHT
    for k in range (0, nMN):
        for l in range (0, nMN):
            if (0.2<RD_MN_MN[k,l]< 4.5 ):    
                RGap_MN_MN[k,l] =  MN_MN_gap_weight*gauss(1, sigma)  
            else:
                RGap_MN_MN[k,l] = 0.0
                
    for k in range (0, ndI6):
        for l in range (0, ndI6):
            if (0.2<RD_dI6_dI6[k,l]< 3.5 ):    
                RGap_dI6_dI6[k,l] = dI6_dI6_gap_weight*gauss(1, sigma)  
            else:
                RGap_dI6_dI6[k,l] = 0.0
                
    for k in range (0, nV0v):
        for l in range (0, nV0v):
            if (0.2<RD_V0v_V0v[k,l]< 3.5 ):    
                RGap_V0v_V0v[k,l] = V0v_V0v_gap_weight*gauss(1, sigma) 
            else:
                RGap_V0v_V0v[k,l] = 0.0
                              
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            if (0.2<RD_V2a_V2a[k,l]< 3.5 ):    
                RGap_V2a_V2a[k,l] = V2a_V2a_gap_weight*gauss(1, sigma) 
            else:
                RGap_V2a_V2a[k,l] = 0.0
                
    for k in range (0, nV2a):
        for l in range (0, nMN):
            if (0.2<RD_V2a_MN[k,l]< 3.5 ):    
                RGap_V2a_MN[k,l] =  V2a_MN_gap_weight*gauss(1, sigma) 
            else:
                RGap_V2a_MN[k,l] = 0.0
                
    for k in range (0, nMN):
        for l in range (0, ndI6):
            if (R_dI6[l].x-1.5<R_MN[k].x< R_dI6[l].x + 1.5 ):    
                RGap_MN_dI6[k,l] =  MN_dI6_gap_weight*gauss(1, sigma) 
            else:
                RGap_MN_dI6[k,l] = 0.0
                
    for k in range (0, nMN):
        for l in range (0, nV0v):
            if (R_V0v[l].x-1.5 <R_MN[k].x< R_V0v[l].x + 1.5 ):    
                RGap_MN_V0v[k,l] =  MN_V0v_gap_weight*gauss(1, sigma) 
            else:
                RGap_MN_V0v[k,l] = 0.0
            
    #chem syn
    
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            if (0.2<RD_V2a_V2a[k,l]<10 and R_V2a[k].x < R_V2a[l].x): #the second condition is because the connection is descending
                RW_V2a_V2a[k,l] =  V2a_V2a_syn_weight*gauss(1, sigma) 
            else:
                RW_V2a_V2a[k,l] = 0.0
    
    for k in range (0, nV2a):
        for l in range (0, nMN):
            if (0<RD_V2a_MN[k,l]<10 and R_V2a[k].x < R_MN[l].x) or (0.2<RD_V2a_MN[k,l]<4 and R_V2a[k].x > R_MN[l].x): #the second condition is because the connection is descending and short ascending branch
                RW_V2a_MN[k,l] =  V2a_MN_syn_weight*gauss(1, sigma) 
            else:
                RW_V2a_MN[k,l] = 0.0
                  
    for k in range (0, nV2a):
        for l in range (0, ndI6):
            if (0<RD_V2a_dI6[k,l]<10 and R_V2a[k].x < R_dI6[l].x): #the second condition is because the connection is descending and short ascending branch
                RW_V2a_dI6[k,l] =  V2a_dI6_syn_weight *gauss(1, sigma) 
            else:
                RW_V2a_dI6[k,l] = 0.0
                
    for k in range (0, nV2a):
        for l in range (0, nV1):
            if (0<RD_V2a_V1[k,l]<10 and R_V2a[k].x < R_V1[l].x):    #the second condition is because the connection is descending
                RW_V2a_V1[k,l] =  V2a_V1_syn_weight*gauss(1, sigma) 
            else:
                RW_V2a_V1[k,l] = 0.0
                
    for k in range (0, nV2a):
        for l in range (0, nV0v):
            if (0<RD_V2a_V0v[k,l]<10 and R_V2a[k].x < R_V0v[l].x) or (0.2<RD_V2a_V0v[k,l]<4 and R_V2a[k].x > R_V0v[l].x):    #the second condition is because the connection is descending and short ascending branch
                RW_V2a_V0v[k,l] =  V2a_V0v_syn_weight*gauss(1, sigma) 
            else:
                RW_V2a_V0v[k,l] = 0.0   

    for k in range (0, nV1):
        for l in range (0, nMN):
            if (0<RD_V1_MN[k,l]<4.0 and R_MN[l].x < R_V1[k].x):    #the second condition is because the connection is ascending
                RW_V1_MN[k,l] = V1_MN_syn_weight*gauss(1, sigma) 
            else:
                RW_V1_MN[k,l] = 0.0

    for k in range (0, nV1):
        for l in range (0, nV2a):
            if (0<RD_V2a_V1[l,k]<4.0 and R_V2a[l].x < R_V1[k].x):    #the second condition is because the connection is ascending
                RW_V1_V2a[k,l] =  V1_V2a_syn_weight*gauss(1, sigma) 
            else:
                RW_V1_V2a[k,l] = 0.0

    for k in range (0, nV1):
        for l in range (0, ndI6):
            if (0<RD_V1_dI6[k,l]<4.0 and R_dI6[l].x < R_V1[k].x):    #the second condition is because the connection is ascending
                RW_V1_dI6[k,l] = V1_dI6_syn_weight *gauss(1, sigma) 
            else:
                RW_V1_dI6[k,l] = 0.0    
                
    for k in range (0, nV1):
        for l in range (0, nV0v):
            if (0<RD_V1_V0v[k,l]<4.0 and R_V0v[l].x < R_V1[k].x):    #the second condition is because the connection is ascending
                RW_V1_V0v[k,l] = V1_V0v_syn_weight*gauss(1, sigma)  
            else:
                RW_V1_V0v[k,l] = 0.0
                
    for k in range (0, ndI6):
        for l in range (0, nMN):
            if (0<RD_dI6_MN[k,l]<5.0 and R_dI6[k].x < L_MN[l].x) or (0.2<RD_dI6_MN[k,l]<2.0 and R_dI6[k].x > L_MN[l].x):   #because contralateral
                RW_dI6_MN[k,l] = dI6_MN_syn_weight*gauss(1, sigma) 
            else:
                RW_dI6_MN[k,l] = 0.0
                
    for k in range (0, ndI6):
        for l in range (0, nV2a):
            if (0<RD_dI6_V2a[k,l]<5.0 and R_dI6[k].x < L_V2a[l].x) or (0.2<RD_dI6_V2a[k,l]<2.0 and R_dI6[k].x > L_V2a[l].x): #because contralateral
                RW_dI6_V2a[k,l] = dI6_V2a_syn_weight *gauss(1, sigma) 
            else:
                RW_dI6_V2a[k,l] = 0.0

    for k in range (0, ndI6):
        for l in range (0, ndI6):
            if (0<RD_dI6_dI6[k,l]<5.0 and R_dI6[k].x < L_dI6[l].x) or (0.2<RD_dI6_dI6[k,l]<2.0 and R_dI6[k].x > L_dI6[l].x):     #because contralateral
                RW_dI6_dI6[k,l] = dI6_dI6_syn_weight *gauss(1, sigma)*gauss(1, sigma_LR) 
            else:
                RW_dI6_dI6[k,l] = 0.0

    for k in range (0, nV0v):
        for l in range (0, nV2a):
            if (0<RD_V0v_V2a[k,l]<5.0 and R_V0v[k].x < L_V2a[l].x) or (0.2<RD_V0v_V2a[k,l]<2.0 and R_V0v[k].x > L_V2a[l].x): #because contralateral and bifurcating
                RW_V0v_V2a[k,l] = V0v_V2a_syn_weight*gauss(1, sigma) 
            else:
                RW_V0v_V2a[k,l] = 0.0        

    for k in range (0, nMN):
        for l in range (0, nMuscle):
            if (R_Muscle[l].x-1<R_MN[k].x< R_Muscle[l].x+1):         #it is segmental
                RW_MN_Muscle[k,l] = MN_Muscle_syn_weight*gauss(1, sigma)  
            else:
                RW_MN_Muscle[k,l] = 0.0
    
    ## Initialize membrane potential values           
    
    for k in range (0, nMN):
        resLMN[k,:] = L_MN[k].getNextVal(-70,-14,-70)
        VLMN[k,0] = resLMN[k,0]
        
        resRMN[k,:] = R_MN[k].getNextVal(-70,-14,-70)
        VRMN[k,0] = resRMN[k,0]
        
    for k in range (0, ndI6):
        resLdI6[k,:] = L_dI6[k].getNextVal(-70,-14,-70)
        VLdI6[k,0] = resLdI6[k,0]
        
        resRdI6[k,:] = R_dI6[k].getNextVal(-70,-14,-70)
        VRdI6[k,0] = resRdI6[k,0]
        
    for k in range (0, nV0v):
        resLV0v[k,:] = L_V0v[k].getNextVal(-70,-14,-70)
        VLV0v[k,0] = resLV0v[k,0]
        
        resRV0v[k,:] = R_V0v[k].getNextVal(-70,-14,-70)
        VRV0v[k,0] = resRV0v[k,0]
        
    for k in range (0, nV2a):
        resLV2a[k,:] = L_V2a[k].getNextVal(-64,-16,-64)
        VLV2a[k,0] = resLV2a[k,0]
        
        resRV2a[k,:] = R_V2a[k].getNextVal(-64,-16,-64)
        VRV2a[k,0] = resRV2a[k,0]
        
    for k in range (0, nV1):
        resLV1[k,:] = L_V1[k].getNextVal(-64,-16,-64)
        VLV1[k,0] = resLV1[k,0]
        
        resRV1[k,:] = R_V1[k].getNextVal(-64,-16,-64)
        VRV1[k,0] = resRV1[k,0]
    
    for k in range (0, nMuscle):
        resLMuscle[k,:] = L_Muscle[k].getNextVal(0,0)
        VLMuscle[k,0] = resLMuscle[k,0]
        
        resRMuscle[k,:] = R_Muscle[k].getNextVal(0,0)
        VRMuscle[k,0] = resRMuscle[k,0]
        
    ## This loop is the main loop where we solve the ordinary differential equations at every time point     
    
    for t in range (0, nmax):
        Time[t]=dt*t
        
        if not(Time[t] % 500) and (Time[t]>20):
            
            fig, ax = plt.subplots(2,1, sharex=True, figsize=(15, 15)) 
            cmapL = matplotlib.cm.get_cmap('Blues')
            cmapR = matplotlib.cm.get_cmap('Reds')
            ax[0].plot([0], [0], c=cmapL(0.5))

            for k in range (0, nMN):
                ax[0].plot(Time, VLMN[k,:], c=cmapL((k+1)/nMN)) # adding a color gradiant, darker color -> rostrally located
                ax[0].plot(Time, VRMN[k,:], c=cmapR((k+1)/nMN))
            plt.xlabel('Time (ms)')
            plt.xlim([0, Time[t]])
            plt.show()
        
        if t > 2000: # Let the initial conditions dissipate for the first 200 ms
            stim2[t] = stim0 #0.02*math.exp(-(t*dt/1000)) + stim0
                
        ## Calculate synaptic currents
        
        for k in range (0, ndI6):
            for l in range (0, nMN):
                LSyn_dI6_MN[nMN*k+l,:] = L_glysyn_dI6_MN[nMN*k+l].getNextVal(VLdI6[k,t-int(LD_dI6_MN[k,l]/(dt*cv))], VRMN[l,t-1], LSyn_dI6_MN[nMN*k+l,1], LSyn_dI6_MN[nMN*k+l,2]) #Contralateral
                RSyn_dI6_MN[nMN*k+l,:] = R_glysyn_dI6_MN[nMN*k+l].getNextVal(VRdI6[k,t-int(RD_dI6_MN[k,l]/(dt*cv))], VLMN[l,t-1], RSyn_dI6_MN[nMN*k+l,1], RSyn_dI6_MN[nMN*k+l,2]) #Contralateral
    
        for k in range (0, ndI6):
            for l in range (0, nV2a):
                LSyn_dI6_V2a[nV2a*k+l,:] = L_glysyn_dI6_V2a[nV2a*k+l].getNextVal(VLdI6[k,t-int(LD_dI6_V2a[k,l]/(dt*cv))], VRV2a[l,t-1], LSyn_dI6_V2a[nV2a*k+l,1], LSyn_dI6_V2a[nV2a*k+l,2])  #Contralateral
                RSyn_dI6_V2a[nV2a*k+l,:] = R_glysyn_dI6_V2a[nV2a*k+l].getNextVal(VRdI6[k,t-int(RD_dI6_V2a[k,l]/(dt*cv))], VLV2a[l,t-1], RSyn_dI6_V2a[nV2a*k+l,1], RSyn_dI6_V2a[nV2a*k+l,2])  #Contralateral
                
        for k in range (0, ndI6):
            for l in range (0, ndI6):
                LSyn_dI6_dI6[ndI6*k+l,:] = L_glysyn_dI6_dI6[ndI6*k+l].getNextVal(VLdI6[k,t-int(LD_dI6_dI6[k,l]/(dt*cv))], VRdI6[l,t-1], LSyn_dI6_dI6[ndI6*k+l,1], LSyn_dI6_dI6[ndI6*k+l,2])  #Contralateral
                RSyn_dI6_dI6[ndI6*k+l,:] = R_glysyn_dI6_dI6[ndI6*k+l].getNextVal(VRdI6[k,t-int(RD_dI6_dI6[k,l]/(dt*cv))], VLdI6[l,t-1], RSyn_dI6_dI6[ndI6*k+l,1], RSyn_dI6_dI6[ndI6*k+l,2])  #Contralateral
                              
        for k in range (0, nV0v):
            for l in range (0, nV2a):
                LSyn_V0v_V2a[nV2a*k+l,:] = L_glusyn_V0v_V2a[nV2a*k+l].getNextVal(VLV0v[k,t-int(LD_V0v_V2a[k,l]/(dt*cv))], VRV2a[l,t-1], LSyn_V0v_V2a[nV2a*k+l,1], LSyn_V0v_V2a[nV2a*k+l,2])  #Contralateral
                RSyn_V0v_V2a[nV2a*k+l,:] = R_glusyn_V0v_V2a[nV2a*k+l].getNextVal(VRV0v[k,t-int(RD_V0v_V2a[k,l]/(dt*cv))], VLV2a[l,t-1], RSyn_V0v_V2a[nV2a*k+l,1], RSyn_V0v_V2a[nV2a*k+l,2])  #Contralateral                
    
        for k in range (0, nV2a):
            for l in range (0, nV2a):
                LSyn_V2a_V2a[nV2a*k+l,:] = L_glusyn_V2a_V2a[nV2a*k+l].getNextVal(VLV2a[k,t-int(LD_V2a_V2a[k,l]/(dt*cv))], VLV2a[l,t-1], LSyn_V2a_V2a[nV2a*k+l,1], LSyn_V2a_V2a[nV2a*k+l,2])  #Contralateral
                RSyn_V2a_V2a[nV2a*k+l,:] = R_glusyn_V2a_V2a[nV2a*k+l].getNextVal(VRV2a[k,t-int(RD_V2a_V2a[k,l]/(dt*cv))], VRV2a[l,t-1], RSyn_V2a_V2a[nV2a*k+l,1], RSyn_V2a_V2a[nV2a*k+l,2])  #Contralateral
    
        for k in range (0, nV2a):
            for l in range (0, nMN):
                LSyn_V2a_MN[nMN*k+l,:] = L_glusyn_V2a_MN[nMN*k+l].getNextVal(VLV2a[k,t-int(LD_V2a_MN[k,l]/(dt*cv))], VLMN[l,t-1], LSyn_V2a_MN[nMN*k+l,1], LSyn_V2a_MN[nMN*k+l,2])  #Contralateral
                RSyn_V2a_MN[nMN*k+l,:] = R_glusyn_V2a_MN[nMN*k+l].getNextVal(VRV2a[k,t-int(RD_V2a_MN[k,l]/(dt*cv))], VRMN[l,t-1], RSyn_V2a_MN[nMN*k+l,1], RSyn_V2a_MN[nMN*k+l,2])  #Contralateral
    
        for k in range (0, nV2a):
            for l in range (0, ndI6):
                LSyn_V2a_dI6[ndI6*k+l,:] = L_glusyn_V2a_dI6[ndI6*k+l].getNextVal(VLV2a[k,t-int(LD_V2a_dI6[k,l]/(dt*cv))], VLdI6[l,t-1], LSyn_V2a_dI6[ndI6*k+l,1], LSyn_V2a_dI6[ndI6*k+l,2])  #Contralateral
                RSyn_V2a_dI6[ndI6*k+l,:] = R_glusyn_V2a_dI6[ndI6*k+l].getNextVal(VRV2a[k,t-int(RD_V2a_dI6[k,l]/(dt*cv))], VRdI6[l,t-1], RSyn_V2a_dI6[ndI6*k+l,1], RSyn_V2a_dI6[ndI6*k+l,2])  #Contralateral
    
        for k in range (0, nV2a):
            for l in range (0, nV1):
                LSyn_V2a_V1[nV1*k+l,:] = L_glusyn_V2a_V1[nV1*k+l].getNextVal(VLV2a[k,t-int(LD_V2a_V1[k,l]/(dt*cv))], VLV1[l,t-1], LSyn_V2a_V1[nV1*k+l,1], LSyn_V2a_V1[nV1*k+l,2])
                RSyn_V2a_V1[nV1*k+l,:] = R_glusyn_V2a_V1[nV1*k+l].getNextVal(VRV2a[k,t-int(RD_V2a_V1[k,l]/(dt*cv))], VRV1[l,t-1], RSyn_V2a_V1[nV1*k+l,1], RSyn_V2a_V1[nV1*k+l,2])
                
        for k in range (0, nV2a):
            for l in range (0, nV0v):
                LSyn_V2a_V0v[nV0v*k+l,:] = L_glusyn_V2a_V0v[nV0v*k+l].getNextVal(VLV2a[k,t-int(LD_V2a_V0v[k,l]/(dt*cv))], VLV0v[l,t-1], LSyn_V2a_V0v[nV0v*k+l,1], LSyn_V2a_V0v[nV0v*k+l,2])
                RSyn_V2a_V0v[nV0v*k+l,:] = R_glusyn_V2a_V0v[nV0v*k+l].getNextVal(VRV2a[k,t-int(RD_V2a_V0v[k,l]/(dt*cv))], VRV0v[l,t-1], RSyn_V2a_V0v[nV0v*k+l,1], RSyn_V2a_V0v[nV0v*k+l,2])
                
        for k in range (0, nV1):
            for l in range (0, nMN):
                LSyn_V1_MN[nMN*k+l,:] = L_glysyn_V1_MN[nMN*k+l].getNextVal(VLV1[k,t-int(LD_V1_MN[k,l]/(dt*cv))], VLMN[l,t-1], LSyn_V1_MN[nMN*k+l,1], LSyn_V1_MN[nMN*k+l,2])
                RSyn_V1_MN[nMN*k+l,:] = R_glysyn_V1_MN[nMN*k+l].getNextVal(VRV1[k,t-int(RD_V1_MN[k,l]/(dt*cv))], VRMN[l,t-1], RSyn_V1_MN[nMN*k+l,1], RSyn_V1_MN[nMN*k+l,2])
                
        for k in range (0, nV1):
            for l in range (0, nV2a):
                LSyn_V1_V2a[nV2a*k+l,:] = L_glysyn_V1_V2a[nV2a*k+l].getNextVal(VLV1[k,t-int(LD_V2a_V1[l,k]/(dt*cv))], VLV2a[l,t-1], LSyn_V1_V2a[nV2a*k+l,1], LSyn_V1_V2a[nV2a*k+l,2])
                RSyn_V1_V2a[nV2a*k+l,:] = R_glysyn_V1_V2a[nV2a*k+l].getNextVal(VRV1[k,t-int(RD_V2a_V1[l,k]/(dt*cv))], VRV2a[l,t-1], RSyn_V1_V2a[nV2a*k+l,1], RSyn_V1_V2a[nV2a*k+l,2])
    
        for k in range (0, nV1):
            for l in range (0, nV0v):
                LSyn_V1_V0v[nV0v*k+l,:] = L_glysyn_V1_V0v[nV0v*k+l].getNextVal(VLV1[k,t-int(LD_V1_V0v[k,l]/(dt*cv))], VLV0v[l,t-1], LSyn_V1_V0v[nV0v*k+l,1], LSyn_V1_V0v[nV0v*k+l,2])
                RSyn_V1_V0v[nV0v*k+l,:] = R_glysyn_V1_V0v[nV0v*k+l].getNextVal(VRV1[k,t-int(RD_V1_V0v[k,l]/(dt*cv))], VRV0v[l,t-1], RSyn_V1_V0v[nV0v*k+l,1], RSyn_V1_V0v[nV0v*k+l,2])
                
        for k in range (0, nV1):
            for l in range (0, ndI6):
                LSyn_V1_dI6[ndI6*k+l,:] = L_glysyn_V1_dI6[ndI6*k+l].getNextVal(VLV1[k,t-int(LD_V1_dI6[k,l]/(dt*cv))], VLdI6[l,t-1], LSyn_V1_dI6[ndI6*k+l,1], LSyn_V1_dI6[ndI6*k+l,2])
                RSyn_V1_dI6[ndI6*k+l,:] = R_glysyn_V1_dI6[ndI6*k+l].getNextVal(VRV1[k,t-int(RD_V1_dI6[k,l]/(dt*cv))], VRdI6[l,t-1], RSyn_V1_dI6[ndI6*k+l,1], RSyn_V1_dI6[ndI6*k+l,2])
                
        for k in range (0, nMN):
            for l in range (0, nMuscle):
                LSyn_MN_Muscle[nMuscle*k+l,:] = L_glusyn_MN_Muscle[nMuscle*k+l].getNextVal(VLMN[k,t-10], VLMuscle[l,t-1], LSyn_MN_Muscle[nMuscle*k+l,1], LSyn_MN_Muscle[nMuscle*k+l,2])
                RSyn_MN_Muscle[nMuscle*k+l,:] = R_glusyn_MN_Muscle[nMuscle*k+l].getNextVal(VRMN[k,t-10], VRMuscle[l,t-1], RSyn_MN_Muscle[nMuscle*k+l,1], RSyn_MN_Muscle[nMuscle*k+l,2])
                
        for k in range (0, nMN):
            for l in range (0, nMN):   
                RSGap_MN_MN[k,l] = RGap_MN_MN[k,l]*(VRMN[k,t-int(RD_MN_MN[k,l]/(dt*cv))-1]-VRMN[l,t-1])
                LSGap_MN_MN[k,l] = LGap_MN_MN[k,l]*(VLMN[k,t-int(LD_MN_MN[k,l]/(dt*cv))-1]-VLMN[l,t-1])

        for k in range (0, nMN):
            for l in range (0, ndI6):   
                RSGap_MN_dI6[k,l] = RGap_MN_dI6[k,l]*(VRMN[k,t-int(RD_MN_dI6[k,l]/(dt*cv))-1]-VRdI6[l,t-1])
                LSGap_MN_dI6[k,l] = LGap_MN_dI6[k,l]*(VLMN[k,t-int(LD_MN_dI6[k,l]/(dt*cv))-1]-VLdI6[l,t-1])
                
        for k in range (0, nMN):
            for l in range (0, nV0v):   
                RSGap_MN_V0v[k,l] = RGap_MN_V0v[k,l]*(VRMN[k,t-int(RD_MN_V0v[k,l]/(dt*cv))-1]-VRV0v[l,t-1])
                LSGap_MN_V0v[k,l] = LGap_MN_V0v[k,l]*(VLMN[k,t-int(LD_MN_V0v[k,l]/(dt*cv))-1]-VLV0v[l,t-1])
                
        for k in range (0, nMN):
            for l in range (0, nV2a):   
                RSGap_MN_V2a[k,l] = RGap_V2a_MN[l,k]*(VRMN[k,t-int(RD_V2a_MN[l,k]/(dt*cv))]-VRV2a[l,t-1])
                LSGap_MN_V2a[k,l] = LGap_V2a_MN[l,k]*(VLMN[k,t-int(LD_V2a_MN[l,k]/(dt*cv))]-VLV2a[l,t-1])
    
        for k in range (0, ndI6):
            for l in range (0, nMN):   
                RSGap_dI6_MN[k,l] = RGap_MN_dI6[l,k]*(VRdI6[k,t-int(RD_MN_dI6[l,k]/(dt*cv))-1]-VRMN[l,t-1])
                LSGap_dI6_MN[k,l] = LGap_MN_dI6[l,k]*(VLdI6[k,t-int(LD_MN_dI6[l,k]/(dt*cv))-1]-VLMN[l,t-1])
        
        for k in range (0, ndI6):
            for l in range (0, ndI6):   
                RSGap_dI6_dI6[k,l] = RGap_dI6_dI6[k,l]*(VRdI6[k,t-int(RD_dI6_dI6[k,l]/(dt*cv))-1]-VRdI6[l,t-1])
                LSGap_dI6_dI6[k,l] = LGap_dI6_dI6[k,l]*(VLdI6[k,t-int(LD_dI6_dI6[k,l]/(dt*cv))-1]-VLdI6[l,t-1])
                        
        for k in range (0, nV0v):
            for l in range (0, nMN):   
                RSGap_V0v_MN[k,l] = RGap_MN_V0v[l,k]*(VRV0v[k,t-int(RD_MN_V0v[l,k]/(dt*cv))-1]-VRMN[l,t-1])
                LSGap_V0v_MN[k,l] = LGap_MN_V0v[l,k]*(VLV0v[k,t-int(LD_MN_V0v[l,k]/(dt*cv))-1]-VLMN[l,t-1])
                
        for k in range (0, nV0v):
            for l in range (0, nV0v):   
                RSGap_V0v_V0v[k,l] = RGap_V0v_V0v[k,l]*(VRV0v[k,t-int(RD_V0v_V0v[k,l]/(dt*cv))-1]-VRV0v[l,t-1])
                LSGap_V0v_V0v[k,l] = LGap_V0v_V0v[k,l]*(VLV0v[k,t-int(LD_V0v_V0v[k,l]/(dt*cv))-1]-VLV0v[l,t-1])    
                
        for k in range (0, nV2a):
            for l in range (0, nV2a):   
                RSGap_V2a_V2a[k,l] = RGap_V2a_V2a[k,l]*(VRV2a[k,t-int(RD_V2a_V2a[k,l]/(dt*cv))]-VRV2a[l,t-1])
                LSGap_V2a_V2a[k,l] = LGap_V2a_V2a[k,l]*(VLV2a[k,t-int(LD_V2a_V2a[k,l]/(dt*cv))]-VLV2a[l,t-1])
                
        for k in range (0, nV2a):
            for l in range (0, nMN):   
                RSGap_V2a_MN[k,l] = RGap_V2a_MN[k,l]*(VRV2a[k,t-int(RD_V2a_MN[k,l]/(dt*cv))]-VRMN[l,t-1])
                LSGap_V2a_MN[k,l] = LGap_V2a_MN[k,l]*(VLV2a[k,t-int(LD_V2a_MN[k,l]/(dt*cv))]-VLMN[l,t-1])
                  
        ## Calculate membrane potentials
        
        for k in range (0, nMN):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(RSyn_dI6_MN[nMN*l+k,0]*LW_dI6_MN[l,k]*R_str for l in range (0, ndI6)) + sum(LSyn_V2a_MN[nMN*m+k,0]*LW_V2a_MN[m,k] for m in range (0, nV2a)) + sum(LSyn_V1_MN[nMN*p+k,0]*LW_V1_MN[p,k]*R_str for p in range (0, nV1))
                IsynR = sum(LSyn_dI6_MN[nMN*l+k,0]*RW_dI6_MN[l,k]*R_str for l in range (0, ndI6)) + sum(RSyn_V2a_MN[nMN*m+k,0]*RW_V2a_MN[m,k] for m in range (0, nV2a)) + sum(RSyn_V1_MN[nMN*p+k,0]*RW_V1_MN[p,k]*R_str for p in range (0, nV1))
                
            resLMN[k,:] = L_MN[k].getNextVal(resLMN[k,0],resLMN[k,1], - sum(LSGap_MN_MN[k,:]) + sum(LSGap_MN_MN[:,k]) - sum(LSGap_MN_dI6[k,:]) + sum(LSGap_dI6_MN[:,k]) - sum(LSGap_MN_V0v[k,:]) + sum(LSGap_V0v_MN[:,k]) - sum(LSGap_MN_V2a[k,:]) + sum(LSGap_V2a_MN[:,k]) + IsynL)
            VLMN[k,t] = resLMN[k,0]
            
            resRMN[k,:] = R_MN[k].getNextVal(resRMN[k,0],resRMN[k,1], - sum(RSGap_MN_MN[k,:]) + sum(RSGap_MN_MN[:,k]) - sum(RSGap_MN_dI6[k,:]) + sum(RSGap_dI6_MN[:,k]) - sum(RSGap_MN_V0v[k,:]) + sum(RSGap_V0v_MN[:,k]) - sum(LSGap_MN_V2a[k,:]) + sum(LSGap_V2a_MN[:,k]) + IsynR)
            VRMN[k,t] = resRMN[k,0]
    
    
        for k in range (0, ndI6):
            if t < 500: 
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(LSyn_V2a_dI6[ndI6*l+k,0]*LW_V2a_dI6[l,k] for l in range (0, nV2a)) + sum(RSyn_dI6_dI6[ndI6*l+k,0]*LW_dI6_dI6[l,k]*R_str for l in range (0, ndI6)) + sum(LSyn_V1_dI6[ndI6*l+k,0]*LW_V1_dI6[l,k]*R_str for l in range (0, nV1))
                IsynR = sum(RSyn_V2a_dI6[ndI6*l+k,0]*RW_V2a_dI6[l,k] for l in range (0, nV2a)) + sum(LSyn_dI6_dI6[ndI6*l+k,0]*RW_dI6_dI6[l,k]*R_str for l in range (0, ndI6)) + sum(RSyn_V1_dI6[ndI6*l+k,0]*RW_V1_dI6[l,k]*R_str for l in range (0, nV1))
                
            resLdI6[k,:] = L_dI6[k].getNextVal(resLdI6[k,0],resLdI6[k,1], - sum(LSGap_dI6_dI6[k,:]) + sum(LSGap_dI6_dI6[:,k]) - sum(LSGap_dI6_MN[k,:]) + sum(LSGap_MN_dI6[:,k]) + IsynL)
            VLdI6[k,t] = resLdI6[k,0]
            resRdI6[k,:] = R_dI6[k].getNextVal(resRdI6[k,0],resRdI6[k,1], - sum(RSGap_dI6_dI6[k,:]) + sum(RSGap_dI6_dI6[:,k]) - sum(RSGap_dI6_MN[k,:]) + sum(RSGap_MN_dI6[:,k]) + IsynR)
            VRdI6[k,t] = resRdI6[k,0]
            
        for k in range (0, nV0v):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(LSyn_V2a_V0v[nV0v*l+k,0]*LW_V2a_V0v[l,k] for l in range (0, nV2a)) + sum(LSyn_V1_V0v[nV0v*l+k,0]*LW_V1_V0v[l,k]*R_str for l in range (0, nV1))
                IsynR = sum(RSyn_V2a_V0v[nV0v*l+k,0]*RW_V2a_V0v[l,k] for l in range (0, nV2a)) + sum(RSyn_V1_V0v[nV0v*l+k,0]*RW_V1_V0v[l,k]*R_str for l in range (0, nV1))
               
            resLV0v[k,:] = L_V0v[k].getNextVal(resLV0v[k,0],resLV0v[k,1],  - sum(LSGap_V0v_V0v[k,:]) + sum(LSGap_V0v_V0v[:,k]) -sum(LSGap_V0v_MN[k,:])  + sum(LSGap_MN_V0v[:,k]) + IsynL)
            VLV0v[k,t] = resLV0v[k,0]
            resRV0v[k,:] = R_V0v[k].getNextVal(resRV0v[k,0],resRV0v[k,1],  - sum(RSGap_V0v_V0v[k,:]) + sum(RSGap_V0v_V0v[:,k]) -sum(RSGap_V0v_MN[k,:])  + sum(RSGap_MN_V0v[:,k]) + IsynR)
            VRV0v[k,t] = resRV0v[k,0]
            
        for k in range (0, nV2a):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(LSyn_V2a_V2a[nV2a*m+k,0]*LW_V2a_V2a[m,k] for m in range (0, nV2a)) + sum(RSyn_dI6_V2a[nV2a*l+k,0]*RW_dI6_V2a[l,k]*R_str for l in range (0, ndI6)) + sum(LSyn_V1_V2a[nV2a*p+k,0]*LW_V1_V2a[p,k]*R_str for p in range (0, nV1)) + sum(RSyn_V0v_V2a[nV2a*l+k,0]*LW_V0v_V2a[l,k] for l in range (0, nV0v))
                IsynR = sum(RSyn_V2a_V2a[nV2a*m+k,0]*RW_V2a_V2a[m,k] for m in range (0, nV2a)) + sum(LSyn_dI6_V2a[nV2a*l+k,0]*LW_dI6_V2a[l,k]*R_str for l in range (0, ndI6)) + sum(RSyn_V1_V2a[nV2a*p+k,0]*RW_V1_V2a[p,k]*R_str for p in range (0, nV1)) + sum(LSyn_V0v_V2a[nV2a*l+k,0]*RW_V0v_V2a[l,k] for l in range (0, nV0v))
            if k<20:
                IsynL = IsynL + gauss(1.0,sigma)*stim2[t-180-32*k] #* (nV2a-k)/nV2a        # Tonic drive for the all V2as # (nV2a-k)/nV2a is to produce a decreasing gradient of descending drive # 32*k represents the conduction delay, which is 1.6 ms according to McDermid and Drapeau JNeurophysiol (2006). Since we consider each somite to be two real somites, then 16*2 
                IsynR = IsynR + gauss(1.0,sigma)*stim2[t-180-32*k] #* (nV2a-k)/nV2a
                
            resLV2a[k,:] = L_V2a[k].getNextVal(resLV2a[k,0],resLV2a[k,1], - sum(LSGap_V2a_V2a[k,:]) + sum(LSGap_V2a_V2a[:,k]) - sum(LSGap_V2a_MN[k,:])+ sum(LSGap_MN_V2a[:,k]) + IsynL)         
            VLV2a[k,t] = resLV2a[k,0]
            resRV2a[k,:] = R_V2a[k].getNextVal(resRV2a[k,0],resRV2a[k,1],  - sum(RSGap_V2a_V2a[k,:]) + sum(RSGap_V2a_V2a[:,k]) - sum(RSGap_V2a_MN[k,:])+ sum(RSGap_MN_V2a[:,k])  + IsynR)    
            VRV2a[k,t] = resRV2a[k,0]
        
        for k in range (0, nV1):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(LSyn_V2a_V1[nV1*m+k,0]*LW_V2a_V1[m,k] for m in range (0, nV2a))
                IsynR = sum(RSyn_V2a_V1[nV1*m+k,0]*RW_V2a_V1[m,k] for m in range (0, nV2a))
            resLV1[k,:] = L_V1[k].getNextVal(resLV1[k,0],resLV1[k,1], IsynL)  
            VLV1[k,t] = resLV1[k,0]
            resRV1[k,:] = R_V1[k].getNextVal(resRV1[k,0],resRV1[k,1], IsynR)  
            VRV1[k,t] = resRV1[k,0]
            
        for k in range (0, nMuscle):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(LSyn_MN_Muscle[nMuscle*l+k,0]*LW_MN_Muscle[l,k] for l in range (0, nMN))
                IsynR = sum(RSyn_MN_Muscle[nMuscle*l+k,0]*RW_MN_Muscle[l,k] for l in range (0, nMN))
                
            resLMuscle[k,:] = L_Muscle[k].getNextVal(resLMuscle[k,0], IsynL + 0.4*gauss(1.0,sigma)-0.4) #the last term is to add variability but equals 0 if sigma = 0
            VLMuscle[k,t] = resLMuscle[k,0]
            
            resRMuscle[k,:] = R_Muscle[k].getNextVal(resRMuscle[k,0], IsynR + 0.4*gauss(1.0,sigma)-0.4) #the last term is to add variability but equals 0 if sigma = 0
            VRMuscle[k,t] = resRMuscle[k,0]

    
    VLMNnew = VLMN[:,2000:]
    VRMNnew = VRMN[:,2000:]
    
    VLdI6new = VLdI6[:,2000:]
    VRdI6new = VRdI6[:,2000:]
    
    VLV0vnew = VLV0v[:,2000:]
    VRV0vnew = VRV0v[:,2000:]
    
    VLV2anew = VLV2a[:,2000:]
    VRV2anew = VRV2a[:,2000:]
    
    VLV1new = VLV1[:,2000:]
    VRV1new = VRV1[:,2000:]
    
    VLMusclenew = VLMuscle[:,2000:]
    VRMusclenew = VRMuscle[:,2000:]
    
    Timenew = Time[2000:]-Time[2000:][0]
    
    
    return (VLMNnew, VRMNnew), (VLdI6new, VRdI6new), (VLV0vnew, VRV0vnew), (VLV2anew, VRV2anew), (VLV1new, VRV1new), (VLMusclenew, VRMusclenew), Timenew

