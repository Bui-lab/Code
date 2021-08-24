#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
"""

from random import *
from Izhikevich_class import * # Where definition of single cell models are found based on Izhikevich models
from Analysis_tools import *

# This function sets up the connectome and runs a simulation for the time tmax.
# rand is a seed for a random function
# stim0 is a constant for scaling the drive to IC neurons
# sigma is a variance for gaussian randomization of the gap junction coupling and synaptic weights
# dt is the time step size
# E_glu and E_gly are the reversal potential of glutamate and glycine respectively
# cv is the transmission speed
# nIC, nMN, nV2a, nV0d, nV0v and nMuscle is the number of IC, MN, V2a, V0d, V0v and Muscle cells

def connectome_double_coiling(rand=0, stim0=8, sigma=0,
                                tmax=1000, dt=0.1, E_glu=0, E_gly=-70, cv=0.55,
                                nIC=5, nMN=10, nV2a = 10, nV0d=10, nV0v=10, nMuscle=10, weight_V0d_MN=0.3):
    
    seed(rand)
    
    ## Declare constants    
    tmax += 200 # add 200 ms to give time for IC to settle.  This initial 200 ms is not used in the data analysis    
    nmax = int(tmax/dt)
    
    ## Declare Neuron Types

    L_IC = [ Izhikevich_9P(a=0.0002,b=0.5,c=-40, d=5, vmax=0, vr=-60, vt=-45, k=0.3, Cm = 50, dt=dt, x=1.0,y=-1) for i in range(nIC)]
    R_IC = L_IC = [ Izhikevich_9P(a=0.0002,b=0.5,c=-40, d=5, vmax=0, vr=-60, vt=-45, k=0.3, Cm = 50, dt=dt, x=1.0,y=-1) for i in range(nIC)]

    L_MN = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-50, k=0.05, Cm = 20, dt=dt, x=5.0+1.6*i,y=-1) for i in range(nMN)]
    R_MN = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-50, k=0.05, Cm = 20, dt=dt, x=5.0+1.6*i,y=1) for i in range(nMN)]
    
    L_V0d = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=3.75, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt=dt, x=5.0+1.6*i,y=-1) for i in range(nV0d)]
    R_V0d = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=3.75, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt=dt, x=5.0+1.6*i,y=1) for i in range(nV0d)]
    
    L_V0v = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=11.6, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt=dt, x=5.1+1.6*i,y=-1) for i in range(nV0v)]
    R_V0v = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=11.6, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt=dt, x=5.1+1.6*i,y=1) for i in range(nV0v)]
    
    L_V2a = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt=dt, x=5.1+1.6*i,y=-1) for i in range(nV2a)]
    R_V2a = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt=dt, x=5.1+1.6*i,y=1) for i in range(nV2a)]
    
    L_Muscle = [ Leaky_Integrator(50.0, 5.0, dt, 5.0+1.6*i,-1) for i in range(nMuscle)]
    R_Muscle = [ Leaky_Integrator(50.0, 5.0, dt, 5.0+1.6*i,-1) for i in range(nMuscle)]
    
    
    ## Declare Synapses
    
    # Below is the declaration of the chemical synapses
    
    L_achsyn_MN_Muscle = [TwoExp_syn(0.5, 1.0, -15, dt, 120) for i in range (nMN*nMuscle)]
    R_achsyn_MN_Muscle = [TwoExp_syn(0.5, 1.0, -15, dt, 120) for i in range (nMN*nMuscle)]
    L_glysyn_V0d_MN = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV0d*nMN)]
    R_glysyn_V0d_MN = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV0d*nMN)]
    L_glysyn_V0d_IC = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV0d*nIC)]
    R_glysyn_V0d_IC = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV0d*nIC)]
    L_glysyn_V0d_V2a = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV0d*nV2a)]
    R_glysyn_V0d_V2a = [ TwoExp_syn(0.5, 1.0, -15, dt, E_gly) for i in range (nV0d*nV2a)]
    
    L_glusyn_V2a_V0v = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV0v)]
    R_glusyn_V2a_V0v = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV2a*nV0v)]

    L_glusyn_V0v_IC = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV0v*nIC)]
    R_glusyn_V0v_IC = [TwoExp_syn(0.5, 1.0, -15, dt, E_glu) for i in range (nV0v*nIC)]
    
    ## Declare Storage tables
    
    Time =zeros(nmax)
    
    VLIC =zeros((nIC, nmax))
    VRIC =zeros((nIC, nmax))
    VLMN =zeros((nMN, nmax))
    VRMN =zeros((nMN, nmax))
    VLV0d = zeros ((nV0d,nmax))
    VRV0d = zeros ((nV0d,nmax))
    VLV0v = zeros ((nV0v,nmax))
    VRV0v = zeros ((nV0v,nmax))
    VLV2a = zeros ((nV2a,nmax))
    VRV2a = zeros ((nV2a,nmax))
    VLMuscle = zeros((nMuscle, nmax))
    VRMuscle = zeros((nMuscle, nmax))
    
    #storing synaptic currents
    
    #gly
    LSyn_V0d_MN = zeros((nV0d*nMN,3))
    RSyn_V0d_MN = zeros((nV0d*nMN,3))
    LSyn_V0d_IC = zeros((nV0d*nIC,3))
    RSyn_V0d_IC = zeros((nV0d*nIC,3))
    LSyn_V0d_V2a = zeros((nV0d*nV2a,3))
    RSyn_V0d_V2a = zeros((nV0d*nV2a,3))
    #ach
    LSyn_MN_Muscle = zeros((nMN*nMuscle,3))
    RSyn_MN_Muscle = zeros((nMN*nMuscle,3))
    #glu
    LSyn_V0v_IC = zeros((nV0v*nIC,3))
    RSyn_V0v_IC = zeros((nV0v*nIC,3))
    LSyn_V2a_V0v = zeros((nV2a*nV0v,3))
    RSyn_V2a_V0v = zeros((nV2a*nV0v,3))
    
    LSGap_IC_IC = zeros((nIC,nIC))
    LSGap_IC_MN = zeros((nIC,nMN))      
    LSGap_IC_V0d = zeros((nIC,nV0d))
    LSGap_IC_V0v = zeros((nIC,nV0v))
    LSGap_IC_V2a = zeros((nIC,nV2a))
    LSGap_MN_IC = zeros((nMN,nIC))
    LSGap_MN_MN = zeros((nMN,nMN))
    LSGap_MN_V0d = zeros((nMN,nV0d))
    LSGap_MN_V0v = zeros((nMN,nV0v))
    LSGap_MN_V2a = zeros((nMN,nV2a))
    LSGap_V0d_IC = zeros((nV0d,nIC))
    LSGap_V0d_MN = zeros((nV0d,nMN))
    LSGap_V0d_V0d = zeros((nV0d,nV0d))
    LSGap_V0v_IC = zeros((nV0v,nIC))
    LSGap_V0v_MN = zeros((nV0v,nMN))
    LSGap_V0v_V0v = zeros((nV0v,nV0v))
    LSGap_V2a_IC = zeros((nV2a,nIC))
    LSGap_V2a_V2a = zeros((nV2a,nV2a))
    LSGap_V2a_MN = zeros((nV2a,nMN))
    
    RSGap_IC_IC = zeros((nIC,nIC))
    RSGap_IC_MN = zeros((nIC,nMN))      
    RSGap_IC_V0d = zeros((nIC,nV0d))
    RSGap_IC_V0v = zeros((nIC,nV0v))
    RSGap_IC_V2a = zeros((nIC,nV2a))
    RSGap_MN_IC = zeros((nMN,nIC))
    RSGap_MN_MN = zeros((nMN,nMN))
    RSGap_MN_V0d = zeros((nMN,nV0d))
    RSGap_MN_V0v = zeros((nMN,nV0v))
    RSGap_MN_V2a = zeros((nMN,nV2a))
    RSGap_V0d_IC = zeros((nV0d,nIC))
    RSGap_V0d_MN = zeros((nV0d,nMN))
    RSGap_V0d_V0d = zeros((nV0d,nV0d))
    RSGap_V0v_IC = zeros((nV0v,nIC))
    RSGap_V0v_MN = zeros((nV0v,nMN))
    RSGap_V0v_V0v = zeros((nV0v,nV0v))
    RSGap_V2a_IC = zeros((nV2a,nIC))
    RSGap_V2a_V2a = zeros((nV2a,nV2a))
    RSGap_V2a_MN = zeros((nV2a,nMN))    
    
        #Synaptic weight
    LW_V0d_MN = zeros((nV0d,nMN))      
    RW_V0d_MN = zeros((nV0d,nMN))
    LW_V0d_IC = zeros((nV0d,nIC))      
    RW_V0d_IC = zeros((nV0d,nIC))
    LW_V0d_V2a = zeros((nV0d,nV2a))      
    RW_V0d_V2a = zeros((nV0d,nV2a))
    LW_V0v_IC = zeros((nV0v,nIC))
    RW_V0v_IC = zeros((nV0v,nIC))
    LW_V2a_V0v = zeros((nV2a,nV0v))
    RW_V2a_V0v = zeros((nV2a,nV0v))
    LW_MN_Muscle = zeros((nMN,nMuscle))
    RW_MN_Muscle = zeros((nMN,nMuscle))
    
        #Gap junctions coupling
    LGap_IC_MN = zeros((nIC,nMN))      
    LGap_IC_V0d = zeros((nIC,nV0d))
    LGap_IC_IC = zeros((nIC,nIC))
    LGap_IC_V0v = zeros((nIC,nV0v))
    LGap_IC_V2a = zeros((nIC,nV2a))
    LGap_MN_MN = zeros((nMN,nMN))
    LGap_MN_V0d = zeros((nMN,nV0d))
    LGap_MN_V0v = zeros((nMN,nV0v))
    LGap_V0d_V0d = zeros((nV0d,nV0d))
    LGap_V0v_V0v = zeros((nV0v,nV0v))
    LGap_V2a_V2a = zeros((nV2a,nV2a))
    LGap_V2a_MN = zeros((nV2a,nMN))
    
    RGap_IC_MN = zeros((nIC,nMN))      
    RGap_IC_V0d = zeros((nIC,nV0d))
    RGap_IC_IC = zeros((nIC,nIC))
    RGap_IC_V0v = zeros((nIC,nV0v))
    RGap_IC_V2a = zeros((nIC,nV2a))
    RGap_MN_MN = zeros((nMN,nMN))
    RGap_MN_V0d = zeros((nMN,nV0d))
    RGap_MN_V0v = zeros((nMN,nV0v))
    RGap_V0d_V0d = zeros((nV0d,nV0d))
    RGap_V0v_V0v = zeros((nV0v,nV0v))
    RGap_V2a_V2a = zeros((nV2a,nV2a))
    RGap_V2a_MN = zeros((nV2a,nMN))

    
        #res
    resLIC=zeros((nIC,3))
    resRIC=zeros((nIC,3))
    resLMN=zeros((nMN,3))
    resRMN=zeros((nMN,3))
    resLV0d = zeros((nV0d,3))
    resRV0d = zeros((nV0d,3))
    resLV0v = zeros((nV0v,3))
    resRV0v = zeros((nV0v,3))
    resLV2a = zeros((nV2a,3))
    resRV2a = zeros((nV2a,3))
    resLMuscle = zeros((nMuscle,2))
    resRMuscle = zeros((nMuscle,2))
    
    stim= zeros (nmax)
    
        #Distance Matrix 
    LD_IC_MN = zeros((nIC, nMN))       
    LD_IC_V0d = zeros((nIC,nV0d))
    LD_IC_V0v = zeros((nIC,nV0v))
    LD_IC_V2a = zeros((nIC,nV2a))  
    LD_MN_MN = zeros((nMN,nMN))
    LD_MN_V0d = zeros((nMN,nV0d))
    LD_MN_V0v = zeros((nMN,nV0v))
    LD_V0d_V0d = zeros((nV0d,nV0d))
    LD_V0d_IC = zeros((nV0d,nV0d))
    LD_V0d_MN = zeros((nV0d,nMN))
    LD_V0d_V2a = zeros((nV0d,nV2a))
    LD_V0v_IC = zeros((nV0v,nIC))    
    LD_V0v_V0v = zeros((nV0v,nV0v))
    LD_V2a_V2a = zeros((nV2a,nV2a))
    LD_V2a_MN = zeros((nV2a,nMN))
    LD_V2a_V0v = zeros((nV2a,nV0v))  
        
    RD_IC_MN = zeros((nIC, nMN))     
    RD_IC_V0d = zeros((nIC,nV0d))
    RD_IC_V0v = zeros((nIC,nV0v))
    RD_IC_V2a = zeros((nIC,nV2a)) 
    RD_MN_MN = zeros((nMN,nMN))
    RD_MN_V0d = zeros((nMN,nV0d))
    RD_MN_V0v = zeros((nMN,nV0v))
    RD_V0d_V0d = zeros((nV0d,nV0d))
    RD_V0d_IC = zeros((nV0d,nV0d))
    RD_V0d_MN = zeros((nV0d,nMN))
    RD_V0d_V2a = zeros((nV0d,nV2a))
    RD_V0v_V0v = zeros((nV0v,nV0v))
    RD_V0v_IC = zeros((nV0v,nIC))    
    RD_V2a_V2a = zeros((nV2a,nV2a))
    RD_V2a_MN = zeros((nV2a,nMN))
    RD_V2a_V0v = zeros((nV2a,nV0v))  
    
    ## Compute distance between Neurons
    
    #LEFT
    for k in range (0, nIC):
        for l in range (0, nMN):
            LD_IC_MN[k,l] = Distance(L_IC[k].x,L_MN[l].x,L_IC[k].y,L_MN[l].y)
    
    for k in range (0, nIC):
        for l in range (0, nV0d):
            LD_IC_V0d[k,l] = Distance(L_IC[k].x,L_V0d[l].x,L_IC[k].y,L_V0d[l].y)
            
    for k in range (0, nIC):
        for l in range (0, nV0v):
            LD_IC_V0v[k,l] = Distance(L_IC[k].x,L_V0v[l].x,L_IC[k].y,L_V0v[l].y)
    
    for k in range (0, nIC):
        for l in range (0, nV2a):
            LD_IC_V2a[k,l] = Distance(L_IC[k].x,L_V2a[l].x,L_IC[k].y,L_V2a[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nMN):
            LD_MN_MN[k,l] = Distance(L_MN[k].x,L_MN[l].x,L_MN[k].y,L_MN[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nV0d):
            LD_MN_V0d[k,l] = Distance(L_MN[k].x,L_V0d[l].x,L_MN[k].y,L_V0d[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nV0v):
            LD_MN_V0v[k,l] = Distance(L_MN[k].x,L_V0v[l].x,L_MN[k].y,L_V0v[l].y)
            
    for k in range (0, nV0d):
        for l in range (0, nV0d):
            LD_V0d_V0d[k,l] = Distance(L_V0d[k].x,L_V0d[l].x,L_V0d[k].y,L_V0d[l].y)
            
    for k in range (0, nV0d):
        for l in range (0, nMN):
            LD_V0d_MN[k,l] = Distance(L_V0d[k].x,R_MN[l].x,L_V0d[k].y,R_MN[l].y) #Contalateral
                        
    for k in range (0, nV0d):
        for l in range (0, nIC):
            LD_V0d_IC[k,l] = Distance(L_V0d[k].x,R_IC[l].x,L_V0d[k].y,R_IC[l].y) #Contalateral

    for k in range (0, nV0d):
        for l in range (0, nV2a):
            LD_V0d_V2a[k,l] = Distance(L_V0d[k].x,R_V2a[l].x,L_V0d[k].y,R_V2a[l].y) #Contalateral
            
    for k in range (0, nV0v):
        for l in range (0, nIC):
            LD_V0v_IC[k,l] = Distance(L_V0v[k].x,R_IC[l].x,L_V0v[k].y,R_IC[l].y) #Contalateral
    
    for k in range (0, nV0v):
        for l in range (0, nV0v):
            LD_V0v_V0v[k,l] = Distance(L_V0v[k].x,L_V0v[l].x,L_V0v[k].y,L_V0v[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            LD_V2a_V2a[k,l] = Distance(L_V2a[k].x,L_V2a[l].x,L_V2a[k].y,L_V2a[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nMN):
            LD_V2a_MN[k,l] = Distance(L_V2a[k].x,L_MN[l].x,L_V2a[k].y,L_MN[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nV0v):
            LD_V2a_V0v[k,l] = Distance(L_V2a[k].x,L_V0v[l].x,L_V2a[k].y,L_V0v[l].y)
    
    #RIGHT
    for k in range (0, nIC):
        for l in range (0, nMN):
            RD_IC_MN[k,l] = Distance(R_IC[k].x,R_MN[l].x,R_IC[k].y,R_MN[l].y)
    
    for k in range (0, nIC):
        for l in range (0, nV0d):
            RD_IC_V0d[k,l] = Distance(R_IC[k].x,R_V0d[l].x,R_IC[k].y,R_V0d[l].y)
            
    for k in range (0, nIC):
        for l in range (0, nV0v):
            RD_IC_V0v[k,l] = Distance(R_IC[k].x,R_V0v[l].x,R_IC[k].y,R_V0v[l].y)
            
    for k in range (0, nIC):
        for l in range (0, nV2a):
            RD_IC_V2a[k,l] = Distance(R_IC[k].x,R_V2a[l].x,R_IC[k].y,R_V2a[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nMN):
            RD_MN_MN[k,l] = Distance(R_MN[k].x,R_MN[l].x,R_MN[k].y,R_MN[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nV0d):
            RD_MN_V0d[k,l] = Distance(R_MN[k].x,R_V0d[l].x,R_MN[k].y,R_V0d[l].y)
            
    for k in range (0, nMN):
        for l in range (0, nV0v):
            RD_MN_V0v[k,l] = Distance(R_MN[k].x,R_V0v[l].x,R_MN[k].y,R_V0v[l].y)

    for k in range (0, nV0d):
        for l in range (0, nIC):
            RD_V0d_IC[k,l] = Distance(R_V0d[k].x,L_IC[l].x,R_V0d[k].y,L_IC[l].y) #Contalateral
            
    for k in range (0, nV0d):
        for l in range (0, nMN):
            RD_V0d_MN[k,l] = Distance(R_V0d[k].x,L_MN[l].x,R_V0d[k].y,L_MN[l].y) #Contalateral        
    
    for k in range (0, nV0d):
        for l in range (0, nV0d):
            RD_V0d_V0d[k,l] = Distance(R_V0d[k].x,R_V0d[l].x,R_V0d[k].y,R_V0d[l].y)

    for k in range (0, nV0d):
        for l in range (0, nV2a):
            RD_V0d_V2a[k,l] = Distance(R_V0d[k].x,L_V2a[l].x,R_V0d[k].y,L_V2a[l].y) #Contalateral        

    for k in range (0, nV0v):
        for l in range (0, nV0v):
            RD_V0v_V0v[k,l] = Distance(R_V0v[k].x,R_V0v[l].x,R_V0v[k].y,R_V0v[l].y)
                
    for k in range (0, nV0v):
        for l in range (0, nIC):
            RD_V0v_IC[k,l] = Distance(R_V0v[k].x,L_IC[l].x,R_V0v[k].y,L_IC[l].y) #Contalateral
            
    for k in range (0, nV2a):
        for l in range (0, nMN):
            RD_V2a_MN[k,l] = Distance(R_V2a[k].x,R_MN[l].x,R_V2a[k].y,R_MN[l].y)
            
    for k in range (0, nV2a):
        for l in range (0, nV2a):
            RD_V2a_V2a[k,l] = Distance(R_V2a[k].x,R_V2a[l].x,R_V2a[k].y,R_V2a[l].y)
    
    for k in range (0, nV2a):
        for l in range (0, nV0v):
            RD_V2a_V0v[k,l] = Distance(R_V2a[k].x,R_V0v[l].x,R_V2a[k].y,R_V0v[l].y)

    
    ## Compute synaptic weights and Gaps
    
    #LEFT
    IC_IC_gap_weight = 0.0001
    for k in range (0, nIC):
        for l in range (0, nIC):
            if (k!= l):                                         # because it is a kernel of ICs, there is no distance condition
                LGap_IC_IC[k,l] = IC_IC_gap_weight*gauss(1, sigma) 
            else:
                LGap_IC_IC[k,l] = 0.0
   
    IC_MN_gap_weight = 0.03 
    for k in range (0, nIC):
        for l in range (0, nMN):
            if (0.2<LD_IC_MN[k,l]<10 and L_IC[k].x < L_MN[l].x):     #the second condition is because the connection is descending
                LGap_IC_MN[k,l] = IC_MN_gap_weight*gauss(1, sigma)
            else:
                LGap_IC_MN[k,l] = 0.0

    IC_V0d_gap_weight = 0.05 
    for k in range (0, nIC):
        for l in range (0, nV0d):
            if (0.2<LD_IC_V0d[k,l]<10 and L_IC[k].x < L_V0d[l].x):     #the second condition is because the connection is descending
                LGap_IC_V0d[k,l] = IC_V0d_gap_weight*gauss(1, sigma)
            else:
                LGap_IC_V0d[k,l] = 0.0
    
    IC_V0v_gap_weight = 0.00005 
    for k in range (0, nIC):
        for l in range (0, nV0v):
            if (0.2<LD_IC_V0v[k,l]<10 and L_IC[k].x < L_V0v[l].x):     #the second condition is because the connection is descending
                LGap_IC_V0v[k,l] = IC_V0v_gap_weight*gauss(1, sigma)
            else:
                LGap_IC_V0v[k,l] = 0.0
                
    IC_V2a_gap_weight = 0.15 
    for k in range (0, nIC):
        for l in range (0, nV2a):
            if (0.2<LD_IC_V2a[k,l]<10 and L_IC[k].x < L_V2a[l].x):     #the second condition is because the connection is descending
                LGap_IC_V2a[k,l] = IC_V2a_gap_weight *gauss(1, sigma)
            else:
                LGap_IC_V2a[k,l] = 0.0
    
    MN_MN_gap_weight =  0.07 
    for k in range (0, nMN):
        for l in range (0, nMN):
            if (0.2<LD_MN_MN[k,l]< 6.5 ):    
                LGap_MN_MN[k,l] = MN_MN_gap_weight*gauss(1, sigma)
            else:
                LGap_MN_MN[k,l] = 0.0
                
    V0d_V0d_gap_weight = 0.04
    for k in range (0, nV0d):
        for l in range (0, nV0d):
            if (0.2<LD_V0d_V0d[k,l]< 3.5 ):    
                LGap_V0d_V0d[k,l] = V0d_V0d_gap_weight*gauss(1, sigma)
            else:
                LGap_V0d_V0d[k,l] = 0.0
    
    V0v_V0v_gap_weight = 0.05
    for k in range (0, nV0v):
        for l in range (0, nV0v):
            if (0.2<LD_V0v_V0v[k,l]< 3.5 ):    
                LGap_V0v_V0v[k,l] = V0v_V0v_gap_weight*gauss(1, sigma)
            else:
                LGap_V0v_V0v[k,l] = 0.0
    
    MN_V0d_gap_weight = 0.0001  
    for k in range (0, nMN):
        for l in range (0, nV0d):
            if (L_V0d[l].x-1.5 <L_MN[k].x< L_V0d[l].x + 1.5 ):    
                LGap_MN_V0d[k,l] = MN_V0d_gap_weight *gauss(1, sigma) 
            else:
                LGap_MN_V0d[k,l] = 0.0
    
    MN_V0v_gap_weight = 0.0001  
    for k in range (0, nMN):
        for l in range (0, nV0v):
            if (L_V0v[l].x-1.5 <L_MN[k].x< L_V0v[l].x + 1.5 ):    
                LGap_MN_V0v[k,l] = MN_V0v_gap_weight*gauss(1, sigma)
            else:
                LGap_MN_V0v[k,l] = 0.0
                
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
                
    V0d_MN_syn_weight = weight_V0d_MN
    for k in range (0, nV0d):
        for l in range (0, nMN):
            if (0.2<LD_V0d_MN[k,l]<8.0):
                LW_V0d_MN[k,l] = V0d_MN_syn_weight*gauss(1, sigma)
            else:
                LW_V0d_MN[k,l] = 0.0
           
    V0d_IC_syn_weight = weight_V0d_MN
    for k in range (0, nV0d):
        for l in range (0, nIC):
            if (0.2<LD_V0d_IC[k,l]<20.0): 
                LW_V0d_IC[k,l] = V0d_IC_syn_weight*gauss(1, sigma)
            else:
                LW_V0d_IC[k,l] = 0.0
                           
    V0d_V2a_syn_weight = weight_V0d_MN
    for k in range (0, nV0d):
        for l in range (0, nV2a):
            if (0.2<LD_V0d_V2a[k,l]<8.0):
                LW_V0d_V2a[k,l] = V0d_V2a_syn_weight*gauss(1, sigma)
            else:
                LW_V0d_V2a[k,l] = 0.0
                
    V0v_IC_syn_weight = 0.2 # 0.175 value in base model 
    for k in range (0, nV0v):
        for l in range (0, nIC):
            if (6.0<LD_V0v_IC[k,l]<20.0):  
                LW_V0v_IC[k,l] =  V0v_IC_syn_weight*gauss(1, sigma)
            else:
                LW_V0v_IC[k,l] = 0.0
    
    V2a_V0v_syn_weight = 0.2 # 0.04 value in base model         
    for k in range (0, nV2a):
        for l in range (0, nV0v):
            if (0.2<LD_V2a_V0v[k,l]<10 and L_V2a[k].x < L_V0v[l].x) or (0.2<LD_V2a_V0v[k,l]<4 and L_V2a[k].x > L_V0v[l].x):    #descending and short ascending
                LW_V2a_V0v[k,l] = V2a_V0v_syn_weight*gauss(1, sigma)
            else:
                LW_V2a_V0v[k,l] = 0.0    
    
    MN_Muscle_syn_weight = 0.02
    for k in range (0, nMN):
        for l in range (0, nMuscle):
            if (L_Muscle[l].x-1<L_MN[k].x< L_Muscle[l].x+1):         #because segments
                LW_MN_Muscle[k,l] = MN_Muscle_syn_weight*gauss(1, sigma)
            else:
                LW_MN_Muscle[k,l] = 0.0
    
    #RIGHT
    for k in range (0, nIC):
        for l in range (0, nIC):
            if (k!= l):                                         # because it is a kernel of ICs, there is no distance condition
                RGap_IC_IC[k,l] = IC_IC_gap_weight*gauss(1, sigma)
            else:
                RGap_IC_IC[k,l] = 0.0
                
    for k in range (0, nIC):
        for l in range (0, nMN):
            if (0.2<RD_IC_MN[k,l]<10 and R_IC[k].x < R_MN[l].x):     #the second condition is because the connection is descending
                RGap_IC_MN[k,l] = IC_MN_gap_weight*gauss(1, sigma)
            else:
                RGap_IC_MN[k,l] = 0.0
                
    for k in range (0, nIC):
        for l in range (0, nV0d):
            if (0.2<RD_IC_V0d[k,l]<10 and R_IC[k].x < R_V0d[l].x):     #the second condition is because the connection is descending
                RGap_IC_V0d[k,l] = IC_V0d_gap_weight*gauss(1, sigma)
            else:
                RGap_IC_V0d[k,l] = 0.0
                
    for k in range (0, nIC):
        for l in range (0, nV0v):
            if (0.2<RD_IC_V0v[k,l]<10 and R_IC[k].x < R_V0v[l].x):     #the second condition is because the connection is descending
                RGap_IC_V0v[k,l] = IC_V0v_gap_weight*gauss(1, sigma)
            else:
                RGap_IC_V0v[k,l] = 0.0
                               
    for k in range (0, nIC):
        for l in range (0, nV2a):
            if (0.2<RD_IC_V2a[k,l]<10 and R_IC[k].x < R_V2a[l].x):     #the second condition is because the connection is descending
                RGap_IC_V2a[k,l] = IC_V2a_gap_weight*gauss(1, sigma)
            else:
                RGap_IC_V2a[k,l] = 0.0
                
    for k in range (0, nMN):
        for l in range (0, nMN):
            if (0.2<RD_MN_MN[k,l]< 6.5 ):    
                RGap_MN_MN[k,l] = MN_MN_gap_weight*gauss(1, sigma)
            else:
                RGap_MN_MN[k,l] = 0.0
                
    for k in range (0, nV0d):
        for l in range (0, nV0d):
            if (0.2<RD_V0d_V0d[k,l]< 3.5 ):    
                RGap_V0d_V0d[k,l] = V0d_V0d_gap_weight*gauss(1, sigma)
            else:
                RGap_V0d_V0d[k,l] = 0.0
                
    for k in range (0, nV0v):
        for l in range (0, nV0v):
            if (0.2<RD_V0v_V0v[k,l]< 3.5 ):    
                RGap_V0v_V0v[k,l] = V0v_V0v_gap_weight*gauss(1, sigma)
            else:
                RGap_V0v_V0v[k,l] = 0.0
                
    for k in range (0, nMN):
        for l in range (0, nV0d):
            if (R_V0d[l].x-1.5<R_MN[k].x< R_V0d[l].x + 1.5 ):    
                RGap_MN_V0d[k,l] = MN_V0d_gap_weight*gauss(1, sigma)
            else:
                RGap_MN_V0d[k,l] = 0.0
                
    for k in range (0, nMN):
        for l in range (0, nV0v):
            if (R_V0v[l].x-1.5 <R_MN[k].x< R_V0v[l].x + 1.5 ):    
                RGap_MN_V0v[k,l] = MN_V0v_gap_weight*gauss(1, sigma)
            else:
                RGap_MN_V0v[k,l] = 0.0
                
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
                
    for k in range (0, nV0d):
        for l in range (0, nMN):
            if (0.2<RD_V0d_MN[k,l]<8.0):  
                RW_V0d_MN[k,l] = V0d_MN_syn_weight*gauss(1, sigma)
            else:
                RW_V0d_MN[k,l] = 0.0
    
    for k in range (0, nV0d):
        for l in range (0, nIC):
            if (0.2<LD_V0d_IC[k,l]<20.0):  
                RW_V0d_IC[k,l] = V0d_IC_syn_weight*gauss(1, sigma)
            else:
                RW_V0d_IC[k,l] = 0.0     
    
    for k in range (0, nV0d):
        for l in range (0, nV2a):
            if (0.2<LD_V0d_V2a[k,l]<8.0):  
                RW_V0d_V2a[k,l] = V0d_V2a_syn_weight*gauss(1, sigma)
            else:
                RW_V0d_V2a[k,l] = 0.0                  
                
    for k in range (0, nV0v):
        for l in range (0, nIC):
            if (6.0<RD_V0v_IC[k,l]<20.0):  
                RW_V0v_IC[k,l] = V0v_IC_syn_weight*gauss(1, sigma)
            else:
                RW_V0v_IC[k,l] = 0.0       
    
    for k in range (0, nV2a):
        for l in range (0, nV0v):
            if (0.2<RD_V2a_V0v[k,l]<10 and R_V2a[k].x < R_V0v[l].x) or (0.2<RD_V2a_V0v[k,l]<4 and R_V2a[k].x > R_V0v[l].x):    #descending and short ascending branch
                RW_V2a_V0v[k,l] =  V2a_V0v_syn_weight*gauss(1, sigma)
            else:
                RW_V2a_V0v[k,l] = 0.0   
            
    for k in range (0, nMN):
        for l in range (0, nMuscle):
            if (R_Muscle[l].x-1<R_MN[k].x< R_Muscle[l].x+1):         #segmental connection
                RW_MN_Muscle[k,l] = MN_Muscle_syn_weight*gauss(1, sigma)
            else:
                RW_MN_Muscle[k,l] = 0.0
     
    ## initialize membrane potential values           
    for k in range (0, nIC):
        resLIC[k,:] = L_IC[k].getNextVal(-65,0,0)
        VLIC[k,0] = resLIC[k,0]
        
        resRIC[k,:] = R_IC[k].getNextVal(-65,0,0)
        VRIC[k,0] = resRIC[k,0]
    
    for k in range (0, nMN):
        resLMN[k,:] = L_MN[k].getNextVal(-65,0,0)
        VLMN[k,0] = resLMN[k,0]
        
        resRMN[k,:] = R_MN[k].getNextVal(-65,0,0)
        VRMN[k,0] = resRMN[k,0]
        
    for k in range (0, nV0d):
        resLV0d[k,:] = L_V0d[k].getNextVal(-65,0,0)
        VLV0d[k,0] = resLV0d[k,0]
        
        resRV0d[k,:] = R_V0d[k].getNextVal(-65,0,0)
        VRV0d[k,0] = resRV0d[k,0]
        
    for k in range (0, nV0v):
        resLV0v[k,:] = L_V0v[k].getNextVal(-65,0,0)
        VLV0v[k,0] = resLV0v[k,0]
        
        resRV0v[k,:] = R_V0v[k].getNextVal(-65,0,0)
        VRV0v[k,0] = resRV0v[k,0]

    for k in range (0, nV2a):
        resLV2a[k,:] = L_V2a[k].getNextVal(-64,-16,-64)
        VLV2a[k,0] = resLV2a[k,0]
        
        resRV2a[k,:] = R_V2a[k].getNextVal(-64,-16,-64)
        VRV2a[k,0] = resRV2a[k,0]
        
    for k in range (0, nMuscle):
        resLMuscle[k,:] = L_Muscle[k].getNextVal(0,0)
        VLMuscle[k,0] = resLMuscle[k,0]
        
        resRMuscle[k,:] = R_Muscle[k].getNextVal(0,0)
        VRMuscle[k,0] = resRMuscle[k,0]
        
    ## This loop is the main loop where we solve the ordinary differential equations at every time point     
    
    for t in range (0, nmax):
        Time[t]=dt*t
        
        # Generate plots to visualize the progress of the simulations 
        if not(Time[t] % 10000) and (Time[t]>20):
            
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
        stim[t]=stim0
        
        right_IC=False
        left_IC=False
        
        ## Calculate synaptic currents
        for k in range (0, nV0d):
            for l in range (0, nMN):
                LSyn_V0d_MN[nMN*k+l,:] = L_glysyn_V0d_MN[nMN*k+l].getNextVal(VLV0d[k,t-int(LD_V0d_MN[k,l]/(dt*cv))], VRMN[l,t-1], LSyn_V0d_MN[nMN*k+l,1], LSyn_V0d_MN[nMN*k+l,2]) #Contralateral
                RSyn_V0d_MN[nMN*k+l,:] = R_glysyn_V0d_MN[nMN*k+l].getNextVal(VRV0d[k,t-int(RD_V0d_MN[k,l]/(dt*cv))], VLMN[l,t-1], RSyn_V0d_MN[nMN*k+l,1], RSyn_V0d_MN[nMN*k+l,2]) #Contralateral
                
        for k in range (0, nV0d):
            for l in range (0, nIC):
                LSyn_V0d_IC[nIC*k+l,:] = L_glysyn_V0d_IC[nIC*k+l].getNextVal(VLV0d[k,t-int(LD_V0d_IC[k,l]/(dt*cv))], VRIC[l,t-1], LSyn_V0d_IC[nIC*k+l,1], LSyn_V0d_IC[nIC*k+l,2]) #Contralateral
                RSyn_V0d_IC[nIC*k+l,:] = R_glysyn_V0d_IC[nIC*k+l].getNextVal(VRV0d[k,t-int(RD_V0d_IC[k,l]/(dt*cv))], VLIC[l,t-1], RSyn_V0d_IC[nIC*k+l,1], RSyn_V0d_IC[nIC*k+l,2]) #Contralateral
                
        for k in range (0, nV0v):
            for l in range (0, nIC):
                LSyn_V0v_IC[nIC*k+l,:] = L_glusyn_V0v_IC[nIC*k+l].getNextVal(VLV0v[k,t-int(LD_V0v_IC[k,l]/(dt*cv))], VRIC[l,t-1], LSyn_V0v_IC[nIC*k+l,1], LSyn_V0v_IC[nIC*k+l,2])  #Contralateral
                RSyn_V0v_IC[nIC*k+l,:] = R_glusyn_V0v_IC[nIC*k+l].getNextVal(VRV0v[k,t-int(RD_V0v_IC[k,l]/(dt*cv))], VLIC[l,t-1], RSyn_V0v_IC[nIC*k+l,1], RSyn_V0v_IC[nIC*k+l,2])  #Contralateral
                               
        for k in range (0, nV2a):
            for l in range (0, nV0v):
                LSyn_V2a_V0v[nV0v*k+l,:] = L_glusyn_V2a_V0v[nV0v*k+l].getNextVal(VLV2a[k,t-int(LD_V2a_V0v[k,l]/(dt*cv))], VLV0v[l,t-1], LSyn_V2a_V0v[nV0v*k+l,1], LSyn_V2a_V0v[nV0v*k+l,2])
                RSyn_V2a_V0v[nV0v*k+l,:] = R_glusyn_V2a_V0v[nV0v*k+l].getNextVal(VRV2a[k,t-int(RD_V2a_V0v[k,l]/(dt*cv))], VRV0v[l,t-1], RSyn_V2a_V0v[nV0v*k+l,1], RSyn_V2a_V0v[nV0v*k+l,2])
    
        for k in range (0, nMN):
            for l in range (0, nMuscle):
                LSyn_MN_Muscle[nMuscle*k+l,:] = L_achsyn_MN_Muscle[nMuscle*k+l].getNextVal(VLMN[k,t-10], VLMuscle[l,t-1], LSyn_MN_Muscle[nMuscle*k+l,1], LSyn_MN_Muscle[nMuscle*k+l,2])
                RSyn_MN_Muscle[nMuscle*k+l,:] = R_achsyn_MN_Muscle[nMuscle*k+l].getNextVal(VRMN[k,t-10], VRMuscle[l,t-1], RSyn_MN_Muscle[nMuscle*k+l,1], RSyn_MN_Muscle[nMuscle*k+l,2])
                        
        for k in range (0, nIC):
            for l in range (0, nMN):   
                RSGap_IC_MN[k,l] = RGap_IC_MN[k,l]*(VRIC[k,t-int(RD_IC_MN[k,l]/(dt*cv))-1]-VRMN[l,t-1])
                LSGap_IC_MN[k,l] = LGap_IC_MN[k,l]*(VLIC[k,t-int(LD_IC_MN[k,l]/(dt*cv))-1]-VLMN[l,t-1])
                
        for k in range (0, nIC):
            for l in range (0, nIC):   
                RSGap_IC_IC[k,l] = RGap_IC_IC[k,l]*(VRIC[k,t-1]-VRIC[l,t-1])
                LSGap_IC_IC[k,l] = LGap_IC_IC[k,l]*(VLIC[k,t-1]-VLIC[l,t-1])
                
        for k in range (0, nIC):
            for l in range (0, nV0d):   
                RSGap_IC_V0d[k,l] = RGap_IC_V0d[k,l]*(VRIC[k,t-int(RD_IC_V0d[k,l]/(dt*cv))-1]-VRV0d[l,t-1])
                LSGap_IC_V0d[k,l] = LGap_IC_V0d[k,l]*(VLIC[k,t-int(LD_IC_V0d[k,l]/(dt*cv))-1]-VLV0d[l,t-1])
                
        for k in range (0, nIC):
            for l in range (0, nV0v):   
                RSGap_IC_V0v[k,l] = RGap_IC_V0v[k,l]*(VRIC[k,t-int(RD_IC_V0v[k,l]/(dt*cv))-1]-VRV0v[l,t-1])
                LSGap_IC_V0v[k,l] = LGap_IC_V0v[k,l]*(VLIC[k,t-int(LD_IC_V0v[k,l]/(dt*cv))-1]-VLV0v[l,t-1])  
                                                     
        for k in range (0, nIC):
            for l in range (0, nV2a):   
                RSGap_IC_V2a[k,l] = RGap_IC_V2a[k,l]*(VRIC[k,t-int(RD_IC_V2a[k,l]/(dt*cv))]-VRV2a[l,t-1])
                LSGap_IC_V2a[k,l] = LGap_IC_V2a[k,l]*(VLIC[k,t-int(LD_IC_V2a[k,l]/(dt*cv))]-VLV2a[l,t-1])

        for k in range (0, nMN):
            for l in range (0, nIC):   
                RSGap_MN_IC[k,l] = RGap_IC_MN[l,k]*(VRMN[k,t-int(RD_IC_MN[l,k]/(dt*cv))-1]-VRIC[l,t-1])
                LSGap_MN_IC[k,l] = LGap_IC_MN[l,k]*(VLMN[k,t-int(LD_IC_MN[l,k]/(dt*cv))-1]-VLIC[l,t-1])
                
        for k in range (0, nMN):
            for l in range (0, nMN):   
                RSGap_MN_MN[k,l] = RGap_MN_MN[k,l]*(VRMN[k,t-int(RD_MN_MN[k,l]/(dt*cv))-1]-VRMN[l,t-1])
                LSGap_MN_MN[k,l] = LGap_MN_MN[k,l]*(VLMN[k,t-int(LD_MN_MN[k,l]/(dt*cv))-1]-VLMN[l,t-1])

        for k in range (0, nMN):
            for l in range (0, nV0d):   
                RSGap_MN_V0d[k,l] = RGap_MN_V0d[k,l]*(VRMN[k,t-int(RD_MN_V0d[k,l]/(dt*cv))-1]-VRV0d[l,t-1])
                LSGap_MN_V0d[k,l] = LGap_MN_V0d[k,l]*(VLMN[k,t-int(LD_MN_V0d[k,l]/(dt*cv))-1]-VLV0d[l,t-1])
                
        for k in range (0, nMN):
            for l in range (0, nV0v):   
                RSGap_MN_V0v[k,l] = RGap_MN_V0v[k,l]*(VRMN[k,t-int(RD_MN_V0v[k,l]/(dt*cv))-1]-VRV0v[l,t-1])
                LSGap_MN_V0v[k,l] = LGap_MN_V0v[k,l]*(VLMN[k,t-int(LD_MN_V0v[k,l]/(dt*cv))-1]-VLV0v[l,t-1])
                
        for k in range (0, nMN):
            for l in range (0, nV2a):   
                RSGap_MN_V2a[k,l] = RGap_V2a_MN[l,k]*(VRMN[k,t-int(RD_V2a_MN[l,k]/(dt*cv))]-VRV2a[l,t-1])
                LSGap_MN_V2a[k,l] = LGap_V2a_MN[l,k]*(VLMN[k,t-int(LD_V2a_MN[l,k]/(dt*cv))]-VLV2a[l,t-1])
                        
        for k in range (0, nV0d):
            for l in range (0, nIC):   
                RSGap_V0d_IC[k,l] = RGap_IC_V0d[l,k]*(VRV0d[k,t-int(RD_IC_V0d[l,k]/(dt*cv))-1]-VRIC[l,t-1])
                LSGap_V0d_IC[k,l] = LGap_IC_V0d[l,k]*(VLV0d[k,t-int(LD_IC_V0d[l,k]/(dt*cv))-1]-VLIC[l,t-1])
 
        for k in range (0, nV0d):
            for l in range (0, nMN):   
                RSGap_V0d_MN[k,l] = RGap_MN_V0d[l,k]*(VRV0d[k,t-int(RD_MN_V0d[l,k]/(dt*cv))-1]-VRMN[l,t-1])
                LSGap_V0d_MN[k,l] = LGap_MN_V0d[l,k]*(VLV0d[k,t-int(LD_MN_V0d[l,k]/(dt*cv))-1]-VLMN[l,t-1])
        
        for k in range (0, nV0d):
            for l in range (0, nV0d):   
                RSGap_V0d_V0d[k,l] = RGap_V0d_V0d[k,l]*(VRV0d[k,t-int(RD_V0d_V0d[k,l]/(dt*cv))-1]-VRV0d[l,t-1])
                LSGap_V0d_V0d[k,l] = LGap_V0d_V0d[k,l]*(VLV0d[k,t-int(LD_V0d_V0d[k,l]/(dt*cv))-1]-VLV0d[l,t-1])
                        
        for k in range (0, nV0v):
            for l in range (0, nIC):   
                RSGap_V0v_IC[k,l] = RGap_IC_V0v[l,k]*(VRV0v[k,t-int(RD_IC_V0v[l,k]/(dt*cv))-1]-VRIC[l,t-1])
                LSGap_V0v_IC[k,l] = LGap_IC_V0v[l,k]*(VLV0v[k,t-int(LD_IC_V0v[l,k]/(dt*cv))-1]-VLIC[l,t-1])
 
        for k in range (0, nV0v):
            for l in range (0, nMN):   
                RSGap_V0v_MN[k,l] = RGap_MN_V0v[l,k]*(VRV0v[k,t-int(RD_MN_V0v[l,k]/(dt*cv))-1]-VRMN[l,t-1])
                LSGap_V0v_MN[k,l] = LGap_MN_V0v[l,k]*(VLV0v[k,t-int(LD_MN_V0v[l,k]/(dt*cv))-1]-VLMN[l,t-1])
                
        for k in range (0, nV0v):
            for l in range (0, nV0v):   
                RSGap_V0v_V0v[k,l] = RGap_V0v_V0v[k,l]*(VRV0v[k,t-int(RD_V0v_V0v[k,l]/(dt*cv))-1]-VRV0v[l,t-1])
                LSGap_V0v_V0v[k,l] = LGap_V0v_V0v[k,l]*(VLV0v[k,t-int(LD_V0v_V0v[k,l]/(dt*cv))-1]-VLV0v[l,t-1])
                                      
        for k in range (0, nV2a):
            for l in range (0, nIC):   
                RSGap_V2a_IC[k,l] = RGap_IC_V2a[l,k]*(VRV2a[k,t-int(RD_IC_V2a[l,k]/(dt*cv))]-VRIC[l,t-1])
                LSGap_V2a_IC[k,l] = LGap_IC_V2a[l,k]*(VLV2a[k,t-int(LD_IC_V2a[l,k]/(dt*cv))]-VLIC[l,t-1])
                
        for k in range (0, nV2a):
            for l in range (0, nV2a):   
                RSGap_V2a_V2a[k,l] = RGap_V2a_V2a[k,l]*(VRV2a[k,t-int(RD_V2a_V2a[k,l]/(dt*cv))]-VRV2a[l,t-1])
                LSGap_V2a_V2a[k,l] = LGap_V2a_V2a[k,l]*(VLV2a[k,t-int(LD_V2a_V2a[k,l]/(dt*cv))]-VLV2a[l,t-1])
                
        for k in range (0, nV2a):
            for l in range (0, nMN):   
                RSGap_V2a_MN[k,l] = RGap_V2a_MN[k,l]*(VRV2a[k,t-int(RD_V2a_MN[k,l]/(dt*cv))]-VRMN[l,t-1])
                LSGap_V2a_MN[k,l] = LGap_V2a_MN[k,l]*(VLV2a[k,t-int(LD_V2a_MN[k,l]/(dt*cv))]-VLMN[l,t-1])
                
        ## Determine membrane potentials from synaptic and external currents
        for k in range (0, nIC):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(RSyn_V0v_IC[nIC*l+k,0]*LW_V0v_IC[l,k] for l in range (0, nV0v)) + sum(RSyn_V0d_IC[nIC*l+k,0]*LW_V0d_IC[l,k] for l in range (0, nV0d))
                IsynR = sum(LSyn_V0v_IC[nIC*l+k,0]*RW_V0v_IC[l,k] for l in range (0, nV0v)) + sum(LSyn_V0d_IC[nIC*l+k,0]*RW_V0d_IC[l,k] for l in range (0, nV0d))
            
            if t < 15000: #to ensure that double coils are not due to coincident coiling 
                right_side_delay=0
            else:
                right_side_delay=1

            resLIC[k,:] = L_IC[k].getNextVal(resLIC[k,0],resLIC[k,1], stim[t] + 0*stim[t]*int(left_IC) + sum(LSGap_IC_IC[:,k]) - sum(LSGap_IC_IC[k,:]) + sum(LSGap_MN_IC[:,k]) - sum(LSGap_IC_MN[k,:]) + sum(LSGap_V0d_IC[:,k]) - sum(LSGap_IC_V0d[k,:]) + sum(LSGap_V0v_IC[:,k]) - sum(LSGap_IC_V0v[k,:]) + sum(LSGap_V2a_IC[:,k]) - sum(LSGap_IC_V2a[k,:]) + IsynL)
            VLIC[k,t] = resLIC[k,0]
        
            resRIC[k,:] = R_IC[k].getNextVal(resRIC[k,0],resRIC[k,1], stim[t]*right_side_delay+ 0*stim[t]*int(right_IC) + sum(RSGap_IC_IC[:,k])  - sum(RSGap_IC_IC[k,:])  + sum(RSGap_MN_IC[:,k]) - sum(RSGap_IC_MN[k,:]) + sum(RSGap_V0d_IC[:,k]) - sum(RSGap_IC_V0d[k,:])  + sum(RSGap_V0v_IC[:,k]) - sum(RSGap_IC_V0v[k,:]) + sum(RSGap_V2a_IC[:,k]) - sum(RSGap_IC_V2a[k,:]) + IsynR)
            VRIC[k,t] = resRIC[k,0]

        for k in range (0, nMN):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(RSyn_V0d_MN[nMN*l+k,0]*LW_V0d_MN[l,k] for l in range (0, nV0d)) 
                IsynR = sum(LSyn_V0d_MN[nMN*l+k,0]*RW_V0d_MN[l,k] for l in range (0, nV0d))
            #if k == 4: # this is to hyperpolarize a MN to observe periodic depolarizations and synaptic bursts
             #   IsynL = IsynL - 10
            resLMN[k,:] = L_MN[k].getNextVal(resLMN[k,0],resLMN[k,1], - sum(LSGap_MN_IC[k,:]) + sum(LSGap_IC_MN[:,k]) - sum(LSGap_MN_MN[k,:]) + sum(LSGap_MN_MN[:,k]) - sum(LSGap_MN_V0d[k,:]) + sum(LSGap_V0d_MN[:,k]) - sum(LSGap_MN_V0v[k,:]) + sum(LSGap_V0v_MN[:,k]) - sum(LSGap_MN_V2a[k,:]) + sum(LSGap_V2a_MN[:,k]) + IsynL)  
            VLMN[k,t] = resLMN[k,0]
            
            resRMN[k,:] = R_MN[k].getNextVal(resRMN[k,0],resRMN[k,1],  - sum(RSGap_MN_IC[k,:]) + sum(RSGap_IC_MN[:,k]) - sum(RSGap_MN_MN[k,:]) + sum(RSGap_MN_MN[:,k]) - sum(RSGap_MN_V0d[k,:]) + sum(RSGap_V0d_MN[:,k]) - sum(RSGap_MN_V0v[k,:]) + sum(RSGap_V0v_MN[:,k]) - sum(LSGap_MN_V2a[k,:]) + sum(LSGap_V2a_MN[:,k]) + IsynR) 
            VRMN[k,t] = resRMN[k,0]
         
        for k in range (0, nV0d):
            resLV0d[k,:] = L_V0d[k].getNextVal(resLV0d[k,0],resLV0d[k,1],  - sum(LSGap_V0d_IC[k,:])+ sum(LSGap_IC_V0d[:,k]) - sum(LSGap_V0d_V0d[k,:]) + sum(LSGap_V0d_V0d[:,k]) - sum(LSGap_V0d_MN[k,:]) + sum(LSGap_MN_V0d[:,k]))
            VLV0d[k,t] = resLV0d[k,0]
            resRV0d[k,:] = R_V0d[k].getNextVal(resRV0d[k,0],resRV0d[k,1], - sum(RSGap_V0d_IC[k,:]) + sum(RSGap_IC_V0d[:,k]) - sum(RSGap_V0d_V0d[k,:]) + sum(RSGap_V0d_V0d[:,k]) - sum(RSGap_V0d_MN[k,:]) + sum(RSGap_MN_V0d[:,k]))
            VRV0d[k,t] = resRV0d[k,0]

        for k in range (0, nV0v):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(LSyn_V2a_V0v[nV0v*l+k,0]*LW_V2a_V0v[l,k] for l in range (0, nV2a))
                IsynR = sum(RSyn_V2a_V0v[nV0v*l+k,0]*RW_V2a_V0v[l,k] for l in range (0, nV2a))             
            resLV0v[k,:] = L_V0v[k].getNextVal(resLV0v[k,0],resLV0v[k,1], - sum(LSGap_V0v_IC[k,:]) + sum(LSGap_IC_V0v[:,k]) - sum(LSGap_V0v_V0v[k,:]) + sum(LSGap_V0v_V0v[:,k]) -sum(LSGap_V0v_MN[k,:])  + sum(LSGap_MN_V0v[:,k]) + IsynL)
            VLV0v[k,t] = resLV0v[k,0]
            resRV0v[k,:] = R_V0v[k].getNextVal(resRV0v[k,0],resRV0v[k,1], - sum(RSGap_V0v_IC[k,:]) + sum(RSGap_IC_V0v[:,k]) - sum(RSGap_V0v_V0v[k,:]) + sum(RSGap_V0v_V0v[:,k]) -sum(RSGap_V0v_MN[k,:])  + sum(RSGap_MN_V0v[:,k]) + IsynR)
            VRV0v[k,t] = resRV0v[k,0]

        for k in range (0, nV2a):
            if t < 500: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(RSyn_V0d_V2a[nV2a*l+k,0]*LW_V0d_V2a[l,k] for l in range (0, nV0d)) 
                IsynR = sum(LSyn_V0d_V2a[nV2a*l+k,0]*RW_V0d_V2a[l,k] for l in range (0, nV0d))
            resLV2a[k,:] = L_V2a[k].getNextVal(resLV2a[k,0],resLV2a[k,1], - sum(LSGap_V2a_IC[k,:]) + sum(LSGap_IC_V2a[:,k]) - sum(LSGap_V2a_V2a[k,:]) + sum(LSGap_V2a_V2a[:,k]) - sum(LSGap_V2a_MN[k,:])+ sum(LSGap_MN_V2a[:,k]))         
            VLV2a[k,t] = resLV2a[k,0]
            resRV2a[k,:] = R_V2a[k].getNextVal(resRV2a[k,0],resRV2a[k,1],  - sum(RSGap_V2a_IC[k,:]) + sum(RSGap_IC_V2a[:,k]) - sum(RSGap_V2a_V2a[k,:]) + sum(RSGap_V2a_V2a[:,k]) - sum(RSGap_V2a_MN[k,:])+ sum(RSGap_MN_V2a[:,k]))    
            VRV2a[k,t] = resRV2a[k,0]
            
        for k in range (0, nMuscle):
            resLMuscle[k,:] = L_Muscle[k].getNextVal(resLMuscle[k,0], sum(LSyn_MN_Muscle[nMuscle*l+k,0]*LW_MN_Muscle[l,k] for l in range (0, nMN)))
            VLMuscle[k,t] = resLMuscle[k,0]
            
            resRMuscle[k,:] = R_Muscle[k].getNextVal(resRMuscle[k,0], sum(RSyn_MN_Muscle[nMuscle*l+k,0]*RW_MN_Muscle[l,k] for l in range (0, nMN)))
            VRMuscle[k,t] = resRMuscle[k,0]
    
        
      ## Removing the first 200 ms to let the initial conditions dissipate
    index_offset = int(200/dt)
    
    VLICnew = VLIC[:,index_offset:]
    VRICnew = VRIC[:,index_offset:]
    
    VLMNnew = VLMN[:,index_offset:]
    VRMNnew = VRMN[:,index_offset:]
    
    VLV0dnew = VLV0d[:,index_offset:]
    VRV0dnew = VRV0d[:,index_offset:]
    
    VLV0vnew = VLV0v[:,index_offset:]
    VRV0vnew = VRV0v[:,index_offset:]
    
    VLV2anew = VLV2a[:,index_offset:]
    VRV2anew = VRV2a[:,index_offset:]
    
    VLMusclenew = VLMuscle[:,index_offset:]
    VRMusclenew = VRMuscle[:,index_offset:]
    
    Timenew = Time[index_offset:]-Time[index_offset:][0]
    
    
    return (VLICnew, VRICnew), (VLMNnew, VRMNnew), (VLV0dnew, VRV0dnew), (VLV0vnew, VRV0vnew), (VLV2anew, VRV2anew), (VLMusclenew, VRMusclenew), Timenew