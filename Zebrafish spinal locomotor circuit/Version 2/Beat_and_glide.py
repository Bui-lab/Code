
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""

import matplotlib
from matplotlib.pyplot import plot
import numpy as np
from numpy import mat, zeros
from os import replace
from pylab import plt
from random import seed, gauss

#The definition of single cell models are found based on Izhikevich models
from Izhikevich_class import Izhikevich_9P, TwoExp_syn, Distance, Leaky_Integrator
from Util import saveToCSV, saveAnimation, saveToJSON, plotProgress

class Beat_and_glide_base:
    #%% Class Variables
    #region Class Variables

    #region General Parameters
    model_run = False #to check whether the model has run successfuly - used for plotting and saving functions
    __tmax = 100000 # in time points (value in ms / dt)
    __tskip = 10000 # in time points (value in ms / dt)
    __dt = 0.1
    #the currents are shut off to wait for the initial conditions to subside
    __tshutoff = 500 # in time points (value in ms / dt)
    #tasyncdelay = 15000
    #endregion 

    #region Other constants
    stim0 = 2.89
    sigma = 0
    sigma_LR = 0.1
    E_glu = 0
    E_gly = -70
    cv = 0.80
    R_str = 1.0
    #endregion

    #region Number of cells
    nMN = 15
    ndI6 = 15
    nV0v = 15
    nV2a = 15
    nV1 = 15
    nMuscle = 15
    #endregion

    #region Weight Parameters 
    MN_MN_gap_weight = 0.005 
    dI6_dI6_gap_weight = 0.04
    V0v_V0v_gap_weight = 0.05
    V2a_V2a_gap_weight = 0.005
    V2a_MN_gap_weight = 0.005
    MN_dI6_gap_weight = 0.0001
    MN_V0v_gap_weight = 0.005 

    V2a_V2a_syn_weight = 0.3 
    V2a_MN_syn_weight = 0.5
    V2a_dI6_syn_weight = 0.5       
    V2a_V1_syn_weight = 0.5           
    V2a_V0v_syn_weight = 0.3
    V1_MN_syn_weight = 1.0      
    V1_V2a_syn_weight = 0.5    
    V1_V0v_syn_weight = 0.1
    V1_dI6_syn_weight = 0.2 
    dI6_MN_syn_weight = 1.5  
    dI6_V2a_syn_weight = 1.5 
    dI6_dI6_syn_weight = 0.25
    V0v_V2a_syn_weight = 0.4   
    MN_Muscle_syn_weight = 0.1 

    #endregion
    
    #region Neurons
    L_MN = []
    R_MN = []
    L_dI6 = []
    R_dI6 = []
    L_V0v = []
    R_V0v = []
    L_V2a = []
    R_V2a = []
    L_V1 = []
    R_V1 = []
    L_Muscle = []
    R_Muscle = []
    #endregion

    #region Synapses
    L_achsyn_MN_Muscle = None
    R_achsyn_MN_Muscle = None

    L_glysyn_dI6_MN = None
    R_glysyn_dI6_MN = None
    L_glysyn_dI6_V2a = None
    R_glysyn_dI6_V2a = None
    L_glysyn_dI6_dI6 = None
    R_glysyn_dI6_dI6 = None

    L_glysyn_V1_MN = None
    R_glysyn_V1_MN = None
    L_glysyn_V1_V2a = None
    R_glysyn_V1_V2a = None
    L_glysyn_V1_V0v = None
    R_glysyn_V1_V0v = None
    L_glysyn_V1_dI6 = None
    R_glysyn_V1_dI6 = None

    L_glusyn_V0v_V2a = None
    R_glusyn_V0v_V2a = None

    L_glusyn_V2a_V2a = None
    R_glusyn_V2a_V2a = None
    L_glusyn_V2a_MN = None
    R_glusyn_V2a_MN = None
    L_glusyn_V2a_dI6 = None
    R_glusyn_V2a_dI6 = None
    L_glusyn_V2a_V1 = None
    R_glusyn_V2a_V1 = None
    L_glusyn_V2a_V0v = None
    R_glusyn_V2a_V0v = None
    #endregion

    #region Lists of Membrane Potentials
    VLMN = []
    VRMN = []
    VLdI6 = []
    VRdI6 = []
    VLV0v = []
    VRV0v = []
    VLV2a = []
    VRV2a = []
    VLV1 = []
    VRV1 = []
    VLMuscle =[]
    VRMuscle = []
    skippedVLists = dict()
    #endregion
     
    #region Lists of Synaptic and Gap Junction Currents
    LSyn_MN_Muscle = []
    RSyn_MN_Muscle = []
    
    LSyn_dI6_MN = []
    RSyn_dI6_MN = []
    LSyn_dI6_V2a = []
    RSyn_dI6_V2a = []
    LSyn_dI6_dI6 = []
    RSyn_dI6_dI6 = []
    
    LSyn_V1_MN = []
    RSyn_V1_MN = []
    LSyn_V1_V2a = []
    RSyn_V1_V2a = []
    LSyn_V1_V0v = []
    RSyn_V1_V0v = []
    LSyn_V1_dI6 = []
    RSyn_V1_dI6 = []

    LSyn_V0v_V2a = []
    RSyn_V0v_V2a = []
    LSyn_V2a_V2a = []
    RSyn_V2a_V2a = []
    LSyn_V2a_MN = []
    RSyn_V2a_MN = []
    LSyn_V2a_dI6 = []
    RSyn_V2a_dI6 = []
    LSyn_V2a_V1 = []
    RSyn_V2a_V1 = []
    LSyn_V2a_V0v = []
    RSyn_V2a_V0v = []

    LSGap_MN_MN = []
    LSGap_MN_dI6 = []
    LSGap_MN_V0v = []
    LSGap_MN_V2a = []
    LSGap_dI6_MN = []
    LSGap_dI6_dI6 = []
    LSGap_V0v_MN = []
    LSGap_V0v_V0v = []
    LSGap_V2a_V2a = []
    LSGap_V2a_MN = []

    RSGap_MN_MN = []
    RSGap_MN_dI6 = []
    RSGap_MN_V0v = []
    RSGap_MN_V2a = []
    RSGap_dI6_MN = []
    RSGap_dI6_dI6 = []
    RSGap_V0v_MN = []
    RSGap_V0v_V0v = []
    RSGap_V2a_V2a = []
    RSGap_V2a_MN = []
    #endregion
    
    #region Lists of Synaptic and Gap Junction Weights
    LW_dI6_MN = []
    RW_dI6_MN = []
    LW_dI6_V2a = []
    RW_dI6_V2a = []
    LW_dI6_dI6 = []
    RW_dI6_dI6 = []
    LW_V1_MN = []
    RW_V1_MN = []
    LW_V1_V2a = []
    RW_V1_V2a = []
    LW_V1_V0v = []
    RW_V1_V0v = []
    LW_V1_dI6 = []
    RW_V1_dI6 = []
    LW_V0v_V2a = []
    RW_V0v_V2a = []
    LW_V2a_V2a = []
    RW_V2a_V2a = []
    LW_V2a_MN = []
    RW_V2a_MN = []
    LW_V2a_dI6 = []
    RW_V2a_dI6 = []
    LW_V2a_V1 = []
    RW_V2a_V1 = []
    LW_V2a_V0v = []
    RW_V2a_V0v = []
    LW_MN_Muscle = []
    RW_MN_Muscle = []
        

    LGap_MN_MN = []
    LGap_MN_dI6 = []
    LGap_MN_V0v = []
    LGap_dI6_dI6 = []
    LGap_V0v_V0v = []
    LGap_V2a_V2a = []
    LGap_V2a_MN = []

    RGap_MN_MN = []
    RGap_MN_dI6 = []
    RGap_MN_V0v = []
    RGap_dI6_dI6 = []
    RGap_V0v_V0v = []
    RGap_V2a_V2a = []
    RGap_V2a_MN = []
    #endregion
    
    #region Lists of residuals for solving membrane potentials
    resLMN = []
    resRMN = []
    resLdI6 = []
    resRdI6 = []
    resLV0v = []
    resRV0v = []
    resLV2a = []
    resRV2a = []
    resLV1 = []
    resRV1 = []

    resLMuscle = []
    resRMuscle = []
    #endregion
    
    #region Distance Matrices
    LD_MN_MN = []
    LD_MN_dI6 = []
    LD_MN_V0v = []
    LD_dI6_dI6 = []
    LD_V0v_V0v = []
    LD_V0v_V2a = []
    
    LD_V2a_V2a = []     
    LD_V2a_MN = []           
    LD_V2a_dI6 = []   
    LD_V2a_V1 = []   
    LD_V2a_V0v = []   
    
    LD_V1_MN = []
    LD_V1_V0v = []
    LD_V1_dI6 = []
    
    LD_dI6_MN = []     
    LD_dI6_V2a = []   
    
    RD_MN_MN = []
    RD_MN_dI6 = []
    RD_MN_V0v = []
    RD_dI6_dI6 = []
    RD_V0v_V0v = []
    RD_V0v_V2a = []
    
    RD_V2a_V2a = []     
    RD_V2a_MN = []           
    RD_V2a_dI6 = []   
    RD_V2a_V1 = []   
    RD_V2a_V0v = []
    
    RD_V1_MN = []
    RD_V1_V0v = []
    RD_V1_dI6 = []
    
    RD_dI6_MN = []     
    RD_dI6_V2a = []       
        
    #endregion
    
    #region Upper range parameters for neuron to neuron/muscle cell reach
    rangeMin = 0.2 # lower range value
    rangeMN_MN = 4.5
    rangedI6_dI6 = 3.5
    rangeV0v_V0v = 3.5
    rangeV2a_V2a_gap = 3.5
    rangeV2a_MN = 3.5
    rangeMN_dI6 = 1.5
    rangeMN_V0v = 1.5
    rangeV2a_V2a_syn = 10
    rangeV2a_MN_asc = 4
    rangeV2a_MN_desc = 10
    rangeV2a_dI6 = 10
    rangeV2a_V1 = 10
    rangeV2a_V0v_asc = 4
    rangeV2a_V0v_desc = 10
    rangeV1_MN = 4
    rangeV1_V2a = 4
    rangeV1_V0v = 4
    rangeV1_dI6 = 4
    rangedI6_MN_desc = 5
    rangedI6_MN_asc = 2
    rangedI6_V2a_desc = 5
    rangedI6_V2a_asc = 2
    rangedI6_dI6_desc = 5
    rangedI6_dI6_asc = 2
    rangeV0v_V2a_desc = 5
    rangeV0v_V2a_asc = 2
    rangeMN_Muscle = 1
    #endregion
    
    #region
    Time = []
    stim = []
    paramDict = dict()
    #endregion
    
    #endregion Class Variables

    #region Functions
    #%% Functions
   
    # stim0 is a constant for scaling the drive to V2a
    # sigma is a variance for gaussian randomization of the gap junction coupling
    # E_glu and E_gly are the reversal potential of glutamate and glycine respectively
    # c is the transmission speed
    # nMN, nV2a, ndI6, nV0v, nV1, and nMuscle is the number of MN, V2a, dI6, V0v, V1 and Muscle cells
    # R_str is an indication of whether glycinergic synapses are present or blocked by strychnine (str). Ranges from 0 to 1. 
    #    1: they are present; 0: they are all blocked
    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1,
                          E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                          R_str = 1.0):
        self.stim0 = stim0
        self.sigma = sigma
        self.sigma_LR = sigma_LR
        self.R_str = R_str

        self.E_glu = E_glu
        self.E_gly = E_gly
        self.cv = cv
        self.nMN = nMN
        self.ndI6 = ndI6
        self.nV0v = nV0v
        self.nV2a = nV2a
        self.nV1 = nV1
        self.nMuscle = nMuscle
        self.setTimeParameters() #to initialize with default values

    def setWeightParameters(self, MN_MN_gap_weight = None, dI6_dI6_gap_weight = None, V0v_V0v_gap_weight = None, V2a_V2a_gap_weight = None, V2a_MN_gap_weight = None,
                    MN_dI6_gap_weight = None, MN_V0v_gap_weight = None, V2a_V2a_syn_weight = None, V2a_MN_syn_weight = None, V2a_dI6_syn_weight = None, 
                    V2a_V1_syn_weight = None, V2a_V0v_syn_weight = None, V1_MN_syn_weight = None, V1_V2a_syn_weight = None, V1_V0v_syn_weight = None,
                    V1_dI6_syn_weight = None, dI6_MN_syn_weight = None, dI6_V2a_syn_weight = None, dI6_dI6_syn_weight = None, V0v_V2a_syn_weight = None,
                    MN_Muscle_syn_weight = None):
        if MN_MN_gap_weight is not None:
            self.MN_MN_gap_weight = MN_MN_gap_weight
        if dI6_dI6_gap_weight is not None:             
            self.dI6_dI6_gap_weight = dI6_dI6_gap_weight
        if V0v_V0v_gap_weight is not None:             
            self.V0v_V0v_gap_weight = V0v_V0v_gap_weight
        if V2a_V2a_gap_weight is not None:             
            self.V2a_V2a_gap_weight = V2a_V2a_gap_weight
        if V2a_MN_gap_weight is not None:             
            self.V2a_MN_gap_weight = V2a_MN_gap_weight
        if MN_dI6_gap_weight is not None:             
            self.MN_dI6_gap_weight = MN_dI6_gap_weight
        if MN_V0v_gap_weight is not None:             
            self.MN_V0v_gap_weight = MN_V0v_gap_weight

        if V2a_V2a_syn_weight is not None:             
            self.V2a_V2a_syn_weight = V2a_V2a_syn_weight
        if V2a_MN_syn_weight is not None:             
            self.V2a_MN_syn_weight = V2a_MN_syn_weight
        if V2a_dI6_syn_weight is not None:             
            self.V2a_dI6_syn_weight = V2a_dI6_syn_weight
        if V2a_V1_syn_weight is not None:             
            self.V2a_V1_syn_weight = V2a_V1_syn_weight
        if V2a_V0v_syn_weight is not None:             
            self.V2a_V0v_syn_weight = V2a_V0v_syn_weight
        if V1_MN_syn_weight is not None:             
            self.V1_MN_syn_weight = V1_MN_syn_weight
        if V1_V2a_syn_weight is not None:             
            self.V1_V2a_syn_weight = V1_V2a_syn_weight
        if V1_V0v_syn_weight is not None:             
            self.V1_V0v_syn_weight = V1_V0v_syn_weight
        if V1_dI6_syn_weight is not None:             
            self.V1_dI6_syn_weight = V1_dI6_syn_weight
        if dI6_MN_syn_weight is not None:             
            self.dI6_MN_syn_weight = dI6_MN_syn_weight
        if dI6_V2a_syn_weight is not None:             
            self.dI6_V2a_syn_weight = dI6_V2a_syn_weight
        if dI6_dI6_syn_weight is not None:             
            self.dI6_dI6_syn_weight = dI6_dI6_syn_weight
        if V0v_V2a_syn_weight is not None:             
            self.V0v_V2a_syn_weight = V0v_V2a_syn_weight
        if MN_Muscle_syn_weight is not None:             
            self.MN_Muscle_syn_weight = MN_Muscle_syn_weight

    def setRangeParameters(self, rangeMin = None, rangeMN_MN = None, rangedI6_dI6 = None, rangeV0v_V0v = None, rangeV2a_V2a_gap = None, rangeV2a_MN = None,
                            rangeMN_dI6 = None, rangeMN_V0v = None, rangeV2a_V2a_syn = None, rangeV2a_MN_asc = None, rangeV2a_MN_desc = None, rangeV2a_dI6 = None,
                            rangeV2a_V1 = None, rangeV2a_V0v_asc = None, rangeV2a_V0v_desc = None, rangeV1_MN = None, rangeV1_V2a = None, rangeV1_V0v = None,
                            rangeV1_dI6 = None, rangedI6_MN_desc = None, rangedI6_MN_asc = None, rangedI6_V2a_desc = None, rangedI6_V2a_asc = None,
                            rangedI6_dI6_desc = None, rangedI6_dI6_asc = None, rangeV0v_V2a_desc = None, rangeV0v_V2a_asc = None, rangeMN_Muscle = None):
        if rangeMin is not None:
            self.rangeMin = rangeMin
        if rangeMN_MN is not None:             
            self.rangeMN_MN = rangeMN_MN
        if rangedI6_dI6 is not None:             
            self.rangedI6_dI6 = rangedI6_dI6
        if rangeV0v_V0v is not None:             
            self.rangeV0v_V0v = rangeV0v_V0v
        if rangeV2a_V2a_gap is not None:             
            self.rangeV2a_V2a_gap = rangeV2a_V2a_gap
        if rangeV2a_MN is not None:             
            self.rangeV2a_MN = rangeV2a_MN
        if rangeMN_dI6 is not None:             
            self.rangeMN_dI6 = rangeMN_dI6
        if rangeMN_V0v is not None:             
            self.rangeMN_V0v = rangeMN_V0v
        if rangeV2a_V2a_syn is not None:             
            self.rangeV2a_V2a_syn = rangeV2a_V2a_syn
        if rangeV2a_MN_asc is not None:             
            self.rangeV2a_MN_asc = rangeV2a_MN_asc
        if rangeV2a_MN_desc is not None:             
            self.rangeV2a_MN_desc = rangeV2a_MN_desc
        if rangeV2a_dI6 is not None:             
            self.rangeV2a_dI6 = rangeV2a_dI6
        if rangeV2a_V1 is not None:             
            self.rangeV2a_V1 = rangeV2a_V1
        if rangeV2a_V0v_asc is not None:             
            self.rangeV2a_V0v_asc = rangeV2a_V0v_asc
        if rangeV2a_V0v_desc is not None:             
            self.rangeV2a_V0v_desc = rangeV2a_V0v_desc
        if rangeV1_MN is not None:             
            self.rangeV1_MN = rangeV1_MN
        if rangeV1_V2a is not None:             
            self.rangeV1_V2a = rangeV1_V2a
        if rangeV1_V0v is not None:             
            self.rangeV1_V0v = rangeV1_V0v
        if rangeV1_dI6 is not None:             
            self.rangeV1_dI6 = rangeV1_dI6
        if rangedI6_MN_desc is not None:             
            self.rangedI6_MN_desc = rangedI6_MN_desc
        if rangedI6_MN_asc is not None:             
            self.rangedI6_MN_asc = rangedI6_MN_asc
        if rangedI6_V2a_desc is not None:             
            self.rangedI6_V2a_desc = rangedI6_V2a_desc
        if rangedI6_V2a_asc is not None:             
            self.rangedI6_V2a_asc = rangedI6_V2a_asc
        if rangedI6_dI6_desc is not None:             
            self.rangedI6_dI6_desc = rangedI6_dI6_desc
        if rangedI6_dI6_asc is not None:             
            self.rangedI6_dI6_asc = rangedI6_dI6_asc
        if rangeV0v_V2a_desc is not None:             
            self.rangeV0v_V2a_desc = rangeV0v_V2a_desc
        if rangeV0v_V2a_asc is not None:             
            self.rangeV0v_V2a_asc = rangeV0v_V2a_asc
        if rangeMN_Muscle is not None:             
            self.rangeMN_Muscle = rangeMN_Muscle

    #Functions called from the main loop
    def initializeStructures(self, nmax):
        self.initNeurons()
        self.initSynapses()
        self.initStorageLists(nmax)
        self.initAndSetDistanceMatrices()

    def initNeurons(self):
        ## Declare Neuron Types

        self.L_MN = [ Izhikevich_9P(a=0.5,b=0.01,c=-55, d=100, vmax=10, vr=-65, vt=-58, k=0.5, Cm = 20, dt = self.__dt, x=5.0+1.6*i*gauss(1, self.sigma),y=-1) for i in range(self.nMN)]
        self.R_MN = [ Izhikevich_9P(a=0.5,b=0.01,c=-55, d=100, vmax=10, vr=-65, vt=-58, k=0.5, Cm = 20, dt = self.__dt, x=5.0+1.6*i*gauss(1, self.sigma),y=1) for i in range(self.nMN)]

        self.L_dI6 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=5.1+1.6*i*gauss(1, self.sigma),y=-1) for i in range(self.ndI6)]
        self.R_dI6 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=5.1+1.6*i*gauss(1, self.sigma),y=1) for i in range(self.ndI6)]

        self.L_V0v = [ Izhikevich_9P(a=0.01,b=0.002,c=-55, d=2, vmax=8, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=5.1+1.6*i*gauss(1, self.sigma),y=-1) for i in range(self.nV0v)]
        self.R_V0v = [ Izhikevich_9P(a=0.01,b=0.002,c=-55, d=2, vmax=8, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=5.1+1.6*i*gauss(1, self.sigma),y=1) for i in range(self.nV0v)]

        self.L_V2a = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=5.1+1.6*i*gauss(1, self.sigma),y=-1) for i in range(self.nV2a)]
        self.R_V2a = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=5.1+1.6*i*gauss(1, self.sigma),y=1) for i in range(self.nV2a)]
        
        self.L_V1 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=7.1+1.6*i*gauss(1, self.sigma),y=-1) for i in range(self.nV1)]
        self.R_V1 = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt = self.__dt, x=7.1+1.6*i*gauss(1, self.sigma),y=1) for i in range(self.nV1)]
        
        self.L_Muscle = [ Leaky_Integrator(1.0, 3.0, self.__dt, 5.0+1.6*i,-1) for i in range(self.nMuscle)]
        self.R_Muscle = [ Leaky_Integrator(1.0, 3.0, self.__dt, 5.0+1.6*i, 1) for i in range(self.nMuscle)]

    def initSynapses(self):
        ## Declare Synapses
    
        # Below is the declaration of the chemical synapses
        self.L_achsyn_MN_Muscle = TwoExp_syn(0.5, 1.0, -15, self.__dt, 120)
        self.R_achsyn_MN_Muscle = TwoExp_syn(0.5, 1.0, -15, self.__dt, 120)

        self.L_glysyn_dI6_MN = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.R_glysyn_dI6_MN = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.L_glysyn_dI6_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.R_glysyn_dI6_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.L_glysyn_dI6_dI6 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.R_glysyn_dI6_dI6 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)

        self.L_glysyn_V1_MN = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.R_glysyn_V1_MN = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.L_glysyn_V1_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.R_glysyn_V1_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.L_glysyn_V1_V0v = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.R_glysyn_V1_V0v = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.L_glysyn_V1_dI6 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)
        self.R_glysyn_V1_dI6 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_gly)

        self.L_glusyn_V0v_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.R_glusyn_V0v_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)

        self.L_glusyn_V2a_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.R_glusyn_V2a_V2a = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.L_glusyn_V2a_MN = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.R_glusyn_V2a_MN = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.L_glusyn_V2a_dI6 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.R_glusyn_V2a_dI6 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.L_glusyn_V2a_V1 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.R_glusyn_V2a_V1 = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.L_glusyn_V2a_V0v = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)
        self.R_glusyn_V2a_V0v = TwoExp_syn(0.5, 1.0, -15, self.__dt, self.E_glu)

    def initStorageLists(self, nmax):

        ## Initialize Storage tables
        self.Time =zeros(nmax)
        
        self.VLMN =zeros((self.nMN, nmax))
        self.VRMN =zeros((self.nMN, nmax))
        self.VLdI6 = zeros ((self.ndI6,nmax))
        self.VRdI6 = zeros ((self.ndI6,nmax))
        self.VLV0v = zeros ((self.nV0v,nmax))
        self.VRV0v = zeros ((self.nV0v,nmax))
        self.VLV2a = zeros ((self.nV2a,nmax))
        self.VRV2a = zeros ((self.nV2a,nmax))
        self.VLV1 = zeros ((self.nV1,nmax))
        self.VRV1 = zeros ((self.nV1,nmax))
        self.VLMuscle = zeros((self.nMuscle, nmax))
        self.VRMuscle = zeros((self.nMuscle, nmax))
        
        #Lists to store synaptic currents
        
        #gly
        self.LSyn_dI6_MN = zeros((self.ndI6*self.nMN,3))
        self.RSyn_dI6_MN = zeros((self.ndI6*self.nMN,3))
        self.LSyn_dI6_V2a = zeros((self.ndI6*self.nV2a,3))
        self.RSyn_dI6_V2a = zeros((self.ndI6*self.nV2a,3))
        self.LSyn_dI6_dI6 = zeros((self.ndI6*self.ndI6,3))
        self.RSyn_dI6_dI6 = zeros((self.ndI6*self.ndI6,3))
        self.LSyn_V1_MN = zeros((self.nV1*self.nMN,3))
        self.RSyn_V1_MN = zeros((self.nV1*self.nMN,3))
        self.LSyn_V1_V2a = zeros((self.nV1*self.nV2a,3))
        self.RSyn_V1_V2a = zeros((self.nV1*self.nV2a,3))
        self.LSyn_V1_V0v = zeros((self.nV1*self.nV0v,3))
        self.RSyn_V1_V0v = zeros((self.nV1*self.nV0v,3))
        self.LSyn_V1_dI6 = zeros((self.nV1*self.ndI6,3))
        self.RSyn_V1_dI6 = zeros((self.nV1*self.ndI6,3))

        
        #glu
        self.LSyn_V0v_V2a = zeros((self.nV0v*self.nV2a,3))
        self.RSyn_V0v_V2a = zeros((self.nV0v*self.nV2a,3))
        self.LSyn_V2a_V2a = zeros((self.nV2a*self.nV2a,3))
        self.RSyn_V2a_V2a = zeros((self.nV2a*self.nV2a,3))
        self.LSyn_V2a_MN = zeros((self.nV2a*self.nMN,3))
        self.RSyn_V2a_MN = zeros((self.nV2a*self.nMN,3))
        self.LSyn_V2a_dI6 = zeros((self.nV2a*self.ndI6,3))
        self.RSyn_V2a_dI6 = zeros((self.nV2a*self.ndI6,3))
        self.LSyn_V2a_V1 = zeros((self.nV2a*self.nV1,3))
        self.RSyn_V2a_V1 = zeros((self.nV2a*self.nV1,3))
        self.LSyn_V2a_V0v = zeros((self.nV2a*self.nV0v,3))
        self.RSyn_V2a_V0v = zeros((self.nV2a*self.nV0v,3))

        #Ach
        self.LSyn_MN_Muscle = zeros((self.nMN*self.nMuscle,3))
        self.RSyn_MN_Muscle = zeros((self.nMN*self.nMuscle,3))
        
        #Gap
        self.LSGap_MN_MN = zeros((self.nMN,self.nMN))
        self.LSGap_MN_dI6 = zeros((self.nMN,self.ndI6))
        self.LSGap_MN_V0v = zeros((self.nMN,self.nV0v))
        self.LSGap_MN_V2a = zeros((self.nMN,self.nV2a))
        self.LSGap_dI6_MN = zeros((self.ndI6,self.nMN))
        self.LSGap_dI6_dI6 = zeros((self.ndI6,self.ndI6))
        self.LSGap_V0v_MN = zeros((self.nV0v,self.nMN))
        self.LSGap_V0v_V0v = zeros((self.nV0v,self.nV0v))
        self.LSGap_V2a_V2a = zeros((self.nV2a,self.nV2a))
        self.LSGap_V2a_MN = zeros((self.nV2a,self.nMN))
        
        self.RSGap_MN_MN = zeros((self.nMN,self.nMN))
        self.RSGap_MN_dI6 = zeros((self.nMN,self.ndI6))
        self.RSGap_MN_V0v = zeros((self.nMN,self.nV0v))
        self.RSGap_MN_V2a = zeros((self.nMN,self.nV2a))
        self.RSGap_dI6_MN = zeros((self.ndI6,self.nMN))
        self.RSGap_dI6_dI6 = zeros((self.ndI6,self.ndI6))
        self.RSGap_V0v_MN = zeros((self.nV0v,self.nMN))
        self.RSGap_V0v_V0v = zeros((self.nV0v,self.nV0v))
        self.RSGap_V2a_V2a = zeros((self.nV2a,self.nV2a))
        self.RSGap_V2a_MN = zeros((self.nV2a,self.nMN))
        
        
        ### List of synaptic weights
        
        #gly
        self.LW_dI6_MN = zeros((self.ndI6,self.nMN))      
        self.RW_dI6_MN = zeros((self.ndI6,self.nMN))
        self.LW_dI6_V2a = zeros((self.ndI6,self.nV2a))      
        self.RW_dI6_V2a = zeros((self.ndI6,self.nV2a))
        self.LW_dI6_dI6 = zeros((self.ndI6,self.ndI6))      
        self.RW_dI6_dI6 = zeros((self.ndI6,self.ndI6))
        self.LW_V1_MN = zeros((self.nV1,self.nMN))      
        self.RW_V1_MN = zeros((self.nV1,self.nMN))
        self.LW_V1_V2a = zeros((self.nV1,self.nV2a))      
        self.RW_V1_V2a = zeros((self.nV1,self.nV2a))
        self.LW_V1_V0v = zeros((self.nV1,self.nV0v))      
        self.RW_V1_V0v = zeros((self.nV1,self.nV0v))
        self.LW_V1_dI6 = zeros((self.nV1,self.ndI6))      
        self.RW_V1_dI6 = zeros((self.nV1,self.ndI6))
        
        #glu
        self.LW_V0v_V2a = zeros((self.nV0v,self.nV2a))
        self.RW_V0v_V2a = zeros((self.nV0v,self.nV2a))
        self.LW_V2a_V2a = zeros((self.nV2a,self.nV2a))
        self.RW_V2a_V2a = zeros((self.nV2a,self.nV2a))
        self.LW_V2a_MN = zeros((self.nV2a,self.nMN))
        self.RW_V2a_MN = zeros((self.nV2a,self.nMN))
        self.LW_V2a_dI6 = zeros((self.nV2a,self.ndI6))
        self.RW_V2a_dI6 = zeros((self.nV2a,self.ndI6))
        self.LW_V2a_V1 = zeros((self.nV2a,self.nV1))
        self.RW_V2a_V1 = zeros((self.nV2a,self.nV1))
        self.LW_V2a_V0v = zeros((self.nV2a,self.nV0v))
        self.RW_V2a_V0v = zeros((self.nV2a,self.nV0v))
        
        #Ach
        self.LW_MN_Muscle = zeros((self.nMN,self.nMuscle))
        self.RW_MN_Muscle = zeros((self.nMN,self.nMuscle))
        
        
        #List of Gap junctions
        self.LGap_MN_MN = zeros((self.nMN,self.nMN))
        self.LGap_MN_dI6 = zeros((self.nMN,self.ndI6))
        self.LGap_MN_V0v = zeros((self.nMN,self.nV0v))
        self.LGap_dI6_dI6 = zeros((self.ndI6,self.ndI6))
        self.LGap_V0v_V0v = zeros((self.nV0v,self.nV0v))
        self.LGap_V2a_V2a = zeros((self.nV2a,self.nV2a))
        self.LGap_V2a_MN = zeros((self.nV2a,self.nMN))
        
        self.RGap_MN_MN = zeros((self.nMN,self.nMN))
        self.RGap_MN_dI6 = zeros((self.nMN,self.ndI6))
        self.RGap_MN_V0v = zeros((self.nMN,self.nV0v))
        self.RGap_dI6_dI6 = zeros((self.ndI6,self.ndI6))
        self.RGap_V0v_V0v = zeros((self.nV0v,self.nV0v))
        self.RGap_V2a_V2a = zeros((self.nV2a,self.nV2a))
        self.RGap_V2a_MN = zeros((self.nV2a,self.nMN))
        
        #list to store membrane potential self.residuals 
        self.resLMN=zeros((self.nMN,3))
        self.resRMN=zeros((self.nMN,3))
        self.resLdI6 = zeros((self.ndI6,3))
        self.resRdI6 = zeros((self.ndI6,3))
        self.resLV0v = zeros((self.nV0v,3))
        self.resRV0v = zeros((self.nV0v,3))
        self.resLV2a = zeros((self.nV2a,3))
        self.resRV2a = zeros((self.nV2a,3))
        self.resLV1 = zeros((self.nV1,3))
        self.resRV1 = zeros((self.nV1,3))
        self.resLMuscle = zeros((self.nMuscle,2))
        self.resRMuscle = zeros((self.nMuscle,2))

    def initAndSetDistanceMatrices(self):
        #Distance Matrix 
        self.LD_MN_MN = zeros((self.nMN , self.nMN))
        self.LD_MN_dI6 = zeros((self.nMN , self.ndI6))
        self.LD_MN_V0v = zeros((self.nMN , self.nV0v))
        self.LD_dI6_dI6 = zeros((self.ndI6 , self.ndI6))
        self.LD_V0v_V0v = zeros((self.nV0v , self.nV0v))
        self.LD_V0v_V2a = zeros((self.nV0v , self.nV2a))
        
        self.LD_V2a_V2a = zeros((self.nV2a , self.nV2a))     
        self.LD_V2a_MN = zeros((self.nV2a , self.nMN))           
        self.LD_V2a_dI6 = zeros((self.nV2a , self.ndI6))   
        self.LD_V2a_V1 = zeros((self.nV2a , self.nV1))   
        self.LD_V2a_V0v = zeros((self.nV2a , self.nV0v))   
        
        self.LD_V1_MN = zeros((self.nV1 , self.nMN))
        self.LD_V1_V0v = zeros((self.nV1 , self.nV0v))
        self.LD_V1_dI6 = zeros((self.nV1 , self.ndI6))
        
        self.LD_dI6_MN = zeros((self.ndI6 , self.nMN))     
        self.LD_dI6_V2a = zeros((self.ndI6 , self.nV2a))   
        
        self.RD_MN_MN = zeros((self.nMN , self.nMN))
        self.RD_MN_dI6 = zeros((self.nMN , self.ndI6))
        self.RD_MN_V0v = zeros((self.nMN , self.nV0v))
        self.RD_dI6_dI6 = zeros((self.ndI6 , self.ndI6))
        self.RD_V0v_V0v = zeros((self.nV0v , self.nV0v))
        self.RD_V0v_V2a = zeros((self.nV0v , self.nV2a))
        
        self.RD_V2a_V2a = zeros((self.nV2a , self.nV2a))     
        self.RD_V2a_MN = zeros((self.nV2a , self.nMN))           
        self.RD_V2a_dI6 = zeros((self.nV2a , self.ndI6))   
        self.RD_V2a_V1 = zeros((self.nV2a , self.nV1))   
        self.RD_V2a_V0v = zeros((self.nV2a , self.nV0v))
        
        self.RD_V1_MN = zeros((self.nV1 , self.nMN))
        self.RD_V1_V0v = zeros((self.nV1 , self.nV0v))
        self.RD_V1_dI6 = zeros((self.nV1 , self.ndI6))
        
        self.RD_dI6_MN = zeros((self.ndI6 , self.nMN))     
        self.RD_dI6_V2a = zeros((self.ndI6 , self.nV2a))   
        
        ## Compute distance between Neurons
        
        #LEFT
                
        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                self.LD_MN_MN[k,l] = Distance(self.L_MN[k].x,self.L_MN[l].x,self.L_MN[k].y,self.L_MN[l].y)
                
        for k in range (0, self.nMN):
            for l in range (0, self.ndI6):
                self.LD_MN_dI6[k,l] = Distance(self.L_MN[k].x,self.L_dI6[l].x,self.L_MN[k].y,self.L_dI6[l].y)
                
        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                self.LD_MN_V0v[k,l] = Distance(self.L_MN[k].x,self.L_V0v[l].x,self.L_MN[k].y,self.L_V0v[l].y)
    
        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                self.LD_V2a_V2a[k,l] = Distance(self.L_V2a[k].x,self.L_V2a[l].x,self.L_V2a[k].y,self.L_V2a[l].y)
                
        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                self.LD_V2a_MN[k,l] = Distance(self.L_V2a[k].x,self.L_MN[l].x,self.L_V2a[k].y,self.L_MN[l].y)
                        
        for k in range (0, self.nV2a):
            for l in range (0, self.ndI6):
                self.LD_V2a_dI6[k,l] = Distance(self.L_V2a[k].x,self.L_dI6[l].x,self.L_V2a[k].y,self.L_dI6[l].y)
                
        for k in range (0, self.nV2a):
            for l in range (0, self.nV1):
                self.LD_V2a_V1[k,l] = Distance(self.L_V2a[k].x,self.L_V1[l].x,self.L_V2a[k].y,self.L_V1[l].y)

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                self.LD_V2a_V0v[k,l] = Distance(self.L_V2a[k].x,self.L_V0v[l].x,self.L_V2a[k].y,self.L_V0v[l].y)

        for k in range (0, self.nV1):
            for l in range (0, self.nMN):
                self.LD_V1_MN[k,l] = Distance(self.L_V1[k].x,self.L_MN[l].x,self.L_V1[k].y,self.L_MN[l].y) 
                
        for k in range (0, self.nV1):
            for l in range (0, self.nV0v):
                self.LD_V1_V0v[k,l] = Distance(self.L_V1[k].x,self.L_V0v[l].x,self.L_V1[k].y,self.L_V0v[l].y)    
        
        for k in range (0, self.nV1):
            for l in range (0, self.ndI6):
                self.LD_V1_dI6[k,l] = Distance(self.L_V1[k].x,self.L_dI6[l].x,self.L_V1[k].y,self.L_dI6[l].y) 
                    
        for k in range (0, self.ndI6):
            for l in range (0, self.nMN):
                self.LD_dI6_MN[k,l] = Distance(self.L_dI6[k].x,self.R_MN[l].x,self.L_dI6[k].y,self.R_MN[l].y) #Contralateral
                        
        for k in range (0, self.ndI6):
            for l in range (0, self.nV2a):
                self.LD_dI6_V2a[k,l] = Distance(self.L_dI6[k].x,self.R_V2a[l].x,self.L_dI6[k].y,self.R_V2a[l].y) #Contralateral
                
        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):
                self.LD_dI6_dI6[k,l] = Distance(self.L_dI6[k].x,self.L_dI6[l].x,self.L_dI6[k].y,self.L_dI6[l].y) #Contralateral
                        
        for k in range (0, self.nV0v):
            for l in range (0, self.nV2a):
                self.LD_V0v_V2a[k,l] = Distance(self.L_V0v[k].x,self.R_V2a[l].x,self.L_V0v[k].y,self.R_V2a[l].y) #Contralateral
                
        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                self.LD_V0v_V0v[k,l] = Distance(self.L_V0v[k].x,self.L_V0v[l].x,self.L_V0v[k].y,self.L_V0v[l].y) #Contralateral
                

        
        #RIGHT
        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                self.RD_MN_MN[k,l] = Distance(self.R_MN[k].x,self.R_MN[l].x,self.R_MN[k].y,self.R_MN[l].y)
                
        for k in range (0, self.nMN):
            for l in range (0, self.ndI6):
                self.RD_MN_dI6[k,l] = Distance(self.R_MN[k].x,self.R_dI6[l].x,self.R_MN[k].y,self.R_dI6[l].y)
                
        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                self.RD_MN_V0v[k,l] = Distance(self.R_MN[k].x,self.R_V0v[l].x,self.R_MN[k].y,self.R_V0v[l].y)
                
        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                self.RD_V2a_V2a[k,l] = Distance(self.R_V2a[k].x,self.R_V2a[l].x,self.R_V2a[k].y,self.R_V2a[l].y)
                
        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                self.RD_V2a_MN[k,l] = Distance(self.R_V2a[k].x,self.R_MN[l].x,self.R_V2a[k].y,self.R_MN[l].y)
                
        for k in range (0, self.nV2a):
            for l in range (0, self.ndI6):
                self.RD_V2a_dI6[k,l] = Distance(self.R_V2a[k].x,self.R_dI6[l].x,self.R_V2a[k].y,self.R_dI6[l].y)
                
        for k in range (0, self.nV2a):
            for l in range (0, self.nV1):
                self.RD_V2a_V1[k,l] = Distance(self.R_V2a[k].x,self.R_V1[l].x,self.R_V2a[k].y,self.R_V1[l].y)

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                self.RD_V2a_V0v[k,l] = Distance(self.R_V2a[k].x,self.R_V0v[l].x,self.R_V2a[k].y,self.R_V0v[l].y)

        for k in range (0, self.nV1):
            for l in range (0, self.nMN):
                self.RD_V1_MN[k,l] = Distance(self.R_V1[k].x,self.R_MN[l].x,self.R_V1[k].y,self.R_MN[l].y) 
                
        for k in range (0, self.nV1):
            for l in range (0, self.nV0v):
                self.RD_V1_V0v[k,l] = Distance(self.R_V1[k].x,self.R_V0v[l].x,self.R_V1[k].y,self.R_V0v[l].y)     
                
        for k in range (0, self.nV1):
            for l in range (0, self.ndI6):
                self.RD_V1_dI6[k,l] = Distance(self.R_V1[k].x,self.R_dI6[l].x,self.R_V1[k].y,self.R_dI6[l].y)
                
        for k in range (0, self.ndI6):
            for l in range (0, self.nMN):
                self.RD_dI6_MN[k,l] = Distance(self.R_dI6[k].x,self.L_MN[l].x,self.R_dI6[k].y,self.L_MN[l].y) #Contralateral
                
        for k in range (0, self.ndI6):
            for l in range (0, self.nV2a):
                self.RD_dI6_V2a[k,l] = Distance(self.R_dI6[k].x,self.L_V2a[l].x,self.R_dI6[k].y,self.L_V2a[l].y) #Contralateral
                
        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):
                self.RD_dI6_dI6[k,l] = Distance(self.R_dI6[k].x,self.R_dI6[l].x,self.R_dI6[k].y,self.R_dI6[l].y) 
                
        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                self.RD_V0v_V0v[k,l] = Distance(self.R_V0v[k].x,self.R_V0v[l].x,self.R_V0v[k].y,self.R_V0v[l].y)
        
        for k in range (0, self.nV0v):
            for l in range (0, self.nV2a):
                self.RD_V0v_V2a[k,l] = Distance(self.R_V0v[k].x,self.L_V2a[l].x,self.R_V0v[k].y,self.L_V2a[l].y) #Contralateral

    def rangeNoiseMultiplier(self):
        return 1

    def weightNoiseMultiplier(self):
        return 1

    def computeSynapticAndGapWeights(self):
        ## Compute synaptic weights and gap junction weights
        
        #region LEFT
        #region gap junctions
        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.LD_MN_MN[k,l] < self.rangeMN_MN * self.rangeNoiseMultiplier()):
                    self.LGap_MN_MN[k,l] = self.MN_MN_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.LGap_MN_MN[k,l] = 0.0

        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):
                if (self.rangeMin < self.LD_dI6_dI6[k,l] < self.rangedI6_dI6 * self.rangeNoiseMultiplier()):
                    self.LGap_dI6_dI6[k,l] = self.dI6_dI6_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.LGap_dI6_dI6[k,l] = 0.0

        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.LD_V0v_V0v[k,l] < self.rangeV0v_V0v * self.rangeNoiseMultiplier()):
                    self.LGap_V0v_V0v[k,l] = self.V0v_V0v_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.LGap_V0v_V0v[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.LD_V2a_V2a[k,l] < self.rangeV2a_V2a_gap * self.rangeNoiseMultiplier()):
                    self.LGap_V2a_V2a[k,l] = self.V2a_V2a_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.LGap_V2a_V2a[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.LD_V2a_MN[k,l] < self.rangeV2a_MN * self.rangeNoiseMultiplier()):
                    self.LGap_V2a_MN[k,l] = self.V2a_MN_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.LGap_V2a_MN[k,l] = 0.0
        
        for k in range (0, self.nMN):
            for l in range (0, self.ndI6):
                if (self.L_dI6[l].x - self.rangeMN_dI6 * self.rangeNoiseMultiplier() < self.L_MN[k].x< self.L_dI6[l].x + self.rangeMN_dI6 * self.rangeNoiseMultiplier()):    
                    self.LGap_MN_dI6[k,l] = self.MN_dI6_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.LGap_MN_dI6[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                if (self.L_V0v[l].x - self.rangeMN_V0v * self.rangeNoiseMultiplier() <self.L_MN[k].x < self.L_V0v[l].x + self.rangeMN_V0v * self.rangeNoiseMultiplier()):
                    self.LGap_MN_V0v[k,l] = self.MN_V0v_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.LGap_MN_V0v[k,l] = 0.0
                    
        #endregion
        
        #region chemical synapses
        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.LD_V2a_V2a[k,l] < self.rangeV2a_V2a_syn * self.rangeNoiseMultiplier() and self.L_V2a[k].x < self.L_V2a[l].x): #the second condition is because the connection is descending
                    self.LW_V2a_V2a[k,l] = self.V2a_V2a_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_V2a_V2a[k,l] = 0.0
        
        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                if (0 < self.LD_V2a_MN[k,l] < self.rangeV2a_MN_desc * self.rangeNoiseMultiplier()  and self.L_V2a[k].x < self.L_MN[l].x) or \
                   (self.rangeMin < self.LD_V2a_MN[k,l] < self.rangeV2a_MN_asc * self.rangeNoiseMultiplier()  and self.L_V2a[k].x > self.L_MN[l].x): #the second condition is because the connection is descending and short ascending branch, used to be 0<self.LD_V2a_MN[k,l]<10
                    self.LW_V2a_MN[k,l] = self.V2a_MN_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_V2a_MN[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.ndI6):
                if (0<self.LD_V2a_dI6[k,l] < self.rangeV2a_dI6 * self.rangeNoiseMultiplier()  and self.L_V2a[k].x < self.L_dI6[l].x): #the second condition is because the connection is descending
                    self.LW_V2a_dI6[k,l] = self.V2a_dI6_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_V2a_dI6[k,l] = 0.0
        
        for k in range (0, self.nV2a):
            for l in range (0, self.nV1):
                if (0<self.LD_V2a_V1[k,l] < self.rangeV2a_V1 * self.rangeNoiseMultiplier()  and self.L_V2a[k].x < self.L_V1[l].x):   #the second condition is because the connection is descending
                    self.LW_V2a_V1[k,l] = self.V2a_V1_syn_weight * self.weightNoiseMultiplier()  
                else:
                    self.LW_V2a_V1[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                if (0 < self.LD_V2a_V0v[k,l] < self.rangeV2a_V0v_desc * self.rangeNoiseMultiplier() and self.L_V2a[k].x < self.L_V0v[l].x) or \
                    (self.rangeMin < self.LD_V2a_V0v[k,l] < self.rangeV2a_V0v_asc * self.rangeNoiseMultiplier()  and self.L_V2a[k].x > self.L_V0v[l].x):    #the second condition is because the connection is descending or short ascending
                    self.LW_V2a_V0v[k,l] = self.V2a_V0v_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_V2a_V0v[k,l] = 0.0                
    
        for k in range (0, self.nV1):
            for l in range (0, self.nMN):
                if (0<self.LD_V1_MN[k,l] < self.rangeV1_MN * self.rangeNoiseMultiplier()  and self.L_MN[l].x < self.L_V1[k].x):    #the second condition is because the connection is ascending
                    self.LW_V1_MN[k,l] = self.V1_MN_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_V1_MN[k,l] = 0.0

        for k in range (0, self.nV1):
            for l in range (0, self.nV2a):
                if (0<self.LD_V2a_V1[l,k] < self.rangeV1_V2a * self.rangeNoiseMultiplier()  and self.L_V2a[l].x < self.L_V1[k].x):    #the second condition is because the connection is ascending
                    self.LW_V1_V2a[k,l] = self.V1_V2a_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_V1_V2a[k,l] = 0.0

        for k in range (0, self.nV1):
            for l in range (0, self.nV0v):
                if (0<self.LD_V1_V0v[k,l] < self.rangeV1_V0v * self.rangeNoiseMultiplier()  and self.L_V0v[l].x < self.L_V1[k].x):    #the second condition is because the connection is ascending
                    self.LW_V1_V0v[k,l] = self.V1_V0v_syn_weight * self.weightNoiseMultiplier() 
                else:
                    self.LW_V1_V0v[k,l] = 0.0
                    
        for k in range (0, self.nV1):
            for l in range (0, self.ndI6):
                if (0<self.LD_V1_dI6[k,l] < self.rangeV1_dI6 * self.rangeNoiseMultiplier()  and self.L_dI6[l].x < self.L_V1[k].x):    #the second condition is because the connection is ascending
                    self.LW_V1_dI6[k,l] = self.V1_dI6_syn_weight * self.weightNoiseMultiplier()  
                else:
                    self.LW_V1_dI6[k,l] = 0.0
        
        for k in range (0, self.ndI6):
            for l in range (0, self.nMN):
                if (0 < self.LD_dI6_MN[k,l] < self.rangedI6_MN_desc * self.rangeNoiseMultiplier()  and self.L_dI6[k].x < self.R_MN[l].x) or \
                   (self.rangeMin < self.LD_dI6_MN[k,l] < self.rangedI6_MN_asc * self.rangeNoiseMultiplier()  and self.L_dI6[k].x > self.R_MN[l].x):   #because contralateral and bifurcating
                    self.LW_dI6_MN[k,l] = self.dI6_MN_syn_weight * self.weightNoiseMultiplier()  
                else:
                    self.LW_dI6_MN[k,l] = 0.0
        
        for k in range (0, self.ndI6):
            for l in range (0, self.nV2a):
                if (0 < self.LD_dI6_V2a[k,l] < self.rangedI6_V2a_desc * self.rangeNoiseMultiplier()  and self.L_dI6[k].x < self.R_V2a[l].x) or \
                   (self.rangeMin < self.LD_dI6_V2a[k,l] < self.rangedI6_V2a_asc * self.rangeNoiseMultiplier()  and self.L_dI6[k].x > self.R_V2a[l].x):     #because contralateral  and bifurcating
                    self.LW_dI6_V2a[k,l] = self.dI6_V2a_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_dI6_V2a[k,l] = 0.0
        
        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):
                if (0 < self.LD_dI6_dI6[k,l] < self.rangedI6_dI6_desc * self.rangeNoiseMultiplier()  and self.L_dI6[k].x < self.R_dI6[l].x) or \
                   (self.rangeMin < self.LD_dI6_dI6[k,l] < self.rangedI6_dI6_asc * self.rangeNoiseMultiplier()  and self.L_dI6[k].x > self.R_dI6[l].x):     #because contralateral  and bifurcating
                    self.LW_dI6_dI6[k,l] = self.dI6_dI6_syn_weight * gauss(1, self.sigma_LR) * self.weightNoiseMultiplier()
                else:
                    self.LW_dI6_dI6[k,l] = 0.0
        
        for k in range (0, self.nV0v):
            for l in range (0, self.nV2a):
                if (0 < self.LD_V0v_V2a[k,l] < self.rangeV0v_V2a_desc * self.rangeNoiseMultiplier()  and self.L_V0v[k].x < self.R_V2a[l].x) or \
                   (self.rangeMin < self.LD_V0v_V2a[k,l] < self.rangeV0v_V2a_asc * self.rangeNoiseMultiplier()  and self.L_V0v[k].x > self.R_V2a[l].x): #because contralateral and bifurcating
                    self.LW_V0v_V2a[k,l] = self.V0v_V2a_syn_weight * self.weightNoiseMultiplier() 
                else:
                    self.LW_V0v_V2a[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nMuscle):
                if (self.L_Muscle[l].x - self.rangeMN_Muscle < self.L_MN[k].x < self.L_Muscle[l].x + self.rangeMN_Muscle):         #this connection is segmental
                    self.LW_MN_Muscle[k,l] = self.MN_Muscle_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.LW_MN_Muscle[k,l] = 0.0
        #endregion

        #endregion   

        #region RIGHT

        #region gap junctions
        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.RD_MN_MN[k,l] < self.rangeMN_MN * self.rangeNoiseMultiplier()):
                    self.RGap_MN_MN[k,l] =  self.MN_MN_gap_weight * self.weightNoiseMultiplier() 
                else:
                    self.RGap_MN_MN[k,l] = 0.0
                    
        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):
                if (self.rangeMin < self.RD_dI6_dI6[k,l]< self.rangedI6_dI6 * self.rangeNoiseMultiplier()):
                    self.RGap_dI6_dI6[k,l] = self.dI6_dI6_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.RGap_dI6_dI6[k,l] = 0.0

                    
        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.RD_V0v_V0v[k,l] < self.rangeV0v_V0v * self.rangeNoiseMultiplier()):
                    self.RGap_V0v_V0v[k,l] = self.V0v_V0v_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.RGap_V0v_V0v[k,l] = 0.0
                                
        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.RD_V2a_V2a[k,l] < self.rangeV2a_V2a_gap * self.rangeNoiseMultiplier()):
                    self.RGap_V2a_V2a[k,l] = self.V2a_V2a_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.RGap_V2a_V2a[k,l] = 0.0
                    
        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.RD_V2a_MN[k,l] < self.rangeV2a_MN * self.rangeNoiseMultiplier()):
                    self.RGap_V2a_MN[k,l] = self.V2a_MN_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.RGap_V2a_MN[k,l] = 0.0
                    
        for k in range (0, self.nMN):
            for l in range (0, self.ndI6):
                if (self.R_dI6[l].x - self.rangeMN_dI6 * self.rangeNoiseMultiplier() < self.R_MN[k].x < self.R_dI6[l].x + self.rangeMN_dI6 * self.rangeNoiseMultiplier()):
                    self.RGap_MN_dI6[k,l] = self.MN_dI6_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.RGap_MN_dI6[k,l] = 0.0
                    
        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                if (self.R_V0v[l].x - self.rangeMN_V0v * self.rangeNoiseMultiplier() < self.R_MN[k].x < self.R_V0v[l].x + self.rangeMN_V0v * self.rangeNoiseMultiplier()):
                    self.RGap_MN_V0v[k,l] = self.MN_V0v_gap_weight * self.weightNoiseMultiplier()
                else:
                    self.RGap_MN_V0v[k,l] = 0.0
        #endregion
        
        #region chemical synapses
        
        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.RD_V2a_V2a[k,l] < self.rangeV2a_V2a_syn * self.rangeNoiseMultiplier()  and self.R_V2a[k].x < self.R_V2a[l].x): #the second condition is because the connection is descending
                    self.RW_V2a_V2a[k,l] = self.V2a_V2a_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V2a_V2a[k,l] = 0.0
        
        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                if (0 < self.RD_V2a_MN[k,l] < self.rangeV2a_MN_desc * self.rangeNoiseMultiplier()  and self.R_V2a[k].x < self.R_MN[l].x) or \
                   (self.rangeMin < self.RD_V2a_MN[k,l] < self.rangeV2a_MN_asc * self.rangeNoiseMultiplier()  and self.R_V2a[k].x > self.R_MN[l].x): #the second condition is because the connection is descending and short ascending branch
                    self.RW_V2a_MN[k,l] =  self.V2a_MN_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V2a_MN[k,l] = 0.0
                    
        for k in range (0, self.nV2a):
            for l in range (0, self.ndI6):
                if (0<self.RD_V2a_dI6[k,l] < self.rangeV2a_dI6 * self.rangeNoiseMultiplier() and self.R_V2a[k].x < self.R_dI6[l].x): #the second condition is because the connection is descending and short ascending branch
                    self.RW_V2a_dI6[k,l] =  self.V2a_dI6_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V2a_dI6[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nV1):
                if (0<self.RD_V2a_V1[k,l] < self.rangeV2a_V1 * self.rangeNoiseMultiplier() and self.R_V2a[k].x < self.R_V1[l].x):    #the second condition is because the connection is descending
                    self.RW_V2a_V1[k,l] =  self.V2a_V1_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V2a_V1[k,l] = 0.0
                    
        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                if (0 < self.RD_V2a_V0v[k,l] < self.rangeV2a_V0v_desc * self.rangeNoiseMultiplier() and self.R_V2a[k].x < self.R_V0v[l].x) or \
                   (self.rangeMin < self.RD_V2a_V0v[k,l] < self.rangeV2a_V0v_asc * self.rangeNoiseMultiplier() and self.R_V2a[k].x > self.R_V0v[l].x):    #the second condition is because the connection is descending and short ascending branch
                    self.RW_V2a_V0v[k,l] =  self.V2a_V0v_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V2a_V0v[k,l] = 0.0   

        for k in range (0, self.nV1):
            for l in range (0, self.nMN):
                if (0<self.RD_V1_MN[k,l] < self.rangeV1_MN * self.rangeNoiseMultiplier() and self.R_MN[l].x < self.R_V1[k].x):    #the second condition is because the connection is ascending
                    self.RW_V1_MN[k,l] = self.V1_MN_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V1_MN[k,l] = 0.0

        for k in range (0, self.nV1):
            for l in range (0, self.nV2a):
                if (0<self.RD_V2a_V1[l,k] < self.rangeV1_V2a * self.rangeNoiseMultiplier() and self.R_V2a[l].x < self.R_V1[k].x):    #the second condition is because the connection is ascending
                    self.RW_V1_V2a[k,l] = self.V1_V2a_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V1_V2a[k,l] = 0.0

        for k in range (0, self.nV1):
            for l in range (0, self.nV0v):
                if (0 < self.RD_V1_V0v[k,l] < self.rangeV1_V0v * self.rangeNoiseMultiplier() and self.R_V0v[l].x < self.R_V1[k].x):    #the second condition is because the connection is ascending
                    self.RW_V1_V0v[k,l] = self.V1_V0v_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V1_V0v[k,l] = 0.0
                    
        for k in range (0, self.nV1):
            for l in range (0, self.ndI6):
                if (0<self.RD_V1_dI6[k,l] < self.rangeV1_dI6 * self.rangeNoiseMultiplier() and self.R_dI6[l].x < self.R_V1[k].x):    #the second condition is because the connection is ascending
                    self.RW_V1_dI6[k,l] = self.V1_dI6_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V1_dI6[k,l] = 0.0    
                    
        for k in range (0, self.ndI6):
            for l in range (0, self.nMN):
                if (0 < self.RD_dI6_MN[k,l] < self.rangedI6_MN_desc * self.rangeNoiseMultiplier() and self.R_dI6[k].x < self.L_MN[l].x) or \
                   (self.rangeMin < self.RD_dI6_MN[k,l] < self.rangedI6_MN_asc * self.rangeNoiseMultiplier() and self.R_dI6[k].x > self.L_MN[l].x):   #because contralateral
                    self.RW_dI6_MN[k,l] = self.dI6_MN_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_dI6_MN[k,l] = 0.0
                    
        for k in range (0, self.ndI6):
            for l in range (0, self.nV2a):
                if (0 < self.RD_dI6_V2a[k,l] < self.rangedI6_V2a_desc * self.rangeNoiseMultiplier() and self.R_dI6[k].x < self.L_V2a[l].x) or \
                   (self.rangeMin < self.RD_dI6_V2a[k,l] < self.rangedI6_V2a_asc * self.rangeNoiseMultiplier() and self.R_dI6[k].x > self.L_V2a[l].x): #because contralateral
                    self.RW_dI6_V2a[k,l] = self.dI6_V2a_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_dI6_V2a[k,l] = 0.0

        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):
                if (0 < self.RD_dI6_dI6[k,l] < self.rangedI6_dI6_desc * self.rangeNoiseMultiplier() and self.R_dI6[k].x < self.L_dI6[l].x) or \
                   (self.rangeMin < self.RD_dI6_dI6[k,l] < self.rangedI6_dI6_asc * self.rangeNoiseMultiplier() and self.R_dI6[k].x > self.L_dI6[l].x):     #because contralateral
                    self.RW_dI6_dI6[k,l] = self.dI6_dI6_syn_weight *gauss(1, self.sigma_LR) * self.weightNoiseMultiplier()
                else:
                    self.RW_dI6_dI6[k,l] = 0.0

        for k in range (0, self.nV0v):
            for l in range (0, self.nV2a):
                if (0 < self.RD_V0v_V2a[k,l] < self.rangeV0v_V2a_desc * self.rangeNoiseMultiplier() and self.R_V0v[k].x < self.L_V2a[l].x) or \
                   (self.rangeMin < self.RD_V0v_V2a[k,l] < self.rangeV0v_V2a_asc * self.rangeNoiseMultiplier() and self.R_V0v[k].x > self.L_V2a[l].x): #because contralateral and bifurcating
                    self.RW_V0v_V2a[k,l] = self.V0v_V2a_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_V0v_V2a[k,l] = 0.0        

        for k in range (0, self.nMN):
            for l in range (0, self.nMuscle):
                if (self.R_Muscle[l].x - self.rangeMN_Muscle < self.R_MN[k].x < self.R_Muscle[l].x + self.rangeMN_Muscle):         #it is segmental
                    self.RW_MN_Muscle[k,l] = self.MN_Muscle_syn_weight * self.weightNoiseMultiplier()
                else:
                    self.RW_MN_Muscle[k,l] = 0.0

        #endregion

        #endregion

    def getStimulus(self, t):
        if t > 2000: # Let the initial conditions dissipate for the first 200 ms
            return self.stim0
        return 0

    #initializeMembranePotentials is called from the mainLoop.
    #Sets initial membrane potentials to -70 for neurons (with u=-14, stim=-70), 
    # except -64 for V2a's and V1's (u=-16, stim=-64)
    # 0 for muscle cells
    def initializeMembranePotentials(self):
        ## Initialize membrane potential values           
    
        for k in range (0, self.nMN):
            self.resLMN[k,:] = self.L_MN[k].getNextVal(-70,-14,-70)
            self.VLMN[k,0] = self.resLMN[k,0]
            
            self.resRMN[k,:] = self.R_MN[k].getNextVal(-70,-14,-70)
            self.VRMN[k,0] = self.resRMN[k,0]
            
        for k in range (0, self.ndI6):
            self.resLdI6[k,:] = self.L_dI6[k].getNextVal(-70,-14,-70)
            self.VLdI6[k,0] = self.resLdI6[k,0]
            
            self.resRdI6[k,:] = self.R_dI6[k].getNextVal(-70,-14,-70)
            self.VRdI6[k,0] = self.resRdI6[k,0]
            
        for k in range (0, self.nV0v):
            self.resLV0v[k,:] = self.L_V0v[k].getNextVal(-70,-14,-70)
            self.VLV0v[k,0] = self.resLV0v[k,0]
            
            self.resRV0v[k,:] = self.R_V0v[k].getNextVal(-70,-14,-70)
            self.VRV0v[k,0] = self.resRV0v[k,0]
            
        for k in range (0, self.nV2a):
            self.resLV2a[k,:] = self.L_V2a[k].getNextVal(-64,-16,-64)
            self.VLV2a[k,0] = self.resLV2a[k,0]
            
            self.resRV2a[k,:] = self.R_V2a[k].getNextVal(-64,-16,-64)
            self.VRV2a[k,0] = self.resRV2a[k,0]
            
        for k in range (0, self.nV1):
            self.resLV1[k,:] = self.L_V1[k].getNextVal(-64,-16,-64)
            self.VLV1[k,0] = self.resLV1[k,0]
            
            self.resRV1[k,:] = self.R_V1[k].getNextVal(-64,-16,-64)
            self.VRV1[k,0] = self.resRV1[k,0]
        
        for k in range (0, self.nMuscle):
            self.resLMuscle[k,:] = self.L_Muscle[k].getNextVal(0,0)
            self.VLMuscle[k,0] = self.resLMuscle[k,0]
            
            self.resRMuscle[k,:] = self.R_Muscle[k].getNextVal(0,0)
            self.VRMuscle[k,0] = self.resRMuscle[k,0]

    def calcSynapticOutputs(self, t):
        ## Calculate synaptic currents
        self.calcdI6SynOutputs(t)
        self.calcV0vSynOutputs(t)
        self.calcV2aSynOutputs(t)
        self.calcV1SynOutputs(t)
        for k in range (0, self.nMN):
            for l in range (0, self.nMuscle):
                self.LSyn_MN_Muscle[self.nMuscle*k+l,:] = self.L_achsyn_MN_Muscle.getNextVal(self.VLMN[k,t-10], self.VLMuscle[l,t-1], self.LSyn_MN_Muscle[self.nMuscle*k+l,1], self.LSyn_MN_Muscle[self.nMuscle*k+l,2])
                self.RSyn_MN_Muscle[self.nMuscle*k+l,:] = self.R_achsyn_MN_Muscle.getNextVal(self.VRMN[k,t-10], self.VRMuscle[l,t-1], self.RSyn_MN_Muscle[self.nMuscle*k+l,1], self.RSyn_MN_Muscle[self.nMuscle*k+l,2])

    def calcdI6SynOutputs(self, t):
        for k in range (0, self.ndI6):
            for l in range (0, self.nMN):
                self.LSyn_dI6_MN[self.nMN*k+l,:] = self.L_glysyn_dI6_MN.getNextVal(self.VLdI6[k,t-int(self.LD_dI6_MN[k,l]/(self.__dt * self.cv))], self.VRMN[l,t-1], self.LSyn_dI6_MN[self.nMN*k+l,1], self.LSyn_dI6_MN[self.nMN*k+l,2]) #Contralateral
                self.RSyn_dI6_MN[self.nMN*k+l,:] = self.R_glysyn_dI6_MN.getNextVal(self.VRdI6[k,t-int(self.RD_dI6_MN[k,l]/(self.__dt * self.cv))], self.VLMN[l,t-1], self.RSyn_dI6_MN[self.nMN*k+l,1], self.RSyn_dI6_MN[self.nMN*k+l,2]) #Contralateral

        for k in range (0, self.ndI6):
            for l in range (0, self.nV2a):
                self.LSyn_dI6_V2a[self.nV2a*k+l,:] = self.L_glysyn_dI6_V2a.getNextVal(self.VLdI6[k,t-int(self.LD_dI6_V2a[k,l]/(self.__dt * self.cv))], self.VRV2a[l,t-1], self.LSyn_dI6_V2a[self.nV2a*k+l,1], self.LSyn_dI6_V2a[self.nV2a*k+l,2])  #Contralateral
                self.RSyn_dI6_V2a[self.nV2a*k+l,:] = self.R_glysyn_dI6_V2a.getNextVal(self.VRdI6[k,t-int(self.RD_dI6_V2a[k,l]/(self.__dt * self.cv))], self.VLV2a[l,t-1], self.RSyn_dI6_V2a[self.nV2a*k+l,1], self.RSyn_dI6_V2a[self.nV2a*k+l,2])  #Contralateral

        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):
                self.LSyn_dI6_dI6[self.ndI6*k+l,:] = self.L_glysyn_dI6_dI6.getNextVal(self.VLdI6[k,t-int(self.LD_dI6_dI6[k,l]/(self.__dt * self.cv))], self.VRdI6[l,t-1], self.LSyn_dI6_dI6[self.ndI6*k+l,1], self.LSyn_dI6_dI6[self.ndI6*k+l,2])  #Contralateral
                self.RSyn_dI6_dI6[self.ndI6*k+l,:] = self.R_glysyn_dI6_dI6.getNextVal(self.VRdI6[k,t-int(self.RD_dI6_dI6[k,l]/(self.__dt * self.cv))], self.VLdI6[l,t-1], self.RSyn_dI6_dI6[self.ndI6*k+l,1], self.RSyn_dI6_dI6[self.ndI6*k+l,2])  #Contralateral

    def calcV0vSynOutputs(self, t):
        for k in range (0, self.nV0v):
            for l in range (0, self.nV2a):
                self.LSyn_V0v_V2a[self.nV2a*k+l,:] = self.L_glusyn_V0v_V2a.getNextVal(self.VLV0v[k,t-int(self.LD_V0v_V2a[k,l]/(self.__dt * self.cv))], self.VRV2a[l,t-1], self.LSyn_V0v_V2a[self.nV2a*k+l,1], self.LSyn_V0v_V2a[self.nV2a*k+l,2])  #Contralateral
                self.RSyn_V0v_V2a[self.nV2a*k+l,:] = self.R_glusyn_V0v_V2a.getNextVal(self.VRV0v[k,t-int(self.RD_V0v_V2a[k,l]/(self.__dt * self.cv))], self.VLV2a[l,t-1], self.RSyn_V0v_V2a[self.nV2a*k+l,1], self.RSyn_V0v_V2a[self.nV2a*k+l,2])  #Contralateral

    def calcV2aSynOutputs(self, t):
        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                self.LSyn_V2a_V2a[self.nV2a*k+l,:] = self.L_glusyn_V2a_V2a.getNextVal(self.VLV2a[k,t-int(self.LD_V2a_V2a[k,l]/(self.__dt * self.cv))], self.VLV2a[l,t-1], self.LSyn_V2a_V2a[self.nV2a*k+l,1], self.LSyn_V2a_V2a[self.nV2a*k+l,2])  #Contralateral
                self.RSyn_V2a_V2a[self.nV2a*k+l,:] = self.R_glusyn_V2a_V2a.getNextVal(self.VRV2a[k,t-int(self.RD_V2a_V2a[k,l]/(self.__dt * self.cv))], self.VRV2a[l,t-1], self.RSyn_V2a_V2a[self.nV2a*k+l,1], self.RSyn_V2a_V2a[self.nV2a*k+l,2])  #Contralateral

        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                self.LSyn_V2a_MN[self.nMN*k+l,:] = self.L_glusyn_V2a_MN.getNextVal(self.VLV2a[k,t-int(self.LD_V2a_MN[k,l]/(self.__dt * self.cv))], self.VLMN[l,t-1], self.LSyn_V2a_MN[self.nMN*k+l,1], self.LSyn_V2a_MN[self.nMN*k+l,2])  #Contralateral
                self.RSyn_V2a_MN[self.nMN*k+l,:] = self.R_glusyn_V2a_MN.getNextVal(self.VRV2a[k,t-int(self.RD_V2a_MN[k,l]/(self.__dt * self.cv))], self.VRMN[l,t-1], self.RSyn_V2a_MN[self.nMN*k+l,1], self.RSyn_V2a_MN[self.nMN*k+l,2])  #Contralateral

        for k in range (0, self.nV2a):
            for l in range (0, self.ndI6):
                self.LSyn_V2a_dI6[self.ndI6*k+l,:] = self.L_glusyn_V2a_dI6.getNextVal(self.VLV2a[k,t-int(self.LD_V2a_dI6[k,l]/(self.__dt * self.cv))], self.VLdI6[l,t-1], self.LSyn_V2a_dI6[self.ndI6*k+l,1], self.LSyn_V2a_dI6[self.ndI6*k+l,2])  #Contralateral
                self.RSyn_V2a_dI6[self.ndI6*k+l,:] = self.R_glusyn_V2a_dI6.getNextVal(self.VRV2a[k,t-int(self.RD_V2a_dI6[k,l]/(self.__dt * self.cv))], self.VRdI6[l,t-1], self.RSyn_V2a_dI6[self.ndI6*k+l,1], self.RSyn_V2a_dI6[self.ndI6*k+l,2])  #Contralateral

        for k in range (0, self.nV2a):
            for l in range (0, self.nV1):
                self.LSyn_V2a_V1[self.nV1*k+l,:] = self.L_glusyn_V2a_V1.getNextVal(self.VLV2a[k,t-int(self.LD_V2a_V1[k,l]/(self.__dt * self.cv))], self.VLV1[l,t-1], self.LSyn_V2a_V1[self.nV1*k+l,1], self.LSyn_V2a_V1[self.nV1*k+l,2])
                self.RSyn_V2a_V1[self.nV1*k+l,:] = self.R_glusyn_V2a_V1.getNextVal(self.VRV2a[k,t-int(self.RD_V2a_V1[k,l]/(self.__dt * self.cv))], self.VRV1[l,t-1], self.RSyn_V2a_V1[self.nV1*k+l,1], self.RSyn_V2a_V1[self.nV1*k+l,2])

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                self.LSyn_V2a_V0v[self.nV0v*k+l,:] = self.L_glusyn_V2a_V0v.getNextVal(self.VLV2a[k,t-int(self.LD_V2a_V0v[k,l]/(self.__dt * self.cv))], self.VLV0v[l,t-1], self.LSyn_V2a_V0v[self.nV0v*k+l,1], self.LSyn_V2a_V0v[self.nV0v*k+l,2])
                self.RSyn_V2a_V0v[self.nV0v*k+l,:] = self.R_glusyn_V2a_V0v.getNextVal(self.VRV2a[k,t-int(self.RD_V2a_V0v[k,l]/(self.__dt * self.cv))], self.VRV0v[l,t-1], self.RSyn_V2a_V0v[self.nV0v*k+l,1], self.RSyn_V2a_V0v[self.nV0v*k+l,2])

    def calcV1SynOutputs(self, t):
        for k in range (0, self.nV1):
            for l in range (0, self.nMN):
                self.LSyn_V1_MN[self.nMN*k+l,:] = self.L_glysyn_V1_MN.getNextVal(self.VLV1[k,t-int(self.LD_V1_MN[k,l]/(self.__dt * self.cv))], self.VLMN[l,t-1], self.LSyn_V1_MN[self.nMN*k+l,1], self.LSyn_V1_MN[self.nMN*k+l,2])
                self.RSyn_V1_MN[self.nMN*k+l,:] = self.R_glysyn_V1_MN.getNextVal(self.VRV1[k,t-int(self.RD_V1_MN[k,l]/(self.__dt * self.cv))], self.VRMN[l,t-1], self.RSyn_V1_MN[self.nMN*k+l,1], self.RSyn_V1_MN[self.nMN*k+l,2])

        for k in range (0, self.nV1):
            for l in range (0, self.nV2a):
                self.LSyn_V1_V2a[self.nV2a*k+l,:] = self.L_glysyn_V1_V2a.getNextVal(self.VLV1[k,t-int(self.LD_V2a_V1[l,k]/(self.__dt * self.cv))], self.VLV2a[l,t-1], self.LSyn_V1_V2a[self.nV2a*k+l,1], self.LSyn_V1_V2a[self.nV2a*k+l,2])
                self.RSyn_V1_V2a[self.nV2a*k+l,:] = self.R_glysyn_V1_V2a.getNextVal(self.VRV1[k,t-int(self.RD_V2a_V1[l,k]/(self.__dt * self.cv))], self.VRV2a[l,t-1], self.RSyn_V1_V2a[self.nV2a*k+l,1], self.RSyn_V1_V2a[self.nV2a*k+l,2])

        for k in range (0, self.nV1):
            for l in range (0, self.nV0v):
                self.LSyn_V1_V0v[self.nV0v*k+l,:] = self.L_glysyn_V1_V0v.getNextVal(self.VLV1[k,t-int(self.LD_V1_V0v[k,l]/(self.__dt * self.cv))], self.VLV0v[l,t-1], self.LSyn_V1_V0v[self.nV0v*k+l,1], self.LSyn_V1_V0v[self.nV0v*k+l,2])
                self.RSyn_V1_V0v[self.nV0v*k+l,:] = self.R_glysyn_V1_V0v.getNextVal(self.VRV1[k,t-int(self.RD_V1_V0v[k,l]/(self.__dt * self.cv))], self.VRV0v[l,t-1], self.RSyn_V1_V0v[self.nV0v*k+l,1], self.RSyn_V1_V0v[self.nV0v*k+l,2])

        for k in range (0, self.nV1):
            for l in range (0, self.ndI6):
                self.LSyn_V1_dI6[self.ndI6*k+l,:] = self.L_glysyn_V1_dI6.getNextVal(self.VLV1[k,t-int(self.LD_V1_dI6[k,l]/(self.__dt * self.cv))], self.VLdI6[l,t-1], self.LSyn_V1_dI6[self.ndI6*k+l,1], self.LSyn_V1_dI6[self.ndI6*k+l,2])
                self.RSyn_V1_dI6[self.ndI6*k+l,:] = self.R_glysyn_V1_dI6.getNextVal(self.VRV1[k,t-int(self.RD_V1_dI6[k,l]/(self.__dt * self.cv))], self.VRdI6[l,t-1], self.RSyn_V1_dI6[self.ndI6*k+l,1], self.RSyn_V1_dI6[self.ndI6*k+l,2])

    def calcGapJuncOutputs(self, t):
        self.calcMNGapOutputs(t)
        self.calcdI6GapOutputs(t)
        self.calcV0vGapOutputs(t)
        self.calcV2aGapOutputs(t)

    def calcMNGapOutputs(self, t):
        for k in range (0, self.nMN):
            for l in range (0, self.nMN):   
                self.RSGap_MN_MN[k,l] = self.RGap_MN_MN[k,l]*(self.VRMN[k,t-int(self.RD_MN_MN[k,l]/(self.__dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_MN_MN[k,l] = self.LGap_MN_MN[k,l]*(self.VLMN[k,t-int(self.LD_MN_MN[k,l]/(self.__dt * self.cv))]-self.VLMN[l,t-1])

        for k in range (0, self.nMN):
            for l in range (0, self.ndI6):   
                self.RSGap_MN_dI6[k,l] = self.RGap_MN_dI6[k,l]*(self.VRMN[k,t-int(self.RD_MN_dI6[k,l]/(self.__dt * self.cv))]-self.VRdI6[l,t-1])
                self.LSGap_MN_dI6[k,l] = self.LGap_MN_dI6[k,l]*(self.VLMN[k,t-int(self.LD_MN_dI6[k,l]/(self.__dt * self.cv))]-self.VLdI6[l,t-1])
                
        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):   
                self.RSGap_MN_V0v[k,l] = self.RGap_MN_V0v[k,l]*(self.VRMN[k,t-int(self.RD_MN_V0v[k,l]/(self.__dt * self.cv))]-self.VRV0v[l,t-1])
                self.LSGap_MN_V0v[k,l] = self.LGap_MN_V0v[k,l]*(self.VLMN[k,t-int(self.LD_MN_V0v[k,l]/(self.__dt * self.cv))]-self.VLV0v[l,t-1])
                
        for k in range (0, self.nMN):
            for l in range (0, self.nV2a):   
                self.RSGap_MN_V2a[k,l] = self.RGap_V2a_MN[l,k]*(self.VRMN[k,t-int(self.RD_V2a_MN[l,k]/(self.__dt * self.cv))]-self.VRV2a[l,t-1])
                self.LSGap_MN_V2a[k,l] = self.LGap_V2a_MN[l,k]*(self.VLMN[k,t-int(self.LD_V2a_MN[l,k]/(self.__dt * self.cv))]-self.VLV2a[l,t-1])

    def calcdI6GapOutputs(self, t):
        for k in range (0, self.ndI6):
            for l in range (0, self.nMN):   
                self.RSGap_dI6_MN[k,l] = self.RGap_MN_dI6[l,k]*(self.VRdI6[k,t-int(self.RD_MN_dI6[l,k]/(self.__dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_dI6_MN[k,l] = self.LGap_MN_dI6[l,k]*(self.VLdI6[k,t-int(self.LD_MN_dI6[l,k]/(self.__dt * self.cv))]-self.VLMN[l,t-1])
        
        for k in range (0, self.ndI6):
            for l in range (0, self.ndI6):   
                self.RSGap_dI6_dI6[k,l] = self.RGap_dI6_dI6[k,l]*(self.VRdI6[k,t-int(self.RD_dI6_dI6[k,l]/(self.__dt * self.cv))]-self.VRdI6[l,t-1])
                self.LSGap_dI6_dI6[k,l] = self.LGap_dI6_dI6[k,l]*(self.VLdI6[k,t-int(self.LD_dI6_dI6[k,l]/(self.__dt * self.cv))]-self.VLdI6[l,t-1])
        
    def calcV0vGapOutputs(self, t):
        for k in range (0, self.nV0v):
            for l in range (0, self.nMN):   
                self.RSGap_V0v_MN[k,l] = self.RGap_MN_V0v[l,k]*(self.VRV0v[k,t-int(self.RD_MN_V0v[l,k]/(self.__dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_V0v_MN[k,l] = self.LGap_MN_V0v[l,k]*(self.VLV0v[k,t-int(self.LD_MN_V0v[l,k]/(self.__dt * self.cv))]-self.VLMN[l,t-1])
                
        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):   
                self.RSGap_V0v_V0v[k,l] = self.RGap_V0v_V0v[k,l]*(self.VRV0v[k,t-int(self.RD_V0v_V0v[k,l]/(self.__dt * self.cv))]-self.VRV0v[l,t-1])
                self.LSGap_V0v_V0v[k,l] = self.LGap_V0v_V0v[k,l]*(self.VLV0v[k,t-int(self.LD_V0v_V0v[k,l]/(self.__dt * self.cv))]-self.VLV0v[l,t-1])    

    def calcV2aGapOutputs(self, t):
        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):   
                self.RSGap_V2a_V2a[k,l] = self.RGap_V2a_V2a[k,l]*(self.VRV2a[k,t-int(self.RD_V2a_V2a[k,l]/(self.__dt * self.cv))]-self.VRV2a[l,t-1])
                self.LSGap_V2a_V2a[k,l] = self.LGap_V2a_V2a[k,l]*(self.VLV2a[k,t-int(self.LD_V2a_V2a[k,l]/(self.__dt * self.cv))]-self.VLV2a[l,t-1])
                
        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):   
                self.RSGap_V2a_MN[k,l] = self.RGap_V2a_MN[k,l]*(self.VRV2a[k,t-int(self.RD_V2a_MN[k,l]/(self.__dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_V2a_MN[k,l] = self.LGap_V2a_MN[k,l]*(self.VLV2a[k,t-int(self.LD_V2a_MN[k,l]/(self.__dt * self.cv))]-self.VLMN[l,t-1])

    def calcMembranePotentialsFromCurrents(self, t):

        ## Determine membrane potentials from synaptic and external currents
        self.calcMNPotentialsandResidues(t)
        self.calcdI6PotentialsandResidues(t)
        self.calcV0vPotentialsandResidues(t)
        self.calcV2aPotentialsandResidues(t)
        self.calcV1PotentialsandResidues(t)
        self.calcMusclePotentialsandResidues(t)

    def calcMNPotentialsandResidues(self, t):
        for k in range (0, self.nMN):
            if t < (self.__tshutoff): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                ISynLdI6 = sum(self.RSyn_dI6_MN[self.nMN*l+k,0]*self.LW_dI6_MN[l,k]*self.R_str for l in range (0, self.ndI6))
                ISynLV2a = sum(self.LSyn_V2a_MN[self.nMN*m+k,0]*self.LW_V2a_MN[m,k] for m in range (0, self.nV2a))
                ISynLV1 = sum(self.LSyn_V1_MN[self.nMN*p+k,0]*self.LW_V1_MN[p,k]*self.R_str for p in range (0, self.nV1))
                IsynL = ISynLdI6 + ISynLV2a + ISynLV1

                ISynRdI6 = sum(self.LSyn_dI6_MN[self.nMN*l+k,0]*self.RW_dI6_MN[l,k]*self.R_str for l in range (0, self.ndI6))
                ISynRV2a = sum(self.RSyn_V2a_MN[self.nMN*m+k,0]*self.RW_V2a_MN[m,k] for m in range (0, self.nV2a))
                ISynRV1 = sum(self.RSyn_V1_MN[self.nMN*p+k,0]*self.RW_V1_MN[p,k]*self.R_str for p in range (0, self.nV1))
                IsynR = ISynRdI6 + ISynRV2a + ISynRV1
                
            IGapL = - sum(self.LSGap_MN_MN[k,:]) + sum(self.LSGap_MN_MN[:,k]) - sum(self.LSGap_MN_dI6[k,:]) + sum(self.LSGap_dI6_MN[:,k]) - sum(self.LSGap_MN_V0v[k,:]) + sum(self.LSGap_V0v_MN[:,k]) - sum(self.LSGap_MN_V2a[k,:]) + sum(self.LSGap_V2a_MN[:,k])
            self.resLMN[k,:] = self.L_MN[k].getNextVal(self.resLMN[k,0],self.resLMN[k,1], IGapL + IsynL)
            self.VLMN[k,t] = self.resLMN[k,0]

            IGapR = - sum(self.RSGap_MN_MN[k,:]) + sum(self.RSGap_MN_MN[:,k]) - sum(self.RSGap_MN_dI6[k,:]) + sum(self.RSGap_dI6_MN[:,k]) - sum(self.RSGap_MN_V0v[k,:]) + sum(self.RSGap_V0v_MN[:,k]) - sum(self.RSGap_MN_V2a[k,:]) + sum(self.RSGap_V2a_MN[:,k])
            self.resRMN[k,:] = self.R_MN[k].getNextVal(self.resRMN[k,0],self.resRMN[k,1], IGapR + IsynR)
            self.VRMN[k,t] = self.resRMN[k,0]
    
    def calcdI6PotentialsandResidues(self, t):
        for k in range (0, self.ndI6):
            if t < (self.__tshutoff): 
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.LSyn_V2a_dI6[self.ndI6*l+k,0]*self.LW_V2a_dI6[l,k] for l in range (0, self.nV2a)) + sum(self.RSyn_dI6_dI6[self.ndI6*l+k,0]*self.LW_dI6_dI6[l,k]*self.R_str for l in range (0, self.ndI6)) + sum(self.LSyn_V1_dI6[self.ndI6*l+k,0]*self.LW_V1_dI6[l,k]*self.R_str for l in range (0, self.nV1))
                IsynR = sum(self.RSyn_V2a_dI6[self.ndI6*l+k,0]*self.RW_V2a_dI6[l,k] for l in range (0, self.nV2a)) + sum(self.LSyn_dI6_dI6[self.ndI6*l+k,0]*self.RW_dI6_dI6[l,k]*self.R_str for l in range (0, self.ndI6)) + sum(self.RSyn_V1_dI6[self.ndI6*l+k,0]*self.RW_V1_dI6[l,k]*self.R_str for l in range (0, self.nV1))
                
            IGapL = - sum(self.LSGap_dI6_dI6[k,:]) + sum(self.LSGap_dI6_dI6[:,k]) - sum(self.LSGap_dI6_MN[k,:]) + sum(self.LSGap_MN_dI6[:,k])
            self.resLdI6[k,:] = self.L_dI6[k].getNextVal(self.resLdI6[k,0],self.resLdI6[k,1], IGapL + IsynL)
            self.VLdI6[k,t] = self.resLdI6[k,0]
            IGapR = - sum(self.RSGap_dI6_dI6[k,:]) + sum(self.RSGap_dI6_dI6[:,k]) - sum(self.RSGap_dI6_MN[k,:]) + sum(self.RSGap_MN_dI6[:,k])
            self.resRdI6[k,:] = self.R_dI6[k].getNextVal(self.resRdI6[k,0],self.resRdI6[k,1], IGapR + IsynR)
            self.VRdI6[k,t] = self.resRdI6[k,0]
            
    def calcV0vPotentialsandResidues(self, t):
        for k in range (0, self.nV0v):
            if t < (self.__tshutoff): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.LSyn_V2a_V0v[self.nV0v*l+k,0]*self.LW_V2a_V0v[l,k] for l in range (0, self.nV2a)) + sum(self.LSyn_V1_V0v[self.nV0v*l+k,0]*self.LW_V1_V0v[l,k]*self.R_str for l in range (0, self.nV1))
                IsynR = sum(self.RSyn_V2a_V0v[self.nV0v*l+k,0]*self.RW_V2a_V0v[l,k] for l in range (0, self.nV2a)) + sum(self.RSyn_V1_V0v[self.nV0v*l+k,0]*self.RW_V1_V0v[l,k]*self.R_str for l in range (0, self.nV1))
               
            IGapL = - sum(self.LSGap_V0v_V0v[k,:]) + sum(self.LSGap_V0v_V0v[:,k]) -sum(self.LSGap_V0v_MN[k,:])  + sum(self.LSGap_MN_V0v[:,k])
            self.resLV0v[k,:] = self.L_V0v[k].getNextVal(self.resLV0v[k,0],self.resLV0v[k,1],  IGapL + IsynL)
            self.VLV0v[k,t] = self.resLV0v[k,0]
            IGapR = - sum(self.RSGap_V0v_V0v[k,:]) + sum(self.RSGap_V0v_V0v[:,k]) -sum(self.RSGap_V0v_MN[k,:])  + sum(self.RSGap_MN_V0v[:,k])
            self.resRV0v[k,:] = self.R_V0v[k].getNextVal(self.resRV0v[k,0],self.resRV0v[k,1], IGapR + IsynR)
            self.VRV0v[k,t] = self.resRV0v[k,0]
            
    def calcV2aPotentialsandResidues(self, t):
        for k in range (0, self.nV2a):
            if t < (self.__tshutoff): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                ISynLV2a = sum(self.LSyn_V2a_V2a[self.nV2a*m+k,0]*self.LW_V2a_V2a[m,k] for m in range (0, self.nV2a)) 
                ISynLdI6 = sum(self.RSyn_dI6_V2a[self.nV2a*l+k,0]*self.RW_dI6_V2a[l,k]*self.R_str for l in range (0, self.ndI6))
                ISynLV1 = sum(self.LSyn_V1_V2a[self.nV2a*p+k,0]*self.LW_V1_V2a[p,k]*self.R_str for p in range (0, self.nV1))
                ISynLV0v = sum(self.RSyn_V0v_V2a[self.nV2a*l+k,0]*self.LW_V0v_V2a[l,k] for l in range (0, self.nV0v))
                IsynL = ISynLV2a + ISynLdI6 + ISynLV1 + ISynLV0v

                ISynRV2a = sum(self.RSyn_V2a_V2a[self.nV2a*m+k,0]*self.RW_V2a_V2a[m,k] for m in range (0, self.nV2a))
                ISynRdI6 = sum(self.LSyn_dI6_V2a[self.nV2a*l+k,0]*self.LW_dI6_V2a[l,k]*self.R_str for l in range (0, self.ndI6))
                ISynRV1 = sum(self.RSyn_V1_V2a[self.nV2a*p+k,0]*self.RW_V1_V2a[p,k]*self.R_str for p in range (0, self.nV1))
                ISynRV0v = sum(self.LSyn_V0v_V2a[self.nV2a*l+k,0]*self.RW_V0v_V2a[l,k] for l in range (0, self.nV0v))
                IsynR = ISynRV2a + ISynRdI6 + ISynRV1 + ISynRV0v
            if k < 20:
                IsynL = IsynL + gauss(1.0, self.sigma) * self.stim[t-180-32*k] #* (nV2a-k)/nV2a        # Tonic drive for the all V2as # (nV2a-k)/nV2a is to produce a decreasing gradient of descending drive # 32*k repself.resents the conduction delay, which is 1.6 ms according to McDermid and Drapeau JNeurophysiol (2006). Since we consider each somite to be two real somites, then 16*2 
                IsynR = IsynR + gauss(1.0, self.sigma) * self.stim[t-180-32*k] #* (nV2a-k)/nV2a
                
            IGapL = - sum(self.LSGap_V2a_V2a[k,:]) + sum(self.LSGap_V2a_V2a[:,k]) - sum(self.LSGap_V2a_MN[k,:])+ sum(self.LSGap_MN_V2a[:,k])
            self.resLV2a[k,:] = self.L_V2a[k].getNextVal(self.resLV2a[k,0], self.resLV2a[k,1], IGapL + IsynL)         
            self.VLV2a[k,t] = self.resLV2a[k,0]
            
            IGapR = - sum(self.RSGap_V2a_V2a[k,:]) + sum(self.RSGap_V2a_V2a[:,k]) - sum(self.RSGap_V2a_MN[k,:])+ sum(self.RSGap_MN_V2a[:,k])
            self.resRV2a[k,:] = self.R_V2a[k].getNextVal(self.resRV2a[k,0], self.resRV2a[k,1], IGapR + IsynR)    
            self.VRV2a[k,t] = self.resRV2a[k,0]
        
    def calcV1PotentialsandResidues(self, t):
        for k in range (0, self.nV1):
            if t < self.__tshutoff: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.LSyn_V2a_V1[self.nV1*m+k,0] * self.LW_V2a_V1[m,k] for m in range (0, self.nV2a))
                IsynR = sum(self.RSyn_V2a_V1[self.nV1*m+k,0] * self.RW_V2a_V1[m,k] for m in range (0, self.nV2a))
            self.resLV1[k,:] = self.L_V1[k].getNextVal(self.resLV1[k,0], self.resLV1[k,1], IsynL)  
            self.VLV1[k,t] = self.resLV1[k,0]
            self.resRV1[k,:] = self.R_V1[k].getNextVal(self.resRV1[k,0], self.resRV1[k,1], IsynR)  
            self.VRV1[k,t] = self.resRV1[k,0]
            
    def calcMusclePotentialsandResidues(self, t):
        for k in range (0, self.nMuscle):
            if t < (self.__tshutoff): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.LSyn_MN_Muscle[self.nMuscle*l+k,0]*self.LW_MN_Muscle[l,k] for l in range (0, self.nMN))
                IsynR = sum(self.RSyn_MN_Muscle[self.nMuscle*l+k,0]*self.RW_MN_Muscle[l,k] for l in range (0, self.nMN))
                
            self.resLMuscle[k,:] = self.L_Muscle[k].getNextVal(self.resLMuscle[k,0], IsynL + 0.4*gauss(1.0, self.sigma) - 0.4) #the last term is to add variability but equals 0 if sigma = 0
            self.VLMuscle[k,t] = self.resLMuscle[k,0]
            
            self.resRMuscle[k,:] = self.R_Muscle[k].getNextVal(self.resRMuscle[k,0], IsynR + 0.4*gauss(1.0, self.sigma) - 0.4) #the last term is to add variability but equals 0 if sigma = 0
            self.VRMuscle[k,t] = self.resRMuscle[k,0]

    # fills in the parameters dictionary to write to a JSON file
    def getParametersDict(self):
        if (len(self.paramDict) > 0):
            return self.paramDict
        self.paramDict["dt"] = self.__dt
        self.paramDict["tshutoff"] = self.__tshutoff
        #self.paramDict["tasyncdelay"] = self.tasyncdelay
        self.paramDict["stim0"] = self.stim0
        self.paramDict["sigma"] = self.sigma
        self.paramDict["sigma_LR"] = self.sigma_LR
        self.paramDict["E_glu"] = self.E_glu
        self.paramDict["E_gly"] = self.E_gly
        self.paramDict["cv"] = self.cv
        self.paramDict["nMN"] = self.nMN
        self.paramDict["ndI6"] = self.ndI6
        self.paramDict["nV0v"] = self.nV0v
        self.paramDict["nV2a"] = self.nV2a
        self.paramDict["nV1"] = self.nV1
        self.paramDict["nMuscle"] = self.nMuscle

        self.paramDict["MN_MN_gap_weight"] = self.MN_MN_gap_weight
        self.paramDict["dI6_dI6_gap_weight"] = self.dI6_dI6_gap_weight
        self.paramDict["V0v_V0v_gap_weight"] = self.V0v_V0v_gap_weight
        self.paramDict["V2a_V2a_gap_weight"] = self.V2a_V2a_gap_weight
        self.paramDict["V2a_MN_gap_weight"] = self.V2a_MN_gap_weight
        self.paramDict["MN_dI6_gap_weight"] = self.MN_dI6_gap_weight
        self.paramDict["MN_V0v_gap_weight"] = self.MN_V0v_gap_weight
        self.paramDict["V2a_V2a_syn_weight"] = self.V2a_V2a_syn_weight
        self.paramDict["V2a_MN_syn_weight"] = self.V2a_MN_syn_weight
        self.paramDict["V2a_dI6_syn_weight"] = self.V2a_dI6_syn_weight
        self.paramDict["V2a_V1_syn_weight"] = self.V2a_V1_syn_weight
        self.paramDict["V2a_V0v_syn_weight"] = self.V2a_V0v_syn_weight
        self.paramDict["V1_MN_syn_weight"] = self.V1_MN_syn_weight
        self.paramDict["V1_V2a_syn_weight"] = self.V1_V2a_syn_weight
        self.paramDict["V1_V0v_syn_weight"] = self.V1_V0v_syn_weight
        self.paramDict["V1_dI6_syn_weight"] = self.V1_dI6_syn_weight
        self.paramDict["dI6_MN_syn_weight"] = self.dI6_MN_syn_weight
        self.paramDict["dI6_V2a_syn_weight"] = self.dI6_V2a_syn_weight
        self.paramDict["dI6_dI6_syn_weight"] = self.dI6_dI6_syn_weight
        self.paramDict["V0v_V2a_syn_weight"] = self.V0v_V2a_syn_weight
        self.paramDict["MN_Muscle_syn_weight"] = self.MN_Muscle_syn_weight

        self.paramDict["rangeMin"] = self.rangeMin
        self.paramDict["rangeMN_MN"] = self.rangeMN_MN
        self.paramDict["rangedI6_dI6"] = self.rangedI6_dI6
        self.paramDict["rangeV0v_V0v"] = self.rangeV0v_V0v
        self.paramDict["rangeV2a_V2a_gap"] = self.rangeV2a_V2a_gap
        self.paramDict["rangeV2a_MN"] = self.rangeV2a_MN
        self.paramDict["rangeMN_dI6"] = self.rangeMN_dI6
        self.paramDict["rangeMN_V0v"] = self.rangeMN_V0v
        self.paramDict["rangeV2a_V2a_syn"] = self.rangeV2a_V2a_syn
        self.paramDict["rangeV2a_MN_asc"] = self.rangeV2a_MN_asc
        self.paramDict["rangeV2a_MN_desc"] = self.rangeV2a_MN_desc
        self.paramDict["rangeV2a_dI6"] = self.rangeV2a_dI6
        self.paramDict["rangeV2a_V1"] = self.rangeV2a_V1
        self.paramDict["rangeV2a_V0v_asc"] = self.rangeV2a_V0v_asc
        self.paramDict["rangeV2a_V0v_desc"] = self.rangeV2a_V0v_desc
        self.paramDict["rangeV1_MN"] = self.rangeV1_MN
        self.paramDict["rangeV1_V2a"] = self.rangeV1_V2a
        self.paramDict["rangeV1_V0v"] = self.rangeV1_V0v
        self.paramDict["rangeV1_dI6"] = self.rangeV1_dI6
        self.paramDict["rangedI6_MN_desc"] = self.rangedI6_MN_desc
        self.paramDict["rangedI6_MN_asc"] = self.rangedI6_MN_asc
        self.paramDict["rangedI6_V2a_desc"] = self.rangedI6_V2a_desc
        self.paramDict["rangedI6_V2a_asc"] = self.rangedI6_V2a_asc
        self.paramDict["rangedI6_dI6_desc"] = self.rangedI6_dI6_desc
        self.paramDict["rangedI6_dI6_asc"] = self.rangedI6_dI6_asc
        self.paramDict["rangeV0v_V2a_desc"] = self.rangeV0v_V2a_desc
        self.paramDict["rangeV0v_V2a_asc"] = self.rangeV0v_V2a_asc
        self.paramDict["rangeMN_Muscle"] = self.rangeMN_Muscle

        return self.paramDict

    def printParameters(self):
        for k in self.getParametersDict().keys():
            print(k + ': ' + str(self.paramDict[k]))

    #plots the membrane potentials between tstart(by default 0) and tend
    #if plotall is set to True, all cell membrane potentials are plotted
    #   otherwise, only motoneuron membrane potentials are plotted
    def plotProgress(self, tend, tstart = 0, plotall = False):
        leftValues = dict()
        rightValues = dict()
        leftValues['MN'] = self.VLMN
        rightValues['MN'] = self.VRMN
        if plotall:
            leftValues['dI6'] = self.VLdI6
            rightValues['dI6'] = self.VRdI6
            leftValues['V0v'] = self.VLV0v
            rightValues['V0v'] = self.VRV0v
            leftValues['V2a'] = self.VLV2a
            rightValues['V2a'] = self.VRV2a
            leftValues['V1'] = self.VLV1
            rightValues['V1'] = self.VRV1
            leftValues['Muscle'] = self.VLMuscle
            rightValues['Muscle'] = self.VRMuscle

        #call the plotProgress function from util.py
        plotProgress(tstart, tend, self.Time, leftValues, rightValues, onSamePlot=False, colorMode=0, height=2.5)
    
    # Save the data points for the membrane potentials to a CSV file
    # and the parameters to a JSON file of the same name
    def saveToFile(self, filename = None):
        
        if not self.model_run:
            return
        filenamejson = filename
        if filename is None: #if filename is not provided, the class name is used
            filename = type(self).__name__ + ".csv"
            filenamejson =  type(self).__name__ + ".json"
        elif not str(filename).endswith(".csv"):
            filenamejson =  filename + ".json"
            filename += ".csv"
        else:
            filenamejson = replace(filename, ".csv", ".json")

        LeftValues = {'MN': self.VLMN,
            'dI6': self.VLdI6,
            'V0v': self.VLV0v,
            'V2a': self.VLV2a,
            'V1': self.VLV1,
            'Muscle': self.VLMuscle}
        RightValues = {'MN': self.VRMN,
            'dI6': self.VRdI6,
            'V0v': self.VRV0v,
            'V2a': self.VRV2a,
            'V1': self.VRV1,
            'Muscle': self.VRMuscle}

        saveToCSV(filename = filename, Time = self.Time, LeftValues = LeftValues, RightValues = RightValues)
        saveToJSON(filenamejson, self.getParametersDict())

    def saveAnimation(self, filename = None):
        
        if not self.model_run:
            return
        
        if filename is None:
            filename = type(self).__name__ + ".mp4"
        elif not str(filename).endswith(".mp4"):
            filename += ".mp4"

        saveAnimation(filename = filename, nMuscle = self.nMuscle, VLMuscle = self.VLMuscle, VRMuscle = self.VRMuscle, Time = self.Time, dt = self.__dt)

    def stripOffsetRegion(self, tskip):
        ## Removing the first "tskip" timepoints to let the initial conditions dissipate
        index_offset = tskip

        #the removed data are saved in a dictionary to prevent data loss - in case they will be used in a different analysis
        self.skippedVLists['VLMN'] = self.VLMN[:,:index_offset]
        self.skippedVLists['VRMN'] = self.VRMN[:,:index_offset]
        self.skippedVLists['VLdI6'] = self.VLdI6[:,:index_offset]
        self.skippedVLists['VRdI6'] = self.VRdI6[:,:index_offset]
        self.skippedVLists['VLV0v'] = self.VLV0v[:,:index_offset]
        self.skippedVLists['VRV0v'] = self.VRV0v[:,:index_offset]
        self.skippedVLists['VLV2a'] = self.VLV2a[:,:index_offset]
        self.skippedVLists['VRV2a'] = self.VRV2a[:,:index_offset]
        self.skippedVLists['VLV1'] = self.VLV1[:,:index_offset]
        self.skippedVLists['VRV1'] = self.VRV1[:,:index_offset]
        self.skippedVLists['VLMuscle'] = self.VLMuscle[:,:index_offset]
        self.skippedVLists['VRMuscle'] = self.VRMuscle[:,:index_offset]

        self.VLMN = self.VLMN[:,index_offset:]
        self.VRMN = self.VRMN[:,index_offset:]
        
        self.VLdI6 = self.VLdI6[:,index_offset:]
        self.VRdI6 = self.VRdI6[:,index_offset:]
        
        self.VLV0v = self.VLV0v[:,index_offset:]
        self.VRV0v = self.VRV0v[:,index_offset:]
        
        self.VLV2a = self.VLV2a[:,index_offset:]
        self.VRV2a = self.VRV2a[:,index_offset:]
        
        self.VLV1 = self.VLV1[:,index_offset:]
        self.VRV1 = self.VRV1[:,index_offset:]
        
        self.VLMuscle = self.VLMuscle[:,index_offset:]
        self.VRMuscle = self.VRMuscle[:,index_offset:]
        
        self.Time = self.Time[index_offset:] - self.Time[index_offset:][0]


    # tmax_ms: the time period the simulation will be run for
    # tshutoff_ms: Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
    # tskip_ms: the initial time period that is added to the simulation to cover the initial conditions to dissipate
    #   tshutoff period is included in the skipped region
    # dt is the time step size
    def setTimeParameters(self, tmax_ms = 10000, tshutoff_ms = 50, tskip_ms = 1000, dt = 0.1):
        self.__dt = dt
        self.__tmax = (int)(tmax_ms / dt)
        self.__tshutoff = (int)(tshutoff_ms / dt)
        self.__tskip = (int)(tskip_ms / dt)

    def gettShutOff(self):
        return self.__tshutoff
    
    def getdt(self):
        return self.__dt


    # This function sets up the connectome and runs a simulation.
    # rand is a seed for a random function
    # tplot_interval: the time period that a plot of membrane potential(s) will be displayed. These plots will include the tskip region.
    # plotProgressOnly: a flag that shows whether at each tplot_interval, the plot will display the potential change from the zero point, or since the end of the previous plot.
    #   Having this parameter True will make it easier to compare each period to each other. Caution: x-axis scales will be the same, but y-axis scales will not.
    # plotAllPotentials: whether the plotting should plot only MN membrane potentials, or all cells (IC, V0a, etc)
    # printParam: a flag that will display the parameters used in the run.
    # plotResult: a flag to display the final plot from 0 to tmax (after the tskip region is removed)
    # saveCSV: if True, the membrane potentials of all of the cells will be saved as a CSV file, and parameters will be saved as a JSON file. 
    #   The file name will be the same as the class name.
    # saveAnim: if True, an animation of the muscle movements will be saved as an mp4 file.
    #   The file name will be the same as the class name.
    def mainLoop(self, rand, tplot_interval_ms = 500, plotProgressOnly = False, plotAllPotentials = False, printParam = False, 
            plotResult = False, saveCSV = False, saveAnim = False):

        self.paramDict = dict() #initialize paramDict, will be filled only once per run
        if printParam:
            self.printParameters()

        seed(rand)
        nmax = self.__tmax + self.__tskip
        self.initializeStructures(nmax)

        self.stim = zeros(nmax)
        self.computeSynapticAndGapWeights()
        self.initializeMembranePotentials()

        ## This loop is the main loop where we solve the ordinary differential equations at every time point
        tlastplot = 0
        for t in range (0, nmax):
            self.Time[t] = self.__dt * t

            # Generate plots to visualize the progself.ress of the simulations
            if not(self.Time[t] % tplot_interval_ms) and (self.Time[t] > 20):
                self.plotProgress(t, tstart = tlastplot, plotall = plotAllPotentials)
                tlastplot = t if plotProgressOnly else 0

            self.stim[t] = self.getStimulus(t)
                
            ## Calculate synaptic inputs
            self.calcSynapticOutputs(t)
            self.calcGapJuncOutputs(t)

            self.calcMembranePotentialsFromCurrents(t)
                
        self.stripOffsetRegion(self.__tskip)
        
        self.model_run = True

        if plotResult:
            self.plotProgress(self.__tmax - 1, tstart = 0, plotall = plotAllPotentials)
        if saveCSV:
            self.saveToFile()
        if saveAnim:
            self.saveAnimation()

        return (self.VLMN, self.VRMN), (self.VLdI6, self.VRdI6), (self.VLV0v, self.VRV0v), \
            (self.VLV2a, self.VRV2a), (self.VLV1, self.VRV1), (self.VLMuscle, self.VRMuscle), self.Time

    #endregion                    

