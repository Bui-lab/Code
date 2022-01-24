#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

import matplotlib
import numpy as np
from numpy import mat, zeros
from os import replace
from pylab import plt
from random import seed, gauss

#The definition of single cell models are found based on Izhikevich models
from Izhikevich_class import Izhikevich_9P, TwoExp_syn, Distance, Leaky_Integrator
from Util import saveToCSV, saveAnimation, saveToJSON, plotProgress

class Double_coil_base:

    #%% Class Variables
    #region Class Variables

    #region General Parameters
    model_run = False #to check whether the model has run successfuly - used for plotting and saving functions
    dt = 0.1
    #the currents are shut off to wait for the initial conditions to subside
    tshutoff = 50 #in ms;
    tasyncdelay = 30000 
    #endregion

    #region Other constants
    stim0 = 8
    sigma_gap = 0
    sigma_chem = 0
    E_glu = 0
    E_gly = -70
    cv = 0.55
    #endregion

    #region Number of cells
    nIC = 5
    nMN = 10
    nV0d = 10
    nV0v = 10
    nV2a = 10
    nMuscle = 10
    #endregion

    #region Weight Parameters

    IC_IC_gap_weight = 0.0001
    IC_MN_gap_weight = 0.05
    IC_V0d_gap_weight = 0.05
    IC_V0v_gap_weight = 0.00005
    IC_V2a_gap_weight = 0.15

    MN_MN_gap_weight =  0.07
    MN_V0d_gap_weight = 0.0001
    MN_V0v_gap_weight = 0.0001
    MN_Muscle_syn_weight = 0.02

    V0d_MN_syn_weight = 0.3
    V0d_IC_syn_weight = 0.3
    V0d_V0d_gap_weight = 0.05
    V0d_V2a_syn_weight = 0.3

    V0v_IC_syn_weight = 0.24
    V0v_V0v_gap_weight = 0.044 

    V2a_MN_gap_weight = 0.005
    V2a_V0v_syn_weight = 0.0375 
    V2a_V2a_gap_weight = 0.005

    #endregion

    #region Neurons
    L_IC = []
    R_IC = []
    L_MN = []
    R_MN = []
    L_V0d = []
    R_V0d = []
    L_V0v = []
    R_V0v = []
    L_V2a = []
    R_V2a = []
    L_Muscle = []
    R_Muscle = []
    #endregion

    #region Synapses
    #ACh synapse from MN to Muscle cell
    L_achsyn_MN_Muscle = None
    R_achsyn_MN_Muscle = None

    #Glycinergic synapses from V0d to MN, IC and V2a's
    L_glysyn_V0d_MN = None
    R_glysyn_V0d_MN = None
    L_glysyn_V0d_IC = None
    R_glysyn_V0d_IC = None
    L_glysyn_V0d_V2a = None
    R_glysyn_V0d_V2a = None

    #Glutamatergic synapses from V0v to IC, from V2a to V0v
    L_glusyn_V0v_IC = None
    R_glusyn_V0v_IC = None
    L_glusyn_V2a_V0v = None
    R_glusyn_V2a_V0v = None
    #endregion

    #region Lists of Membrane Potentials
    VLIC = []
    VRIC = []
    VLMN = []
    VRMN = []
    VLV0d = []
    VRV0d = []
    VLV0v = []
    VRV0v = []
    VLV2a = []
    VRV2a = []
    VLMuscle =[]
    VRMuscle = []
    skippedVLists = dict()
    #endregion

    #region Lists of Synaptic and Gap Junction Currents
    LSyn_MN_Muscle = []
    RSyn_MN_Muscle = []

    LSyn_V0d_MN = []
    RSyn_V0d_MN = []
    LSyn_V0d_IC = []
    RSyn_V0d_IC = []
    LSyn_V0d_V2a = []
    RSyn_V0d_V2a = []

    LSyn_V2a_V0v = []
    RSyn_V2a_V0v = []
    LSyn_V0v_IC = []
    RSyn_V0v_IC = []

    LSGap_IC_IC = []
    LSGap_IC_MN = []
    LSGap_IC_V0d = []
    LSGap_IC_V0v = []
    LSGap_IC_V2a = []
    LSGap_MN_IC = []
    LSGap_MN_MN = []
    LSGap_MN_V0d = []
    LSGap_MN_V0v = []
    LSGap_MN_V2a = []
    LSGap_V0d_IC = []
    LSGap_V0d_MN = []
    LSGap_V0d_V0d = []
    LSGap_V0v_IC = []
    LSGap_V0v_MN = []
    LSGap_V0v_V0v = []
    LSGap_V2a_IC = []
    LSGap_V2a_V2a = []
    LSGap_V2a_MN = []

    RSGap_IC_IC = []
    RSGap_IC_MN = []
    RSGap_IC_V0d = []
    RSGap_IC_V0v = []
    RSGap_IC_V2a = []
    RSGap_MN_IC = []
    RSGap_MN_MN = []
    RSGap_MN_V0d = []
    RSGap_MN_V0v = []
    RSGap_MN_V2a = []
    RSGap_V0d_IC = []
    RSGap_V0d_MN = []
    RSGap_V0d_V0d = []
    RSGap_V0v_IC = []
    RSGap_V0v_MN = []
    RSGap_V0v_V0v = []
    RSGap_V2a_IC = []
    RSGap_V2a_V2a = []
    RSGap_V2a_MN = []
    #endregion

    #region Lists of Synaptic and Gap Junction Weights
    LW_V0d_MN = []
    RW_V0d_MN = []
    LW_V0d_IC = []
    RW_V0d_IC = []
    LW_MN_Muscle = []
    RW_MN_Muscle = []
    LW_V0d_V2a = []
    RW_V0d_V2a = []
    LW_V0v_IC = []
    RW_V0v_IC = []
    LW_V2a_V0v = []
    RW_V2a_V0v = []

    LGap_IC_MN = []
    LGap_IC_V0d = []
    LGap_IC_IC = []
    LGap_IC_V0v = []
    LGap_IC_V2a = []

    LGap_MN_MN = []
    LGap_MN_V0d = []
    LGap_MN_V0v = []

    LGap_V0d_V0d = []

    LGap_V0v_V0v = []

    LGap_V2a_V2a = []
    LGap_V2a_MN = []

    RGap_IC_MN = []
    RGap_IC_V0d = []
    RGap_IC_IC = []
    RGap_IC_V0v = []
    RGap_IC_V2a = []

    RGap_MN_MN = []
    RGap_MN_V0d = []
    RGap_MN_V0v = []

    RGap_V0d_V0d = []

    RGap_V0v_V0v = []

    RGap_V2a_V2a = []
    RGap_V2a_MN = []
    #endregion

    #region Lists of residuals for solving membrane potentials
    resLIC = []
    resRIC = []
    resLMN = []
    resRMN = []
    resLV0d = []
    resRV0d = []
    resLV0v = []
    resRV0v = []
    resLV2a = []
    resRV2a = []

    resLMuscle = []
    resRMuscle = []
    #endregion

    #region Distance Matrices
    LD_IC_MN = []
    LD_IC_V0d = []
    LD_IC_V0v = []
    LD_IC_V2a = []

    LD_MN_MN = []
    LD_MN_V0d = []
    LD_MN_V0v = []

    LD_V0d_V0d = []
    LD_V0d_IC = []
    LD_V0d_MN = []
    LD_V0d_V2a = []

    LD_V0v_IC = []
    LD_V0v_V0v = []

    LD_V2a_V2a = []
    LD_V2a_MN = []
    LD_V2a_V0v = []

    RD_IC_MN = []
    RD_IC_V0d = []
    RD_IC_V0v = []
    RD_IC_V2a = []

    RD_MN_MN = []
    RD_MN_V0d = []
    RD_MN_V0v = []

    RD_V0d_V0d = []
    RD_V0d_IC = []
    RD_V0d_MN = []
    RD_V0d_V2a = []

    RD_V0v_IC = []
    RD_V0v_V0v = []

    RD_V2a_V2a = []
    RD_V2a_MN = []
    RD_V2a_V0v = []
    #endregion

    #region Upper range parameters for neuron to neuron/muscle cell reach
    rangeMin = 0.2 # lower range value
    rangeIC_MN = 10
    rangeIC_V0d = 10
    rangeIC_V0v = 10
    rangeIC_V2a = 10
    rangeMN_MN = 6.5
    rangeMN_V0d = 1.5
    rangeMN_V0v = 1.5
    rangeV0d_IC = 20
    rangeV0d_MN = 8
    rangeV0d_V0d = 3.5
    rangeV0v_V0v = 3.5
    rangeV0d_V2a = 8
    rangeV0v_IC_min = 6
    rangeV0v_IC_max = 20
    rangeV2a_MN = 3.5
    rangeV2a_V0v_asc = 4
    rangeV2a_V0v_desc = 10
    rangeV2a_V2a = 3.5
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
    # stim0 is a constant for scaling the drive to IC neurons
    # sigma is a variance for gaussian randomization of the
    #       gap junction coupling and synaptic weights - sigma_gap and sigma_chem are set to the same value in the base model
    # dt is the time step size
    # E_glu and E_gly are the reversal potential of glutamate and glycine respectively
    # cv is the transmission speed
    # nIC, nMN, nV2a, nV0d, nV0v and nMuscle is the number of IC, MN, V2a, V0d, V0v and Muscle cells
    def __init__ (self, dt = 0.1, stim0 = 8, sigma = 0, E_glu = 0, E_gly = -70,
                  cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nV0v = 10, nV2a = 10, nMuscle = 10):
        self.dt = dt

        self.stim0 = stim0
        self.sigma_chem = sigma
        self.sigma_gap = sigma

        self.E_glu = E_glu
        self.E_gly = E_gly
        self.cv = cv
        self.nIC = nIC
        self.nMN = nMN
        self.nV0d = nV0d
        self.nV0v = nV0v
        self.nV2a = nV2a
        self.nMuscle = nMuscle

    def setWeightParameters(self, IC_IC_gap_weight = None, IC_MN_gap_weight = None,
                    IC_V0d_gap_weight = None, IC_V0v_gap_weight = None, IC_V2a_gap_weight = None,
                    MN_MN_gap_weight =  None, MN_V0d_gap_weight = None, MN_V0v_gap_weight = None,  MN_Muscle_syn_weight = None,
                    V0d_IC_syn_weight = None, V0d_MN_syn_weight = None, V0d_V0d_gap_weight = None, V0d_V2a_syn_weight = None,
                    V0v_IC_syn_weight = None, V0v_V0v_gap_weight = None,
                    V2a_MN_gap_weight = None, V2a_V0v_syn_weight = None, V2a_V2a_gap_weight = None):
        if IC_IC_gap_weight is not None:
            self.IC_IC_gap_weight = IC_IC_gap_weight
        if IC_MN_gap_weight is not None:
            self.IC_MN_gap_weight = IC_MN_gap_weight
        if IC_V0d_gap_weight is not None:
            self.IC_V0d_gap_weight = IC_V0d_gap_weight
        if IC_V0v_gap_weight is not None:
            self.IC_V0v_gap_weight = IC_V0v_gap_weight
        if IC_V2a_gap_weight is not None:
            self.IC_V2a_gap_weight = IC_V2a_gap_weight
        if MN_MN_gap_weight is not None:
            self.MN_MN_gap_weight = MN_MN_gap_weight
        if MN_V0d_gap_weight is not None:
            self.MN_V0d_gap_weight = MN_V0d_gap_weight
        if MN_V0v_gap_weight is not None:
            self.MN_V0v_gap_weight = MN_V0v_gap_weight
        if MN_Muscle_syn_weight is not None:
            self.MN_Muscle_syn_weight = MN_Muscle_syn_weight
        if V0d_IC_syn_weight is not None:
            self.V0d_IC_syn_weight = V0d_IC_syn_weight
        if V0d_MN_syn_weight is not None:
            self.V0d_MN_syn_weight = V0d_MN_syn_weight
        if V0d_V0d_gap_weight is not None:
            self.V0d_V0d_gap_weight = V0d_V0d_gap_weight
        if V0d_V2a_syn_weight is not None:
            self.V0d_V2a_syn_weight = V0d_V2a_syn_weight
        if V0v_IC_syn_weight is not None:
            self.V0v_IC_syn_weight = V0v_IC_syn_weight
        if V0v_V0v_gap_weight is not None:
            self.V0v_V0v_gap_weight = V0v_V0v_gap_weight
        if V2a_MN_gap_weight is not None:
            self.V2a_MN_gap_weight = V2a_MN_gap_weight
        if V2a_V0v_syn_weight is not None:
            self.V2a_V0v_syn_weight = V2a_V0v_syn_weight
        if V2a_V2a_gap_weight is not None:
            self.V2a_V2a_gap_weight = V2a_V2a_gap_weight

    def setRangeParameters(self, rangeMin = None, rangeIC_MN = None, rangeIC_V0d = None, rangeIC_V0v = None, rangeIC_V2a = None,
                            rangeMN_MN = None, rangeMN_V0d = None, rangeMN_V0v = None, rangeMN_Muscle = None,
                            rangeV0d_IC = None, rangeV0d_MN = None, rangeV0d_V0d = None, rangeV0d_V2a = None,
                            rangeV0v_IC_min = None, rangeV0v_IC_max = None, rangeV0v_V0v = None,
                            rangeV2a_MN = None, rangeV2a_V0v_asc = None, rangeV2a_V0v_desc = None, rangeV2a_V2a = None):
        if rangeMin is not None:
            self.rangeMin = rangeMin
        if rangeIC_MN is not None:
            self.rangeIC_MN = rangeIC_MN
        if rangeIC_V0d is not None:
            self.rangeIC_V0d = rangeIC_V0d
        if rangeIC_V0v is not None:
            self.rangeIC_V0v = rangeIC_V0v
        if rangeIC_V2a is not None:
            self.rangeIC_V2a = rangeIC_V2a

        if rangeMN_MN is not None:
            self.rangeMN_MN = rangeMN_MN
        if rangeMN_V0d is not None:
            self.rangeMN_V0d = rangeMN_V0d
        if rangeMN_V0v is not None:
            self.rangeMN_V0v = rangeMN_V0v
        if rangeMN_Muscle is not None:
            self.rangeMN_Muscle = rangeMN_Muscle

        if rangeV0d_IC is not None:
            self.rangeV0d_IC = rangeV0d_IC
        if rangeV0d_MN is not None:
            self.rangeV0d_MN = rangeV0d_MN
        if rangeV0d_V0d is not None:
            self.rangeV0d_V0d = rangeV0d_V0d
        if rangeV0d_V2a is not None:
            self.rangeV0d_V2a = rangeV0d_V2a

        if rangeV0v_IC_min is not None:
            self.rangeV0v_IC_min = rangeV0v_IC_min
        if rangeV0v_IC_max is not None:
            self.rangeV0v_IC_max = rangeV0v_IC_max
        if rangeV0v_V0v is not None:
            self.rangeV0v_V0v = rangeV0v_V0v

        if rangeV2a_MN is not None:
            self.rangeV2a_MN = rangeV2a_MN
        if rangeV2a_V0v_asc is not None:
            self.rangeV2a_V0v_asc = rangeV2a_V0v_asc
        if rangeV2a_V0v_desc is not None:
            self.rangeV2a_V0v_desc = rangeV2a_V0v_desc
        if rangeV2a_V2a is not None:
            self.rangeV2a_V2a = rangeV2a_V2a

    #Functions called from the main loop
    def initializeStructures(self, nmax):
        self.initNeurons()
        self.initSynapses()
        self.initStorageLists(nmax)
        self.initAndSetDistanceMatrices()

    def initNeurons(self):
        ## Declare Neuron Types

        self.L_IC = [ Izhikevich_9P(a=0.0002,b=0.5,c=-40, d=5, vmax=0, vr=-60, vt=-45, k=0.3, Cm = 50, dt = self.dt, x=1.0,y=-1) for i in range(self.nIC)]
        self.R_IC = [ Izhikevich_9P(a=0.0002,b=0.5,c=-40, d=5, vmax=0, vr=-60, vt=-45, k=0.3, Cm = 50, dt = self.dt, x=1.0,y=1) for i in range(self.nIC)]

        self.L_MN = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-50, k=0.05, Cm = 20, dt = self.dt, x=5.0+1.6*i,y=-1) for i in range(self.nMN)]
        self.R_MN = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-50, k=0.05, Cm = 20, dt = self.dt, x=5.0+1.6*i,y=1) for i in range(self.nMN)]

        self.L_V0d = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=3.75, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt = self.dt, x=5.0+1.6*i,y=-1) for i in range(self.nV0d)]
        self.R_V0d = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=3.75, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt = self.dt, x=5.0+1.6*i,y=1) for i in range(self.nV0d)]

        self.L_V0v = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=11.6, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt = self.dt, x=5.1+1.6*i,y=-1) for i in range(self.nV0v)]
        self.R_V0v = [ Izhikevich_9P(a=0.02,b=0.1,c=-30, d=11.6, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt = self.dt, x=5.1+1.6*i,y=1) for i in range(self.nV0v)]

        self.L_V2a = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt = self.dt, x=5.1+1.6*i,y=-1) for i in range(self.nV2a)]
        self.R_V2a = [ Izhikevich_9P(a=0.5,b=0.1,c=-50, d=100, vmax=10, vr=-60, vt=-45, k=0.05, Cm = 20, dt = self.dt, x=5.1+1.6*i,y=1) for i in range(self.nV2a)]

        self.L_Muscle = [ Leaky_Integrator(50.0, 3.0, self.dt, 5.0+1.6*i,-1) for i in range(self.nMuscle)]
        self.R_Muscle = [ Leaky_Integrator(50.0, 3.0, self.dt, 5.0+1.6*i, 1) for i in range(self.nMuscle)]

    def initSynapses(self):
        ## Declare Synapses

        # Below is the declaration of the chemical synapses
        # As all synapses from one cell group to another share the same settings, and no state variables,
        #   one synapse class is created for cell group pair

        self.L_achsyn_MN_Muscle = TwoExp_syn(0.5, 1.0, -15, self.dt, 120)
        self.R_achsyn_MN_Muscle = TwoExp_syn(0.5, 1.0, -15, self.dt, 120)
        self.L_glysyn_V0d_MN = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_gly)
        self.R_glysyn_V0d_MN = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_gly)
        self.L_glysyn_V0d_IC = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_gly)
        self.R_glysyn_V0d_IC = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_gly)
        self.L_glysyn_V0d_V2a = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_gly)
        self.R_glysyn_V0d_V2a = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_gly)

        self.L_glusyn_V2a_V0v = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_glu)
        self.R_glusyn_V2a_V0v = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_glu)

        self.L_glusyn_V0v_IC = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_glu)
        self.R_glusyn_V0v_IC = TwoExp_syn(0.5, 1.0, -15, self.dt, self.E_glu)

    def initStorageLists(self, nmax):
        ## Declare Storage tables
        self.Time = zeros(nmax)
        self.VLIC =zeros((self.nIC, nmax))
        self.VRIC =zeros((self.nIC, nmax))
        self.VLMN =zeros((self.nMN, nmax))
        self.VRMN =zeros((self.nMN, nmax))
        self.VLV0d = zeros ((self.nV0d,nmax))
        self.VRV0d = zeros ((self.nV0d,nmax))
        self.VLV0v = zeros ((self.nV0v,nmax))
        self.VRV0v = zeros ((self.nV0v,nmax))
        self.VLV2a = zeros ((self.nV2a,nmax))
        self.VRV2a = zeros ((self.nV2a,nmax))
        self.VLMuscle = zeros((self.nMuscle, nmax))
        self.VRMuscle = zeros((self.nMuscle, nmax))

        #storing synaptic currents

        #gly
        self.LSyn_V0d_MN = zeros((self.nV0d * self.nMN,3))
        self.RSyn_V0d_MN = zeros((self.nV0d * self.nMN,3))
        self.LSyn_V0d_IC = zeros((self.nV0d * self.nIC,3))
        self.RSyn_V0d_IC = zeros((self.nV0d * self.nIC,3))
        self.LSyn_V0d_V2a = zeros((self.nV0d * self.nV2a,3))
        self.RSyn_V0d_V2a = zeros((self.nV0d * self.nV2a,3))
        #ach
        self.LSyn_MN_Muscle = zeros((self.nMN * self.nMuscle,3))
        self.RSyn_MN_Muscle = zeros((self.nMN * self.nMuscle,3))
        #glu
        self.LSyn_V0v_IC = zeros((self.nV0v * self.nIC,3))
        self.RSyn_V0v_IC = zeros((self.nV0v * self.nIC,3))
        self.LSyn_V2a_V0v = zeros((self.nV2a * self.nV0v,3))
        self.RSyn_V2a_V0v = zeros((self.nV2a * self.nV0v,3))

        self.LSGap_IC_IC = zeros((self.nIC, self.nIC))
        self.LSGap_IC_MN = zeros((self.nIC, self.nMN))
        self.LSGap_IC_V0d = zeros((self.nIC, self.nV0d))
        self.LSGap_IC_V0v = zeros((self.nIC, self.nV0v))
        self.LSGap_IC_V2a = zeros((self.nIC, self.nV2a))
        self.LSGap_MN_IC = zeros((self.nMN, self.nIC))
        self.LSGap_MN_MN = zeros((self.nMN, self.nMN))
        self.LSGap_MN_V0d = zeros((self.nMN, self.nV0d))
        self.LSGap_MN_V0v = zeros((self.nMN, self.nV0v))
        self.LSGap_MN_V2a = zeros((self.nMN, self.nV2a))
        self.LSGap_V0d_IC = zeros((self.nV0d, self.nIC))
        self.LSGap_V0d_MN = zeros((self.nV0d, self.nMN))
        self.LSGap_V0d_V0d = zeros((self.nV0d, self.nV0d))
        self.LSGap_V0v_IC = zeros((self.nV0v, self.nIC))
        self.LSGap_V0v_MN = zeros((self.nV0v, self.nMN))
        self.LSGap_V0v_V0v = zeros((self.nV0v, self.nV0v))
        self.LSGap_V2a_IC = zeros((self.nV2a, self.nIC))
        self.LSGap_V2a_V2a = zeros((self.nV2a, self.nV2a))
        self.LSGap_V2a_MN = zeros((self.nV2a, self.nMN))

        self.RSGap_IC_IC = zeros((self.nIC, self.nIC))
        self.RSGap_IC_MN = zeros((self.nIC, self.nMN))
        self.RSGap_IC_V0d = zeros((self.nIC, self.nV0d))
        self.RSGap_IC_V0v = zeros((self.nIC, self.nV0v))
        self.RSGap_IC_V2a = zeros((self.nIC, self.nV2a))
        self.RSGap_MN_IC = zeros((self.nMN, self.nIC))
        self.RSGap_MN_MN = zeros((self.nMN, self.nMN))
        self.RSGap_MN_V0d = zeros((self.nMN, self.nV0d))
        self.RSGap_MN_V0v = zeros((self.nMN, self.nV0v))
        self.RSGap_MN_V2a = zeros((self.nMN, self.nV2a))
        self.RSGap_V0d_IC = zeros((self.nV0d, self.nIC))
        self.RSGap_V0d_MN = zeros((self.nV0d, self.nMN))
        self.RSGap_V0d_V0d = zeros((self.nV0d, self.nV0d))
        self.RSGap_V0v_IC = zeros((self.nV0v, self.nIC))
        self.RSGap_V0v_MN = zeros((self.nV0v, self.nMN))
        self.RSGap_V0v_V0v = zeros((self.nV0v, self.nV0v))
        self.RSGap_V2a_IC = zeros((self.nV2a, self.nIC))
        self.RSGap_V2a_V2a = zeros((self.nV2a, self.nV2a))
        self.RSGap_V2a_MN = zeros((self.nV2a, self.nMN))

        #Synaptic weight
        self.LW_V0d_MN = zeros((self.nV0d, self.nMN))
        self.RW_V0d_MN = zeros((self.nV0d, self.nMN))
        self.LW_V0d_IC = zeros((self.nV0d, self.nIC))
        self.RW_V0d_IC = zeros((self.nV0d, self.nIC))
        self.LW_V0d_V2a = zeros((self.nV0d, self.nV2a))
        self.RW_V0d_V2a = zeros((self.nV0d, self.nV2a))
        self.LW_V0v_IC = zeros((self.nV0v, self.nIC))
        self.RW_V0v_IC = zeros((self.nV0v, self.nIC))
        self.LW_V2a_V0v = zeros((self.nV2a, self.nV0v))
        self.RW_V2a_V0v = zeros((self.nV2a, self.nV0v))
        self.LW_MN_Muscle = zeros((self.nMN, self.nMuscle))
        self.RW_MN_Muscle = zeros((self.nMN, self.nMuscle))

        #Gap junctions coupling
        self.LGap_IC_MN = zeros((self.nIC, self.nMN))
        self.LGap_IC_V0d = zeros((self.nIC, self.nV0d))
        self.LGap_IC_IC = zeros((self.nIC, self.nIC))
        self.LGap_IC_V0v = zeros((self.nIC, self.nV0v))
        self.LGap_IC_V2a = zeros((self.nIC, self.nV2a))
        self.LGap_MN_MN = zeros((self.nMN, self.nMN))
        self.LGap_MN_V0d = zeros((self.nMN, self.nV0d))
        self.LGap_MN_V0v = zeros((self.nMN, self.nV0v))
        self.LGap_V0d_V0d = zeros((self.nV0d, self.nV0d))
        self.LGap_V0v_V0v = zeros((self.nV0v, self.nV0v))
        self.LGap_V2a_V2a = zeros((self.nV2a, self.nV2a))
        self.LGap_V2a_MN = zeros((self.nV2a, self.nMN))

        self.RGap_IC_MN = zeros((self.nIC, self.nMN))
        self.RGap_IC_V0d = zeros((self.nIC, self.nV0d))
        self.RGap_IC_IC = zeros((self.nIC, self.nIC))
        self.RGap_IC_V0v = zeros((self.nIC, self.nV0v))
        self.RGap_IC_V2a = zeros((self.nIC, self.nV2a))
        self.RGap_MN_MN = zeros((self.nMN, self.nMN))
        self.RGap_MN_V0d = zeros((self.nMN, self.nV0d))
        self.RGap_MN_V0v = zeros((self.nMN, self.nV0v))
        self.RGap_V0d_V0d = zeros((self.nV0d, self.nV0d))
        self.RGap_V0v_V0v = zeros((self.nV0v, self.nV0v))
        self.RGap_V2a_V2a = zeros((self.nV2a, self.nV2a))
        self.RGap_V2a_MN = zeros((self.nV2a, self.nMN))


        #self.res
        self.resLIC=zeros((self.nIC,3))
        self.resRIC=zeros((self.nIC,3))
        self.resLMN=zeros((self.nMN,3))
        self.resRMN=zeros((self.nMN,3))
        self.resLV0d = zeros((self.nV0d,3))
        self.resRV0d = zeros((self.nV0d,3))
        self.resLV0v = zeros((self.nV0v,3))
        self.resRV0v = zeros((self.nV0v,3))
        self.resLV2a = zeros((self.nV2a,3))
        self.resRV2a = zeros((self.nV2a,3))
        self.resLMuscle = zeros((self.nMuscle,2))
        self.resRMuscle = zeros((self.nMuscle,2))

    def initAndSetDistanceMatrices(self):
        #Initialize
        self.LD_IC_MN = zeros((self.nIC, self.nMN))
        self.LD_IC_V0d = zeros((self.nIC, self.nV0d))
        self.LD_IC_V0v = zeros((self.nIC, self.nV0v))
        self.LD_IC_V2a = zeros((self.nIC, self.nV2a))
        self.LD_MN_MN = zeros((self.nMN, self.nMN))
        self.LD_MN_V0d = zeros((self.nMN, self.nV0d))
        self.LD_MN_V0v = zeros((self.nMN, self.nV0v))
        self.LD_V0d_V0d = zeros((self.nV0d, self.nV0d))
        self.LD_V0d_IC = zeros((self.nV0d, self.nIC))
        self.LD_V0d_MN = zeros((self.nV0d, self.nMN))
        self.LD_V0d_V2a = zeros((self.nV0d, self.nV2a))
        self.LD_V0v_IC = zeros((self.nV0v, self.nIC))
        self.LD_V0v_V0v = zeros((self.nV0v, self.nV0v))
        self.LD_V2a_V2a = zeros((self.nV2a, self.nV2a))
        self.LD_V2a_MN = zeros((self.nV2a, self.nMN))
        self.LD_V2a_V0v = zeros((self.nV2a, self.nV0v))

        self.RD_IC_MN = zeros((self.nIC, self.nMN))
        self.RD_IC_V0d = zeros((self.nIC, self.nV0d))
        self.RD_IC_V0v = zeros((self.nIC, self.nV0v))
        self.RD_IC_V2a = zeros((self.nIC, self.nV2a))
        self.RD_MN_MN = zeros((self.nMN, self.nMN))
        self.RD_MN_V0d = zeros((self.nMN, self.nV0d))
        self.RD_MN_V0v = zeros((self.nMN, self.nV0v))
        self.RD_V0d_V0d = zeros((self.nV0d, self.nV0d))
        self.RD_V0d_IC = zeros((self.nV0d, self.nIC))
        self.RD_V0d_MN = zeros((self.nV0d, self.nMN))
        self.RD_V0d_V2a = zeros((self.nV0d, self.nV2a))
        self.RD_V0v_V0v = zeros((self.nV0v, self.nV0v))
        self.RD_V0v_IC = zeros((self.nV0v, self.nIC))
        self.RD_V2a_V2a = zeros((self.nV2a, self.nV2a))
        self.RD_V2a_MN = zeros((self.nV2a, self.nMN))
        self.RD_V2a_V0v = zeros((self.nV2a, self.nV0v))

        ## Compute distance between Neurons

        #LEFT
        for k in range (0, self.nIC):
            for l in range (0, self.nMN):
                self.LD_IC_MN[k,l] = Distance(self.L_IC[k].x,self.L_MN[l].x,self.L_IC[k].y,self.L_MN[l].y)

        for k in range (0, self.nIC):
            for l in range (0, self.nV0d):
                self.LD_IC_V0d[k,l] = Distance(self.L_IC[k].x,self.L_V0d[l].x,self.L_IC[k].y,self.L_V0d[l].y)

        for k in range (0, self.nIC):
            for l in range (0, self.nV0v):
                self.LD_IC_V0v[k,l] = Distance(self.L_IC[k].x,self.L_V0v[l].x,self.L_IC[k].y,self.L_V0v[l].y)

        for k in range (0, self.nIC):
            for l in range (0, self.nV2a):
                self.LD_IC_V2a[k,l] = Distance(self.L_IC[k].x,self.L_V2a[l].x,self.L_IC[k].y,self.L_V2a[l].y)

        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                self.LD_MN_MN[k,l] = Distance(self.L_MN[k].x,self.L_MN[l].x,self.L_MN[k].y,self.L_MN[l].y)

        for k in range (0, self.nMN):
            for l in range (0, self.nV0d):
                self.LD_MN_V0d[k,l] = Distance(self.L_MN[k].x,self.L_V0d[l].x,self.L_MN[k].y,self.L_V0d[l].y)

        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                self.LD_MN_V0v[k,l] = Distance(self.L_MN[k].x,self.L_V0v[l].x,self.L_MN[k].y,self.L_V0v[l].y)

        for k in range (0, self.nV0d):
            for l in range (0, self.nV0d):
                self.LD_V0d_V0d[k,l] = Distance(self.L_V0d[k].x,self.L_V0d[l].x,self.L_V0d[k].y,self.L_V0d[l].y)

        for k in range (0, self.nV0d):
            for l in range (0, self.nMN):
                self.LD_V0d_MN[k,l] = Distance(self.L_V0d[k].x,self.R_MN[l].x,self.L_V0d[k].y,self.R_MN[l].y) #Contralateral

        for k in range (0, self.nV0d):
            for l in range (0, self.nIC):
                self.LD_V0d_IC[k,l] = Distance(self.L_V0d[k].x,self.R_IC[l].x,self.L_V0d[k].y,self.R_IC[l].y) #Contralateral

        for k in range (0, self.nV0d):
            for l in range (0, self.nV2a):
                self.LD_V0d_V2a[k,l] = Distance(self.L_V0d[k].x,self.R_V2a[l].x,self.L_V0d[k].y,self.R_V2a[l].y) #Contralateral

        for k in range (0, self.nV0v):
            for l in range (0, self.nIC):
                self.LD_V0v_IC[k,l] = Distance(self.L_V0v[k].x,self.R_IC[l].x,self.L_V0v[k].y,self.R_IC[l].y) #Contralateral

        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                self.LD_V0v_V0v[k,l] = Distance(self.L_V0v[k].x,self.L_V0v[l].x,self.L_V0v[k].y,self.L_V0v[l].y)

        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                self.LD_V2a_V2a[k,l] = Distance(self.L_V2a[k].x,self.L_V2a[l].x,self.L_V2a[k].y,self.L_V2a[l].y)

        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                self.LD_V2a_MN[k,l] = Distance(self.L_V2a[k].x,self.L_MN[l].x,self.L_V2a[k].y,self.L_MN[l].y)

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                self.LD_V2a_V0v[k,l] = Distance(self.L_V2a[k].x,self.L_V0v[l].x,self.L_V2a[k].y,self.L_V0v[l].y)

        #RIGHT
        for k in range (0, self.nIC):
            for l in range (0, self.nMN):
                self.RD_IC_MN[k,l] = Distance(self.R_IC[k].x,self.R_MN[l].x,self.R_IC[k].y,self.R_MN[l].y)

        for k in range (0, self.nIC):
            for l in range (0, self.nV0d):
                self.RD_IC_V0d[k,l] = Distance(self.R_IC[k].x,self.R_V0d[l].x,self.R_IC[k].y,self.R_V0d[l].y)

        for k in range (0, self.nIC):
            for l in range (0, self.nV0v):
                self.RD_IC_V0v[k,l] = Distance(self.R_IC[k].x,self.R_V0v[l].x,self.R_IC[k].y,self.R_V0v[l].y)

        for k in range (0, self.nIC):
            for l in range (0, self.nV2a):
                self.RD_IC_V2a[k,l] = Distance(self.R_IC[k].x,self.R_V2a[l].x,self.R_IC[k].y,self.R_V2a[l].y)

        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                self.RD_MN_MN[k,l] = Distance(self.R_MN[k].x,self.R_MN[l].x,self.R_MN[k].y,self.R_MN[l].y)

        for k in range (0, self.nMN):
            for l in range (0, self.nV0d):
                self.RD_MN_V0d[k,l] = Distance(self.R_MN[k].x,self.R_V0d[l].x,self.R_MN[k].y,self.R_V0d[l].y)

        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                self.RD_MN_V0v[k,l] = Distance(self.R_MN[k].x,self.R_V0v[l].x,self.R_MN[k].y,self.R_V0v[l].y)

        for k in range (0, self.nV0d):
            for l in range (0, self.nIC):
                self.RD_V0d_IC[k,l] = Distance(self.R_V0d[k].x,self.L_IC[l].x,self.R_V0d[k].y,self.L_IC[l].y) #Contralateral

        for k in range (0, self.nV0d):
            for l in range (0, self.nMN):
                self.RD_V0d_MN[k,l] = Distance(self.R_V0d[k].x,self.L_MN[l].x,self.R_V0d[k].y,self.L_MN[l].y) #Contralateral

        for k in range (0, self.nV0d):
            for l in range (0, self.nV0d):
                self.RD_V0d_V0d[k,l] = Distance(self.R_V0d[k].x,self.R_V0d[l].x,self.R_V0d[k].y,self.R_V0d[l].y)

        for k in range (0, self.nV0d):
            for l in range (0, self.nV2a):
                self.RD_V0d_V2a[k,l] = Distance(self.R_V0d[k].x,self.L_V2a[l].x,self.R_V0d[k].y,self.L_V2a[l].y) #Contralateral

        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                self.RD_V0v_V0v[k,l] = Distance(self.R_V0v[k].x,self.R_V0v[l].x,self.R_V0v[k].y,self.R_V0v[l].y)

        for k in range (0, self.nV0v):
            for l in range (0, self.nIC):
                self.RD_V0v_IC[k,l] = Distance(self.R_V0v[k].x,self.L_IC[l].x,self.R_V0v[k].y,self.L_IC[l].y) #Contralateral

        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                self.RD_V2a_MN[k,l] = Distance(self.R_V2a[k].x,self.R_MN[l].x,self.R_V2a[k].y,self.R_MN[l].y)

        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                self.RD_V2a_V2a[k,l] = Distance(self.R_V2a[k].x,self.R_V2a[l].x,self.R_V2a[k].y,self.R_V2a[l].y)

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                self.RD_V2a_V0v[k,l] = Distance(self.R_V2a[k].x,self.R_V0v[l].x,self.R_V2a[k].y,self.R_V0v[l].y)

    def rangeNoiseMultiplier(self):
        return 1

    def gapWeightNoiseMultiplier(self):
        return gauss(1, self.sigma_gap)

    def synWeightNoiseMultiplier(self):
        return gauss(1, self.sigma_chem)

    def computeSynapticAndGapWeights(self):
        #region LEFT Gap Junctions
        for k in range (0, self.nIC):
            for l in range (0, self.nIC):
                if (k != l):                                         # because it is a kernel of ICs, there is no distance condition
                    self.LGap_IC_IC[k,l] = self.IC_IC_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_IC_IC[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.LD_IC_MN[k,l] < self.rangeIC_MN * self.rangeNoiseMultiplier()  and self.L_IC[k].x < self.L_MN[l].x):     #the second condition is because the connection is descending
                    self.LGap_IC_MN[k,l] = self.IC_MN_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_IC_MN[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nV0d):
                if (self.rangeMin < self.LD_IC_V0d[k,l] < self.rangeIC_V0d * self.rangeNoiseMultiplier()  and self.L_IC[k].x < self.L_V0d[l].x):     #the second condition is because the connection is descending
                    self.LGap_IC_V0d[k,l] = self.IC_V0d_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_IC_V0d[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.LD_IC_V0v[k,l] < self.rangeIC_V0v * self.rangeNoiseMultiplier()  and self.L_IC[k].x < self.L_V0v[l].x):     #the second condition is because the connection is descending
                    self.LGap_IC_V0v[k,l] = self.IC_V0v_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_IC_V0v[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.LD_IC_V2a[k,l] < self.rangeIC_V2a * self.rangeNoiseMultiplier()  and self.L_IC[k].x < self.L_V2a[l].x):     #the second condition is because the connection is descending
                    self.LGap_IC_V2a[k,l] = self.IC_V2a_gap_weight  * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_IC_V2a[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.LD_MN_MN[k,l] < self.rangeMN_MN * self.rangeNoiseMultiplier()  ):
                    self.LGap_MN_MN[k,l] = self.MN_MN_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_MN_MN[k,l] = 0.0

        for k in range (0, self.nV0d):
            for l in range (0, self.nV0d):
                if (self.rangeMin < self.LD_V0d_V0d[k,l] < self.rangeV0d_V0d * self.rangeNoiseMultiplier()  ):
                    self.LGap_V0d_V0d[k,l] = self.V0d_V0d_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_V0d_V0d[k,l] = 0.0

        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.LD_V0v_V0v[k,l] < self.rangeV0v_V0v * self.rangeNoiseMultiplier()  ):
                    self.LGap_V0v_V0v[k,l] = self.V0v_V0v_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_V0v_V0v[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nV0d):
                if (self.L_V0d[l].x-self.rangeMN_V0d * self.rangeNoiseMultiplier()  < self.L_MN[k].x< self.L_V0d[l].x + self.rangeMN_V0d * self.rangeNoiseMultiplier()  ):
                    self.LGap_MN_V0d[k,l] = self.MN_V0d_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_MN_V0d[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                if (self.L_V0v[l].x - self.rangeMN_V0v * self.rangeNoiseMultiplier()  < self.L_MN[k].x< self.L_V0v[l].x + self.rangeMN_V0v * self.rangeNoiseMultiplier()  ):
                    self.LGap_MN_V0v[k,l] = self.MN_V0v_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_MN_V0v[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.LD_V2a_V2a[k,l] < self.rangeV2a_V2a * self.rangeNoiseMultiplier() ):
                    self.LGap_V2a_V2a[k,l] = self.V2a_V2a_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_V2a_V2a[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.LD_V2a_MN[k,l] < self.rangeV2a_MN * self.rangeNoiseMultiplier()  ):
                    self.LGap_V2a_MN[k,l] = self.V2a_MN_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.LGap_V2a_MN[k,l] = 0.0
        #endregion

        #region LEFT Synapses
        for k in range (0, self.nV0d):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.LD_V0d_MN[k,l] < self.rangeV0d_MN * self.rangeNoiseMultiplier() ):
                    self.LW_V0d_MN[k,l] = self.V0d_MN_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.LW_V0d_MN[k,l] = 0.0

        for k in range (0, self.nV0d):
            for l in range (0, self.nIC):
                if (self.rangeMin < self.LD_V0d_IC[k,l] < self.rangeV0d_IC * self.rangeNoiseMultiplier() ):
                    self.LW_V0d_IC[k,l] = self.V0d_IC_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.LW_V0d_IC[k,l] = 0.0

        for k in range (0, self.nV0d):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.LD_V0d_V2a[k,l] < self.rangeV0d_V2a * self.rangeNoiseMultiplier() ):
                    self.LW_V0d_V2a[k,l] = self.V0d_V2a_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.LW_V0d_V2a[k,l] = 0.0

        for k in range (0, self.nV0v):
            for l in range (0, self.nIC):
                if (self.rangeV0v_IC_min < self.LD_V0v_IC[k,l] < self.rangeV0v_IC_max * self.rangeNoiseMultiplier() ):
                    self.LW_V0v_IC[k,l] =  self.V0v_IC_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.LW_V0v_IC[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.LD_V2a_V0v[k,l] < self.rangeV2a_V0v_desc * self.rangeNoiseMultiplier()  and self.L_V2a[k].x < self.L_V0v[l].x) or \
                   (self.rangeMin < self.LD_V2a_V0v[k,l] < self.rangeV2a_V0v_asc * self.rangeNoiseMultiplier()  and self.L_V2a[k].x > self.L_V0v[l].x):    #descending and short ascending
                    self.LW_V2a_V0v[k,l] = self.V2a_V0v_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.LW_V2a_V0v[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nMuscle):
                if (self.L_Muscle[l].x - self.rangeMN_Muscle < self.L_MN[k].x < self.L_Muscle[l].x + self.rangeMN_Muscle):         #because segments
                    self.LW_MN_Muscle[k,l] = self.MN_Muscle_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.LW_MN_Muscle[k,l] = 0.0
        #endregion

        #region RIGHT Gap Junctions
        for k in range (0, self.nIC):
            for l in range (0, self.nIC):
                if (k!= l):                                         # because it is a kernel of ICs, there is no distance condition
                    self.RGap_IC_IC[k,l] = self.IC_IC_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_IC_IC[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.RD_IC_MN[k,l] < self.rangeIC_MN  * self.rangeNoiseMultiplier() and self.R_IC[k].x < self.R_MN[l].x):     #the second condition is because the connection is descending
                    self.RGap_IC_MN[k,l] = self.IC_MN_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_IC_MN[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nV0d):
                if (self.rangeMin < self.RD_IC_V0d[k,l] < self.rangeIC_V0d * self.rangeNoiseMultiplier()  and self.R_IC[k].x < self.R_V0d[l].x):     #the second condition is because the connection is descending
                    self.RGap_IC_V0d[k,l] = self.IC_V0d_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_IC_V0d[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.RD_IC_V0v[k,l] < self.rangeIC_V0v * self.rangeNoiseMultiplier()  and self.R_IC[k].x < self.R_V0v[l].x):     #the second condition is because the connection is descending
                    self.RGap_IC_V0v[k,l] = self.IC_V0v_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_IC_V0v[k,l] = 0.0

        for k in range (0, self.nIC):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.RD_IC_V2a[k,l] < self.rangeIC_V2a * self.rangeNoiseMultiplier()  and self.R_IC[k].x < self.R_V2a[l].x):     #the second condition is because the connection is descending
                    self.RGap_IC_V2a[k,l] = self.IC_V2a_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_IC_V2a[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.RD_MN_MN[k,l] < self.rangeMN_MN * self.rangeNoiseMultiplier()  ):
                    self.RGap_MN_MN[k,l] = self.MN_MN_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_MN_MN[k,l] = 0.0

        for k in range (0, self.nV0d):
            for l in range (0, self.nV0d):
                if (self.rangeMin < self.RD_V0d_V0d[k,l] < self.rangeV0d_V0d  * self.rangeNoiseMultiplier() ):
                    self.RGap_V0d_V0d[k,l] = self.V0d_V0d_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_V0d_V0d[k,l] = 0.0

        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.RD_V0v_V0v[k,l] < self.rangeV0v_V0v  * self.rangeNoiseMultiplier() ):
                    self.RGap_V0v_V0v[k,l] = self.V0v_V0v_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_V0v_V0v[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nV0d):
                if (self.R_V0d[l].x - self.rangeMN_V0d * self.rangeNoiseMultiplier() < self.R_MN[k].x< self.R_V0d[l].x + self.rangeMN_V0d * self.rangeNoiseMultiplier()  ):
                    self.RGap_MN_V0d[k,l] = self.MN_V0d_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_MN_V0d[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                if (self.R_V0v[l].x - self.rangeMN_V0v * self.rangeNoiseMultiplier()  < self.R_MN[k].x< self.R_V0v[l].x + self.rangeMN_V0v * self.rangeNoiseMultiplier()  ):
                    self.RGap_MN_V0v[k,l] = self.MN_V0v_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_MN_V0v[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.RD_V2a_V2a[k,l] < self.rangeV2a_V2a * self.rangeNoiseMultiplier() ):
                    self.RGap_V2a_V2a[k,l] = self.V2a_V2a_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_V2a_V2a[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.RD_V2a_MN[k,l] < self.rangeV2a_MN * self.rangeNoiseMultiplier()  ):
                    self.RGap_V2a_MN[k,l] =  self.V2a_MN_gap_weight * self.gapWeightNoiseMultiplier()
                else:
                    self.RGap_V2a_MN[k,l] = 0.0
        #endregion

        #region RIGHT Synapses
        for k in range (0, self.nV0d):
            for l in range (0, self.nMN):
                if (self.rangeMin < self.RD_V0d_MN[k,l] < self.rangeV0d_MN * self.rangeNoiseMultiplier() ):
                    self.RW_V0d_MN[k,l] = self.V0d_MN_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.RW_V0d_MN[k,l] = 0.0

        for k in range (0, self.nV0d):
            for l in range (0, self.nIC):
                if (self.rangeMin < self.RD_V0d_IC[k,l] < self.rangeV0d_IC * self.rangeNoiseMultiplier() ):
                    self.RW_V0d_IC[k,l] = self.V0d_IC_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.RW_V0d_IC[k,l] = 0.0

        for k in range (0, self.nV0d):
            for l in range (0, self.nV2a):
                if (self.rangeMin < self.RD_V0d_V2a[k,l] < self.rangeV0d_V2a * self.rangeNoiseMultiplier() ):
                    self.RW_V0d_V2a[k,l] = self.V0d_V2a_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.RW_V0d_V2a[k,l] = 0.0

        for k in range (0, self.nV0v):
            for l in range (0, self.nIC):
                if (self.rangeV0v_IC_min < self.RD_V0v_IC[k,l] < self.rangeV0v_IC_max * self.rangeNoiseMultiplier() ):
                    self.RW_V0v_IC[k,l] = self.V0v_IC_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.RW_V0v_IC[k,l] = 0.0

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                if (self.rangeMin < self.RD_V2a_V0v[k,l] < self.rangeV2a_V0v_desc * self.rangeNoiseMultiplier()  and self.R_V2a[k].x < self.R_V0v[l].x) or \
                   (self.rangeMin < self.RD_V2a_V0v[k,l] < self.rangeV2a_V0v_asc * self.rangeNoiseMultiplier()  and self.R_V2a[k].x > self.R_V0v[l].x):    #descending and short ascending branch
                    self.RW_V2a_V0v[k,l] =  self.V2a_V0v_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.RW_V2a_V0v[k,l] = 0.0

        for k in range (0, self.nMN):
            for l in range (0, self.nMuscle):
                if (self.R_Muscle[l].x - self.rangeMN_Muscle < self.R_MN[k].x < self.R_Muscle[l].x + self.rangeMN_Muscle):         #segmental connection
                    self.RW_MN_Muscle[k,l] = self.MN_Muscle_syn_weight * self.synWeightNoiseMultiplier()
                else:
                    self.RW_MN_Muscle[k,l] = 0.0
        #endregion

    def getStimulus(self):
        return self.stim0

    #initializeMembranePotentials is called from the mainLoop.
    #Sets initial membrane potentials to -65 for neurons (with u=0, stim=0),
    # except -64 for V2a's (u=-16, stim=-64)
    # 0 for muscle cells
    def initializeMembranePotentials(self):
        ## initialize membrane potential values
        for k in range (0, self.nIC):
            self.resLIC[k,:] = self.L_IC[k].getNextVal(-65,0,0)
            self.VLIC[k,0] = self.resLIC[k,0]

            self.resRIC[k,:] = self.R_IC[k].getNextVal(-65,0,0)
            self.VRIC[k,0] = self.resRIC[k,0]

        for k in range (0, self.nMN):
            self.resLMN[k,:] = self.L_MN[k].getNextVal(-65,0,0)
            self.VLMN[k,0] = self.resLMN[k,0]

            self.resRMN[k,:] = self.R_MN[k].getNextVal(-65,0,0)
            self.VRMN[k,0] = self.resRMN[k,0]

        for k in range (0, self.nV0d):
            self.resLV0d[k,:] = self.L_V0d[k].getNextVal(-65,0,0)
            self.VLV0d[k,0] = self.resLV0d[k,0]

            self.resRV0d[k,:] = self.R_V0d[k].getNextVal(-65,0,0)
            self.VRV0d[k,0] = self.resRV0d[k,0]

        for k in range (0, self.nV0v):
            self.resLV0v[k,:] = self.L_V0v[k].getNextVal(-65,0,0)
            self.VLV0v[k,0] = self.resLV0v[k,0]

            self.resRV0v[k,:] = self.R_V0v[k].getNextVal(-65,0,0)
            self.VRV0v[k,0] = self.resRV0v[k,0]

        for k in range (0, self.nV2a):
            self.resLV2a[k,:] = self.L_V2a[k].getNextVal(-64,-16,-64)
            self.VLV2a[k,0] = self.resLV2a[k,0]

            self.resRV2a[k,:] = self.R_V2a[k].getNextVal(-64,-16,-64)
            self.VRV2a[k,0] = self.resRV2a[k,0]

        for k in range (0, self.nMuscle):
            self.resLMuscle[k,:] = self.L_Muscle[k].getNextVal(0,0)
            self.VLMuscle[k,0] = self.resLMuscle[k,0]

            self.resRMuscle[k,:] = self.R_Muscle[k].getNextVal(0,0)
            self.VRMuscle[k,0] = self.resRMuscle[k,0]

    def calcSynapticOutputs(self, t):
        for k in range (0, self.nV0d):
            for l in range (0, self.nMN):
                self.LSyn_V0d_MN[self.nMN*k+l,:] = self.L_glysyn_V0d_MN.getNextVal(self.VLV0d[k,t-int(self.LD_V0d_MN[k,l]/(self.dt * self.cv))], self.VRMN[l,t-1], self.LSyn_V0d_MN[self.nMN*k+l,1], self.LSyn_V0d_MN[self.nMN*k+l,2]) #Contralateral
                self.RSyn_V0d_MN[self.nMN*k+l,:] = self.R_glysyn_V0d_MN.getNextVal(self.VRV0d[k,t-int(self.RD_V0d_MN[k,l]/(self.dt * self.cv))], self.VLMN[l,t-1], self.RSyn_V0d_MN[self.nMN*k+l,1], self.RSyn_V0d_MN[self.nMN*k+l,2]) #Contralateral

        for k in range (0, self.nV0d):
            for l in range (0, self.nIC):
                self.LSyn_V0d_IC[self.nIC*k+l,:] = self.L_glysyn_V0d_IC.getNextVal(self.VLV0d[k,t-int(self.LD_V0d_IC[k,l]/(self.dt * self.cv))], self.VRIC[l,t-1], self.LSyn_V0d_IC[self.nIC*k+l,1], self.LSyn_V0d_IC[self.nIC*k+l,2]) #Contralateral
                self.RSyn_V0d_IC[self.nIC*k+l,:] = self.R_glysyn_V0d_IC.getNextVal(self.VRV0d[k,t-int(self.RD_V0d_IC[k,l]/(self.dt * self.cv))], self.VLIC[l,t-1], self.RSyn_V0d_IC[self.nIC*k+l,1], self.RSyn_V0d_IC[self.nIC*k+l,2]) #Contralateral

        for k in range (0, self.nV0d):
            for l in range (0, self.nV2a):
                self.LSyn_V0d_V2a[self.nV2a*k+l,:] = self.L_glysyn_V0d_V2a.getNextVal(self.VLV0d[k,t-int(self.LD_V0d_V2a[k,l]/(self.dt * self.cv))], self.VRV2a[l,t-1], self.LSyn_V0d_V2a[self.nV2a*k+l,1], self.LSyn_V0d_V2a[self.nV2a*k+l,2]) #Contralateral
                self.RSyn_V0d_V2a[self.nV2a*k+l,:] = self.R_glysyn_V0d_V2a.getNextVal(self.VRV0d[k,t-int(self.RD_V0d_V2a[k,l]/(self.dt * self.cv))], self.VLV2a[l,t-1], self.RSyn_V0d_V2a[self.nV2a*k+l,1], self.RSyn_V0d_V2a[self.nV2a*k+l,2]) #Contralateral

        for k in range (0, self.nV0v):
            for l in range (0, self.nIC):
                self.LSyn_V0v_IC[self.nIC*k+l,:] = self.L_glusyn_V0v_IC.getNextVal(self.VLV0v[k,t-int(self.LD_V0v_IC[k,l]/(self.dt * self.cv))], self.VRIC[l,t-1], self.LSyn_V0v_IC[self.nIC*k+l,1], self.LSyn_V0v_IC[self.nIC*k+l,2])  #Contralateral
                self.RSyn_V0v_IC[self.nIC*k+l,:] = self.R_glusyn_V0v_IC.getNextVal(self.VRV0v[k,t-int(self.RD_V0v_IC[k,l]/(self.dt * self.cv))], self.VLIC[l,t-1], self.RSyn_V0v_IC[self.nIC*k+l,1], self.RSyn_V0v_IC[self.nIC*k+l,2])  #Contralateral

        for k in range (0, self.nV2a):
            for l in range (0, self.nV0v):
                self.LSyn_V2a_V0v[self.nV0v*k+l,:] = self.L_glusyn_V2a_V0v.getNextVal(self.VLV2a[k,t-int(self.LD_V2a_V0v[k,l]/(self.dt * self.cv))], self.VLV0v[l,t-1], self.LSyn_V2a_V0v[self.nV0v*k+l,1], self.LSyn_V2a_V0v[self.nV0v*k+l,2])
                self.RSyn_V2a_V0v[self.nV0v*k+l,:] = self.R_glusyn_V2a_V0v.getNextVal(self.VRV2a[k,t-int(self.RD_V2a_V0v[k,l]/(self.dt * self.cv))], self.VRV0v[l,t-1], self.RSyn_V2a_V0v[self.nV0v*k+l,1], self.RSyn_V2a_V0v[self.nV0v*k+l,2])

        for k in range (0, self.nMN):
            for l in range (0, self.nMuscle):
                self.LSyn_MN_Muscle[self.nMuscle*k+l,:] = self.L_achsyn_MN_Muscle.getNextVal(self.VLMN[k,t-10], self.VLMuscle[l,t-1], self.LSyn_MN_Muscle[self.nMuscle*k+l,1], self.LSyn_MN_Muscle[self.nMuscle*k+l,2])
                self.RSyn_MN_Muscle[self.nMuscle*k+l,:] = self.R_achsyn_MN_Muscle.getNextVal(self.VRMN[k,t-10], self.VRMuscle[l,t-1], self.RSyn_MN_Muscle[self.nMuscle*k+l,1], self.RSyn_MN_Muscle[self.nMuscle*k+l,2])

    def calcICOutputs(self,t):
        for k in range (0, self.nIC):
            for l in range (0, self.nMN):
                self.RSGap_IC_MN[k,l] = self.RGap_IC_MN[k,l]*(self.VRIC[k,t-int(self.RD_IC_MN[k,l]/(self.dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_IC_MN[k,l] = self.LGap_IC_MN[k,l]*(self.VLIC[k,t-int(self.LD_IC_MN[k,l]/(self.dt * self.cv))]-self.VLMN[l,t-1])

        for k in range (0, self.nIC):
            for l in range (0, self.nIC):
                self.RSGap_IC_IC[k,l] = self.RGap_IC_IC[k,l]*(self.VRIC[k,t-1]-self.VRIC[l,t-1])
                self.LSGap_IC_IC[k,l] = self.LGap_IC_IC[k,l]*(self.VLIC[k,t-1]-self.VLIC[l,t-1])

        for k in range (0, self.nIC):
            for l in range (0, self.nV0d):
                self.RSGap_IC_V0d[k,l] = self.RGap_IC_V0d[k,l]*(self.VRIC[k,t-int(self.RD_IC_V0d[k,l]/(self.dt * self.cv))]-self.VRV0d[l,t-1])
                self.LSGap_IC_V0d[k,l] = self.LGap_IC_V0d[k,l]*(self.VLIC[k,t-int(self.LD_IC_V0d[k,l]/(self.dt * self.cv))]-self.VLV0d[l,t-1])

        for k in range (0, self.nIC):
            for l in range (0, self.nV0v):
                self.RSGap_IC_V0v[k,l] = self.RGap_IC_V0v[k,l]*(self.VRIC[k,t-int(self.RD_IC_V0v[k,l]/(self.dt * self.cv))]-self.VRV0v[l,t-1])
                self.LSGap_IC_V0v[k,l] = self.LGap_IC_V0v[k,l]*(self.VLIC[k,t-int(self.LD_IC_V0v[k,l]/(self.dt * self.cv))]-self.VLV0v[l,t-1])

        for k in range (0, self.nIC):
            for l in range (0, self.nV2a):
                self.RSGap_IC_V2a[k,l] = self.RGap_IC_V2a[k,l]*(self.VRIC[k,t-int(self.RD_IC_V2a[k,l]/(self.dt * self.cv))]-self.VRV2a[l,t-1])
                self.LSGap_IC_V2a[k,l] = self.LGap_IC_V2a[k,l]*(self.VLIC[k,t-int(self.LD_IC_V2a[k,l]/(self.dt * self.cv))]-self.VLV2a[l,t-1])

    def calcMNOutputs(self,t):

        for k in range (0, self.nMN):
            for l in range (0, self.nIC):
                self.RSGap_MN_IC[k,l] = self.RGap_IC_MN[l,k]*(self.VRMN[k,t-int(self.RD_IC_MN[l,k]/(self.dt * self.cv))]-self.VRIC[l,t-1])
                self.LSGap_MN_IC[k,l] = self.LGap_IC_MN[l,k]*(self.VLMN[k,t-int(self.LD_IC_MN[l,k]/(self.dt * self.cv))]-self.VLIC[l,t-1])

        for k in range (0, self.nMN):
            for l in range (0, self.nMN):
                self.RSGap_MN_MN[k,l] = self.RGap_MN_MN[k,l]*(self.VRMN[k,t-int(self.RD_MN_MN[k,l]/(self.dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_MN_MN[k,l] = self.LGap_MN_MN[k,l]*(self.VLMN[k,t-int(self.LD_MN_MN[k,l]/(self.dt * self.cv))]-self.VLMN[l,t-1])

        for k in range (0, self.nMN):
            for l in range (0, self.nV0d):
                self.RSGap_MN_V0d[k,l] = self.RGap_MN_V0d[k,l]*(self.VRMN[k,t-int(self.RD_MN_V0d[k,l]/(self.dt * self.cv))]-self.VRV0d[l,t-1])
                self.LSGap_MN_V0d[k,l] = self.LGap_MN_V0d[k,l]*(self.VLMN[k,t-int(self.LD_MN_V0d[k,l]/(self.dt * self.cv))]-self.VLV0d[l,t-1])

        for k in range (0, self.nMN):
            for l in range (0, self.nV0v):
                self.RSGap_MN_V0v[k,l] = self.RGap_MN_V0v[k,l]*(self.VRMN[k,t-int(self.RD_MN_V0v[k,l]/(self.dt * self.cv))]-self.VRV0v[l,t-1])
                self.LSGap_MN_V0v[k,l] = self.LGap_MN_V0v[k,l]*(self.VLMN[k,t-int(self.LD_MN_V0v[k,l]/(self.dt * self.cv))]-self.VLV0v[l,t-1])

        for k in range (0, self.nMN):
            for l in range (0, self.nV2a):
                self.RSGap_MN_V2a[k,l] = self.RGap_V2a_MN[l,k]*(self.VRMN[k,t-int(self.RD_V2a_MN[l,k]/(self.dt * self.cv))]-self.VRV2a[l,t-1])
                self.LSGap_MN_V2a[k,l] = self.LGap_V2a_MN[l,k]*(self.VLMN[k,t-int(self.LD_V2a_MN[l,k]/(self.dt * self.cv))]-self.VLV2a[l,t-1])

    def calcV0dOutputs(self,t):

        for k in range (0, self.nV0d):
            for l in range (0, self.nIC):
                self.RSGap_V0d_IC[k,l] = self.RGap_IC_V0d[l,k]*(self.VRV0d[k,t-int(self.RD_IC_V0d[l,k]/(self.dt * self.cv))]-self.VRIC[l,t-1])
                self.LSGap_V0d_IC[k,l] = self.LGap_IC_V0d[l,k]*(self.VLV0d[k,t-int(self.LD_IC_V0d[l,k]/(self.dt * self.cv))]-self.VLIC[l,t-1])

        for k in range (0, self.nV0d):
            for l in range (0, self.nMN):
                self.RSGap_V0d_MN[k,l] = self.RGap_MN_V0d[l,k]*(self.VRV0d[k,t-int(self.RD_MN_V0d[l,k]/(self.dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_V0d_MN[k,l] = self.LGap_MN_V0d[l,k]*(self.VLV0d[k,t-int(self.LD_MN_V0d[l,k]/(self.dt * self.cv))]-self.VLMN[l,t-1])

        for k in range (0, self.nV0d):
            for l in range (0, self.nV0d):
                self.RSGap_V0d_V0d[k,l] = self.RGap_V0d_V0d[k,l]*(self.VRV0d[k,t-int(self.RD_V0d_V0d[k,l]/(self.dt * self.cv))]-self.VRV0d[l,t-1])
                self.LSGap_V0d_V0d[k,l] = self.LGap_V0d_V0d[k,l]*(self.VLV0d[k,t-int(self.LD_V0d_V0d[k,l]/(self.dt * self.cv))]-self.VLV0d[l,t-1])

    def calcV0vOutputs(self,t):

        for k in range (0, self.nV0v):
            for l in range (0, self.nIC):
                self.RSGap_V0v_IC[k,l] = self.RGap_IC_V0v[l,k]*(self.VRV0v[k,t-int(self.RD_IC_V0v[l,k]/(self.dt * self.cv))]-self.VRIC[l,t-1])
                self.LSGap_V0v_IC[k,l] = self.LGap_IC_V0v[l,k]*(self.VLV0v[k,t-int(self.LD_IC_V0v[l,k]/(self.dt * self.cv))]-self.VLIC[l,t-1])

        for k in range (0, self.nV0v):
            for l in range (0, self.nMN):
                self.RSGap_V0v_MN[k,l] = self.RGap_MN_V0v[l,k]*(self.VRV0v[k,t-int(self.RD_MN_V0v[l,k]/(self.dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_V0v_MN[k,l] = self.LGap_MN_V0v[l,k]*(self.VLV0v[k,t-int(self.LD_MN_V0v[l,k]/(self.dt * self.cv))]-self.VLMN[l,t-1])

        for k in range (0, self.nV0v):
            for l in range (0, self.nV0v):
                self.RSGap_V0v_V0v[k,l] = self.RGap_V0v_V0v[k,l]*(self.VRV0v[k,t-int(self.RD_V0v_V0v[k,l]/(self.dt * self.cv))]-self.VRV0v[l,t-1])
                self.LSGap_V0v_V0v[k,l] = self.LGap_V0v_V0v[k,l]*(self.VLV0v[k,t-int(self.LD_V0v_V0v[k,l]/(self.dt * self.cv))]-self.VLV0v[l,t-1])

    def calcV2aOutputs(self,t):

        for k in range (0, self.nV2a):
            for l in range (0, self.nIC):
                self.RSGap_V2a_IC[k,l] = self.RGap_IC_V2a[l,k]*(self.VRV2a[k,t-int(self.RD_IC_V2a[l,k]/(self.dt * self.cv))]-self.VRIC[l,t-1])
                self.LSGap_V2a_IC[k,l] = self.LGap_IC_V2a[l,k]*(self.VLV2a[k,t-int(self.LD_IC_V2a[l,k]/(self.dt * self.cv))]-self.VLIC[l,t-1])

        for k in range (0, self.nV2a):
            for l in range (0, self.nV2a):
                self.RSGap_V2a_V2a[k,l] = self.RGap_V2a_V2a[k,l]*(self.VRV2a[k,t-int(self.RD_V2a_V2a[k,l]/(self.dt * self.cv))]-self.VRV2a[l,t-1])
                self.LSGap_V2a_V2a[k,l] = self.LGap_V2a_V2a[k,l]*(self.VLV2a[k,t-int(self.LD_V2a_V2a[k,l]/(self.dt * self.cv))]-self.VLV2a[l,t-1])

        for k in range (0, self.nV2a):
            for l in range (0, self.nMN):
                self.RSGap_V2a_MN[k,l] = self.RGap_V2a_MN[k,l]*(self.VRV2a[k,t-int(self.RD_V2a_MN[k,l]/(self.dt * self.cv))]-self.VRMN[l,t-1])
                self.LSGap_V2a_MN[k,l] = self.LGap_V2a_MN[k,l]*(self.VLV2a[k,t-int(self.LD_V2a_MN[k,l]/(self.dt * self.cv))]-self.VLMN[l,t-1])

    def calcMembranePotentialsFromCurrents(self, t):

        ## Determine membrane potentials from synaptic and external currents
        self.calcICPotentialsandResidues(t)
        self.calcMNPotentialsandResidues(t)
        self.calcV0dPotentialsandResidues(t)
        self.calcV0vPotentialsandResidues(t)
        self.calcV2aPotentialsandResidues(t)
        self.calcMusclePotentialsandResidues(t)

    def calcICPotentialsandResidues(self, t):
        for k in range (0, self.nIC):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.RSyn_V0v_IC[self.nIC*l+k,0]*self.LW_V0v_IC[l,k] for l in range (0, self.nV0v)) + sum(self.RSyn_V0d_IC[self.nIC*l+k,0]*self.LW_V0d_IC[l,k] for l in range (0, self.nV0d))
                IsynR = sum(self.LSyn_V0v_IC[self.nIC*l+k,0]*self.RW_V0v_IC[l,k] for l in range (0, self.nV0v)) + sum(self.LSyn_V0d_IC[self.nIC*l+k,0]*self.RW_V0d_IC[l,k] for l in range (0, self.nV0d))

            if t < self.tasyncdelay: #to ensure that double coils are not due to coincident coiling
                right_side_delay=0
            else:
                right_side_delay=1

            IgapL = sum(self.LSGap_IC_IC[:,k]) - sum(self.LSGap_IC_IC[k,:]) + sum(self.LSGap_MN_IC[:,k]) - sum(self.LSGap_IC_MN[k,:]) + sum(self.LSGap_V0d_IC[:,k]) - sum(self.LSGap_IC_V0d[k,:]) + sum(self.LSGap_V0v_IC[:,k]) - sum(self.LSGap_IC_V0v[k,:]) + sum(self.LSGap_V2a_IC[:,k]) - sum(self.LSGap_IC_V2a[k,:])
            IgapR = sum(self.RSGap_IC_IC[:,k]) - sum(self.RSGap_IC_IC[k,:]) + sum(self.RSGap_MN_IC[:,k]) - sum(self.RSGap_IC_MN[k,:]) + sum(self.RSGap_V0d_IC[:,k]) - sum(self.RSGap_IC_V0d[k,:])  + sum(self.RSGap_V0v_IC[:,k]) - sum(self.RSGap_IC_V0v[k,:]) + sum(self.RSGap_V2a_IC[:,k]) - sum(self.RSGap_IC_V2a[k,:])

            self.resLIC[k,:] = self.L_IC[k].getNextVal(self.resLIC[k,0],self.resLIC[k,1], self.stim[t] + IgapL + IsynL)
            self.VLIC[k,t] = self.resLIC[k,0]

            self.resRIC[k,:] = self.R_IC[k].getNextVal(self.resRIC[k,0],self.resRIC[k,1], self.stim[t]*right_side_delay + IgapR + IsynR)
            self.VRIC[k,t] = self.resRIC[k,0]

    def calcMNPotentialsandResidues(self, t):
        for k in range (0, self.nMN):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.RSyn_V0d_MN[self.nMN*l+k,0]*self.LW_V0d_MN[l,k] for l in range (0, self.nV0d))
                IsynR = sum(self.LSyn_V0d_MN[self.nMN*l+k,0]*self.RW_V0d_MN[l,k] for l in range (0, self.nV0d))
            #if k == 4: # this is to hyperpolarize a MN to observe periodic depolarizations and synaptic bursts
             #   IsynL = IsynL - 10

            IgapL = - sum(self.LSGap_MN_IC[k,:]) + sum(self.LSGap_IC_MN[:,k]) - sum(self.LSGap_MN_MN[k,:]) + sum(self.LSGap_MN_MN[:,k]) - sum(self.LSGap_MN_V0d[k,:]) + sum(self.LSGap_V0d_MN[:,k]) - sum(self.LSGap_MN_V0v[k,:]) + sum(self.LSGap_V0v_MN[:,k]) - sum(self.LSGap_MN_V2a[k,:]) + sum(self.LSGap_V2a_MN[:,k])
            IgapR = - sum(self.RSGap_MN_IC[k,:]) + sum(self.RSGap_IC_MN[:,k]) - sum(self.RSGap_MN_MN[k,:]) + sum(self.RSGap_MN_MN[:,k]) - sum(self.RSGap_MN_V0d[k,:]) + sum(self.RSGap_V0d_MN[:,k]) - sum(self.RSGap_MN_V0v[k,:]) + sum(self.RSGap_V0v_MN[:,k]) - sum(self.LSGap_MN_V2a[k,:]) + sum(self.LSGap_V2a_MN[:,k])

            self.resLMN[k,:] = self.L_MN[k].getNextVal(self.resLMN[k,0],self.resLMN[k,1], IgapL + IsynL)
            self.VLMN[k,t] = self.resLMN[k,0]

            self.resRMN[k,:] = self.R_MN[k].getNextVal(self.resRMN[k,0],self.resRMN[k,1], IgapR + IsynR)
            self.VRMN[k,t] = self.resRMN[k,0]

    def calcV0dPotentialsandResidues(self, t):

        for k in range (0, self.nV0d):
            IgapL = - sum(self.LSGap_V0d_IC[k,:]) + sum(self.LSGap_IC_V0d[:,k]) - sum(self.LSGap_V0d_V0d[k,:]) + sum(self.LSGap_V0d_V0d[:,k]) - sum(self.LSGap_V0d_MN[k,:]) + sum(self.LSGap_MN_V0d[:,k])
            IgapR = - sum(self.RSGap_V0d_IC[k,:]) + sum(self.RSGap_IC_V0d[:,k]) - sum(self.RSGap_V0d_V0d[k,:]) + sum(self.RSGap_V0d_V0d[:,k]) - sum(self.RSGap_V0d_MN[k,:]) + sum(self.RSGap_MN_V0d[:,k])

            self.resLV0d[k,:] = self.L_V0d[k].getNextVal(self.resLV0d[k,0],self.resLV0d[k,1], IgapL)
            self.VLV0d[k,t] = self.resLV0d[k,0]
            self.resRV0d[k,:] = self.R_V0d[k].getNextVal(self.resRV0d[k,0],self.resRV0d[k,1], IgapR)
            self.VRV0d[k,t] = self.resRV0d[k,0]

    def calcV0vPotentialsandResidues(self, t):
        for k in range (0, self.nV0v):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.LSyn_V2a_V0v[self.nV0v*l+k,0]*self.LW_V2a_V0v[l,k] for l in range (0, self.nV2a))
                IsynR = sum(self.RSyn_V2a_V0v[self.nV0v*l+k,0]*self.RW_V2a_V0v[l,k] for l in range (0, self.nV2a))

            IgapL = - sum(self.LSGap_V0v_IC[k,:]) + sum(self.LSGap_IC_V0v[:,k]) - sum(self.LSGap_V0v_V0v[k,:]) + sum(self.LSGap_V0v_V0v[:,k]) -sum(self.LSGap_V0v_MN[k,:])  + sum(self.LSGap_MN_V0v[:,k])
            IgapR = - sum(self.RSGap_V0v_IC[k,:]) + sum(self.RSGap_IC_V0v[:,k]) - sum(self.RSGap_V0v_V0v[k,:]) + sum(self.RSGap_V0v_V0v[:,k]) -sum(self.RSGap_V0v_MN[k,:])  + sum(self.RSGap_MN_V0v[:,k])

            self.resLV0v[k,:] = self.L_V0v[k].getNextVal(self.resLV0v[k,0],self.resLV0v[k,1], IgapL + IsynL)
            self.VLV0v[k,t] = self.resLV0v[k,0]
            self.resRV0v[k,:] = self.R_V0v[k].getNextVal(self.resRV0v[k,0],self.resRV0v[k,1], IgapR + IsynR)
            self.VRV0v[k,t] = self.resRV0v[k,0]

    def calcV2aPotentialsandResidues(self, t):
        for k in range (0, self.nV2a):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.RSyn_V0d_V2a[self.nV2a*l+k,0]*self.LW_V0d_V2a[l,k] for l in range (0, self.nV0d))
                IsynR = sum(self.LSyn_V0d_V2a[self.nV2a*l+k,0]*self.RW_V0d_V2a[l,k] for l in range (0, self.nV0d))


            IgapL = - sum(self.LSGap_V2a_IC[k,:]) + sum(self.LSGap_IC_V2a[:,k]) - sum(self.LSGap_V2a_V2a[k,:]) + sum(self.LSGap_V2a_V2a[:,k]) - sum(self.LSGap_V2a_MN[k,:])+ sum(self.LSGap_MN_V2a[:,k])
            IgapR = - sum(self.RSGap_V2a_IC[k,:]) + sum(self.RSGap_IC_V2a[:,k]) - sum(self.RSGap_V2a_V2a[k,:]) + sum(self.RSGap_V2a_V2a[:,k]) - sum(self.RSGap_V2a_MN[k,:])+ sum(self.RSGap_MN_V2a[:,k])

            self.resLV2a[k,:] = self.L_V2a[k].getNextVal(self.resLV2a[k,0],self.resLV2a[k,1], IgapL + IsynL)
            self.VLV2a[k,t] = self.resLV2a[k,0]
            self.resRV2a[k,:] = self.R_V2a[k].getNextVal(self.resRV2a[k,0],self.resRV2a[k,1], IgapR + IsynR)
            self.VRV2a[k,t] = self.resRV2a[k,0]

    def calcMusclePotentialsandResidues(self, t):

        for k in range (0, self.nMuscle):
            IsynL = sum(self.LSyn_MN_Muscle[self.nMuscle*l+k,0] * self.LW_MN_Muscle[l,k] for l in range (0, self.nMN))
            IsynR = sum(self.RSyn_MN_Muscle[self.nMuscle*l+k,0] * self.RW_MN_Muscle[l,k] for l in range (0, self.nMN))

            self.resLMuscle[k,:] = self.L_Muscle[k].getNextVal(self.resLMuscle[k,0], IsynL)
            self.VLMuscle[k,t] = self.resLMuscle[k,0]

            self.resRMuscle[k,:] = self.R_Muscle[k].getNextVal(self.resRMuscle[k,0], IsynR)
            self.VRMuscle[k,t] = self.resRMuscle[k,0]

    # fills in the parameters dictionary to write to a JSON file
    def getParametersDict(self):
        if (len(self.paramDict) > 0):
            return self.paramDict
        self.paramDict["dt"] = self.dt
        self.paramDict["tshutoff"] = self.tshutoff
        self.paramDict["tasyncdelay"] = self.tasyncdelay
        self.paramDict["stim0"] = self.stim0
        self.paramDict["sigma_gap"] = self.sigma_gap
        self.paramDict["sigma_chem"] = self.sigma_chem
        self.paramDict["E_glu"] = self.E_glu
        self.paramDict["E_gly"] = self.E_gly
        self.paramDict["cv"] = self.cv
        self.paramDict["nIC"] = self.nIC
        self.paramDict["nMN"] = self.nMN
        self.paramDict["nV0d"] = self.nV0d
        self.paramDict["nV0v"] = self.nV0v
        self.paramDict["nV2a"] = self.nV2a
        self.paramDict["nMuscle"] = self.nMuscle

        self.paramDict["IC_IC_gap_weight"] = self.IC_IC_gap_weight
        self.paramDict["IC_MN_gap_weight"] = self.IC_MN_gap_weight
        self.paramDict["IC_V0d_gap_weight"] = self.IC_V0d_gap_weight
        self.paramDict["IC_V0v_gap_weight"] = self.IC_V0v_gap_weight
        self.paramDict["IC_V2a_gap_weight"] = self.IC_V2a_gap_weight
        self.paramDict["MN_MN_gap_weight"] = self.MN_MN_gap_weight
        self.paramDict["MN_V0d_gap_weight"] = self.MN_V0d_gap_weight
        self.paramDict["MN_V0v_gap_weight"] = self.MN_V0v_gap_weight
        self.paramDict["MN_Muscle_syn_weight"] = self.MN_Muscle_syn_weight
        self.paramDict["V0d_MN_syn_weight"] = self.V0d_MN_syn_weight
        self.paramDict["V0d_IC_syn_weight"] = self.V0d_IC_syn_weight
        self.paramDict["V0d_V0d_gap_weight"] = self.V0d_V0d_gap_weight
        self.paramDict["V0d_V2a_syn_weight"] = self.V0d_V2a_syn_weight
        self.paramDict["V0v_IC_syn_weight"] = self.V0v_IC_syn_weight
        self.paramDict["V0v_V0v_gap_weight"] = self.V0v_V0v_gap_weight
        self.paramDict["V2a_MN_gap_weight"] = self.V2a_MN_gap_weight
        self.paramDict["V2a_V0v_syn_weight"] = self.V2a_V0v_syn_weight
        self.paramDict["V2a_V2a_gap_weight"] = self.V2a_V2a_gap_weight
        self.paramDict["rangeMin"] = self.rangeMin
        self.paramDict["rangeIC_MN"] = self.rangeIC_MN
        self.paramDict["rangeIC_MN"] = self.rangeIC_MN
        self.paramDict["rangeIC_V0d"] = self.rangeIC_V0d
        self.paramDict["rangeIC_V0v"] = self.rangeIC_V0v
        self.paramDict["rangeIC_V2a"] = self.rangeIC_V2a
        self.paramDict["rangeMN_MN"] = self.rangeMN_MN
        self.paramDict["rangeMN_V0d"] = self.rangeMN_V0d
        self.paramDict["rangeMN_V0v"] = self.rangeMN_V0v
        self.paramDict["rangeV0d_MN"] = self.rangeV0d_MN
        self.paramDict["rangeV0v_V0v"] = self.rangeV0v_V0v
        self.paramDict["rangeV0d_V2a"] = self.rangeV0d_V2a
        self.paramDict["rangeV0v_IC_min"] = self.rangeV0v_IC_min
        self.paramDict["rangeV0v_IC_max"] = self.rangeV0v_IC_max
        self.paramDict["rangeV2a_MN"] = self.rangeV2a_MN
        self.paramDict["rangeV2a_V0v_asc"] = self.rangeV2a_V0v_asc
        self.paramDict["rangeV2a_V0v_desc"] = self.rangeV2a_V0v_desc
        self.paramDict["rangeV2a_V2a"] = self.rangeV2a_V2a
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
            leftValues['IC'] = self.VLIC
            rightValues['IC'] = self.VRIC
            leftValues['V0d'] = self.VLV0d
            rightValues['V0d'] = self.VRV0d
            leftValues['V0v'] = self.VLV0v
            rightValues['V0v'] = self.VRV0v
            leftValues['V2a'] = self.VLV2a
            rightValues['V2a'] = self.VRV2a
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

        LeftValues = {'IC':self.VLIC,
            'MN': self.VLMN,
            'V0d': self.VLV0d,
            'V0v': self.VLV0v,
            'V2a': self.VLV2a,
            'Muscle': self.VLMuscle}
        RightValues = {'IC':self.VRIC,
            'MN': self.VRMN,
            'V0d': self.VRV0d,
            'V0v': self.VRV0v,
            'V2a': self.VRV2a,
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

        saveAnimation(filename = filename, nMuscle = self.nMuscle, VLMuscle = self.VLMuscle, VRMuscle = self.VRMuscle, Time = self.Time, dt = self.dt)

    def stripOffsetRegion(self, tskip):
        ## Removing the first "tskip" milliseconds to let the initial conditions dissipate
        index_offset = int(tskip/self.dt)

        #the removed data are saved in a dictionary to prevent data loss - in case they will be used in a different analysis
        self.skippedVLists['VLIC'] = self.VLIC[:,:index_offset]
        self.skippedVLists['VRIC'] = self.VRIC[:,:index_offset]
        self.skippedVLists['VLMN'] = self.VLMN[:,:index_offset]
        self.skippedVLists['VRMN'] = self.VRMN[:,:index_offset]
        self.skippedVLists['VLV0d'] = self.VLV0d[:,:index_offset]
        self.skippedVLists['VRV0d'] = self.VRV0d[:,:index_offset]
        self.skippedVLists['VLV0v'] = self.VLV0v[:,:index_offset]
        self.skippedVLists['VRV0v'] = self.VRV0v[:,:index_offset]
        self.skippedVLists['VLV2a'] = self.VLV2a[:,:index_offset]
        self.skippedVLists['VRV2a'] = self.VRV2a[:,:index_offset]
        self.skippedVLists['VLMuscle'] = self.VLMuscle[:,:index_offset]
        self.skippedVLists['VRMuscle'] = self.VRMuscle[:,:index_offset]

        self.VLIC = self.VLIC[:,index_offset:]
        self.VRIC = self.VRIC[:,index_offset:]

        self.VLMN = self.VLMN[:,index_offset:]
        self.VRMN = self.VRMN[:,index_offset:]

        self.VLV0d = self.VLV0d[:,index_offset:]
        self.VRV0d = self.VRV0d[:,index_offset:]

        self.VLV0v = self.VLV0v[:,index_offset:]
        self.VRV0v = self.VRV0v[:,index_offset:]

        self.VLV2a = self.VLV2a[:,index_offset:]
        self.VRV2a = self.VRV2a[:,index_offset:]

        self.VLMuscle = self.VLMuscle[:,index_offset:]
        self.VRMuscle = self.VRMuscle[:,index_offset:]

        self.Time = self.Time[index_offset:] - self.Time[index_offset:][0]

    # This function sets up the connectome and runs a simulation.
    # rand is a seed for a random function
    # tmax: the time period the simulation will be run for
    # tskip: the initial time period that is added to the simulation to cover the initial conditions to dissipate
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
    def mainLoop(self, rand, tmax = 1000, tskip = 200, tplot_interval = 1000, plotProgressOnly = False, plotAllPotentials = False, printParam = False, plotResult = False, saveCSV = False, saveAnim = False):

        self.paramDict = dict() #initialize paramDict, will be filled only once per run
        if printParam:
            self.printParameters()

        seed(rand)
        nmax = (int) ((tmax + tskip) / self.dt)
        self.initializeStructures(nmax)

        self.stim = zeros(nmax)
        self.computeSynapticAndGapWeights()
        self.initializeMembranePotentials()

        ## This loop is the main loop where we solve the ordinary differential equations at every time point
        tlastplot = 0
        for t in range (0, nmax):
            self.Time[t] = self.dt * t

            # Generate plots to visualize the progself.ress of the simulations
            if not(self.Time[t] % tplot_interval) and (self.Time[t]>20):
                self.plotProgress(t, tstart = tlastplot, plotall = plotAllPotentials)
                tlastplot = t if plotProgressOnly else 0

            #Amplitude of tonic drive to IC neurons
            self.stim[t] = self.getStimulus()

            ## Calculate synaptic inputs
            self.calcSynapticOutputs(t)
            self.calcICOutputs(t)
            self.calcMNOutputs(t)
            self.calcV0dOutputs(t)
            self.calcV0vOutputs(t)
            self.calcV2aOutputs(t)

            self.calcMembranePotentialsFromCurrents(t)

        self.stripOffsetRegion(tskip)

        self.model_run = True

        if plotResult:
            self.plotProgress(int(tmax/self.dt) - 1, tstart = 0, plotall = plotAllPotentials)
        if saveCSV:
            self.saveToFile()
        if saveAnim:
            self.saveAnimation()

        return (self.VLIC, self.VRIC), (self.VLMN, self.VRMN), (self.VLV0d, self.VRV0d), \
            (self.VLV0v, self.VRV0v), (self.VLV2a, self.VRV2a), (self.VLMuscle, self.VRMuscle), self.Time

    #endregion
