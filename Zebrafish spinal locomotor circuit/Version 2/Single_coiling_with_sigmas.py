#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from random import gauss

from Izhikevich_class import Izhikevich_9P, Leaky_Integrator
from Single_coiling_model import Single_coil_base

class Single_coil_sigmas(Single_coil_base):

    sigmaD = 0 # Sets variability of tonic command drive
    sigmaL = 0 # Sets variability of length of axons
    sigmaP = 0 # Sets variability of cell parameters for membrane potential dynamics
    sigmaW = 0 # Sets variability of synaptic weight

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0, sigmaD = 0, sigmaL = 0, sigmaP = 0, sigmaW = 0,
                  E_glu = 0, E_gly = -70, cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigma,
                       E_glu, E_gly, cv, nIC, nMN, nV0d, nMuscle)

        self.sigmaD = sigmaD
        self.sigmaL = sigmaL
        self.sigmaP = sigmaP
        self.sigmaW = sigmaW

        super().setWeightParameters(IC_IC_gap_weight = 0.001, IC_MN_gap_weight = 0.04, IC_V0d_gap_weight = 0.05,
                                    MN_MN_gap_weight = 0.1, V0d_V0d_gap_weight = 0.04, MN_V0d_gap_weight = 0.01,
                                    V0d_MN_syn_weight = 2.0, V0d_IC_syn_weight = 2.0, MN_Muscle_syn_weight = 0.015)
        super().setRangeParameters(rangeMin = 0.2, rangeIC_MN = 10, rangeIC_V0d = 10, rangeMN_MN = 6.5, rangeV0d_V0d = 3.5,
                           rangeMN_V0d = 1.5, rangeV0d_MN = 8, rangeV0d_IC = 20, rangeMN_Muscle = 1)

        self.tasyncdelay = 32000

    def initNeurons(self):
        #adds noise to the Izhikevich parameters
        self.L_IC = [Izhikevich_9P(a = 0.0005 * gauss(1, self.sigmaP),
                                   b = 0.5 * gauss(1, self.sigmaP),
                                   c = -30 * gauss(1, self.sigmaP),
                                   d = 5 * gauss(1, self.sigmaP),
                                   vmax = 0 * gauss(1, self.sigmaP),
                                   vr = -60 * gauss(1, self.sigmaP),
                                   vt = -45 * gauss(1, self.sigmaP),
                                   k = 0.05 * gauss(1, self.sigmaP),
                                   Cm = 50 * gauss(1, self.sigmaP),
                                   dt = self.dt,
                                   x = 1.0 * gauss(1, self.sigma),
                                   y = -1)
                     for i in range(self.nIC)]
        self.R_IC = [Izhikevich_9P(a = 0.0005 * gauss(1,self.sigmaP),
                                   b = 0.5 * gauss(1,self.sigmaP),
                                   c = -30 * gauss(1,self.sigmaP),
                                   d = 5 * gauss(1,self.sigmaP),
                                   vmax = 0 * gauss(1,self.sigmaP),
                                   vr = -60 * gauss(1,self.sigmaP),
                                   vt = -45 * gauss(1,self.sigmaP),
                                   k = 0.05 * gauss(1,self.sigmaP),
                                   Cm = 50 * gauss(1,self.sigmaP),
                                   dt = self.dt,
                                   x = 1.0 * gauss(1, self.sigma),
                                   y = 1)
                     for i in range(self.nIC)]

        self.L_MN = [ Izhikevich_9P(a = 0.5 * gauss(1,self.sigmaP),
                                    b = 0.1 * gauss(1,self.sigmaP),
                                    c = -50 * gauss(1,self.sigmaP),
                                    d = 0.2 * gauss(1,self.sigmaP),
                                    vmax = 10 * gauss(1,self.sigmaP),
                                    vr = -60 * gauss(1,self.sigmaP),
                                    vt = -45 * gauss(1,self.sigmaP),
                                    k = 0.05 * gauss(1,self.sigmaP),
                                    Cm = 20 * gauss(1,self.sigmaP),
                                    dt=self.dt,
                                    x = 5.0 + 1.6 * i * gauss(1, self.sigma),
                                    y = -1)
                     for i in range(self.nMN)]
        self.R_MN = [ Izhikevich_9P(a = 0.5 * gauss(1,self.sigmaP),
                                    b = 0.1 * gauss(1,self.sigmaP),
                                    c = -50 * gauss(1,self.sigmaP),
                                    d = 0.2 * gauss(1,self.sigmaP),
                                    vmax=10 * gauss(1,self.sigmaP),
                                    vr=-60 * gauss(1,self.sigmaP),
                                    vt=-45 * gauss(1,self.sigmaP),
                                    k = 0.05 * gauss(1,self.sigmaP),
                                    Cm = 20 * gauss(1,self.sigmaP),
                                    dt = self.dt,
                                    x = 5.0 + 1.6 * i * gauss(1, self.sigma),
                                    y = 1)
                     for i in range(self.nMN)]

        self.L_V0d = [ Izhikevich_9P(a = 0.5 * gauss(1,self.sigmaP),
                                     b = 0.01 * gauss(1,self.sigmaP),
                                     c = -50 * gauss(1,self.sigmaP),
                                     d = 0.2 * gauss(1,self.sigmaP),
                                     vmax = 10 * gauss(1,self.sigmaP),
                                     vr = -60 * gauss(1,self.sigmaP),
                                     vt = -45 * gauss(1,self.sigmaP),
                                     k = 0.05 * gauss(1,self.sigmaP),
                                     Cm = 20 * gauss(1,self.sigmaP),
                                     dt = self.dt,
                                     x = 5.0 + 1.6 * i * gauss(1, self.sigma),
                                     y = -1)
                      for i in range(self.nV0d)]
        self.R_V0d = [ Izhikevich_9P(a = 0.5 * gauss(1,self.sigmaP),
                                     b = 0.01 * gauss(1,self.sigmaP),
                                     c = -50 * gauss(1,self.sigmaP),
                                     d = 0.2 * gauss(1,self.sigmaP),
                                     vmax = 10 * gauss(1,self.sigmaP),
                                     vr = -60 * gauss(1,self.sigmaP),
                                     vt = -45 * gauss(1,self.sigmaP),
                                     k = 0.05 * gauss(1,self.sigmaP),
                                     Cm = 20 * gauss(1,self.sigmaP),
                                     dt = self.dt,
                                     x = 5.0 + 1.6 * i * gauss(1, self.sigma),
                                     y = 1)
                      for i in range(self.nV0d)]

        self.L_Muscle = [ Leaky_Integrator(25.0, 10.0, self.dt, 5.0+1.6*i,-1)
                         for i in range(self.nMuscle)]
        self.R_Muscle = [ Leaky_Integrator(25.0, 10.0, self.dt, 5.0+1.6*i, 1)
                         for i in range(self.nMuscle)]

    #Called from computeSynapticAndGapWeight to include gaussian noise
    def rangeNoiseMultiplier(self):
        return gauss(1, self.sigmaL)

    #Called from computeSynapticAndGapWeight to include gaussian noise
    def weightNoiseMultiplier(self):
        return gauss(1, self.sigmaW)


    def calcICPotentialsandResidues(self, t):
        ## Determine membrane potentials from synaptic and external currents
        for k in range (0, self.nIC):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.RSyn_V0d_IC[self.nIC*l+k,0]*self.LW_V0d_IC[l,k] for l in range (0, self.nV0d))
                IsynR = sum(self.LSyn_V0d_IC[self.nIC*l+k,0]*self.RW_V0d_IC[l,k] for l in range (0, self.nV0d))

            IgapL = sum(self.LSGap_IC_IC[:,k]) - sum(self.LSGap_IC_IC[k,:]) + sum(self.LSGap_MN_IC[:,k]) - sum(self.LSGap_IC_MN[k,:]) + sum(self.LSGap_V0d_IC[:,k]) - sum(self.LSGap_IC_V0d[k,:])
            IgapR = sum(self.RSGap_IC_IC[:,k]) - sum(self.RSGap_IC_IC[k,:]) + sum(self.RSGap_MN_IC[:,k]) - sum(self.RSGap_IC_MN[k,:]) + sum(self.RSGap_V0d_IC[:,k]) - sum(self.RSGap_IC_V0d[k,:])

            self.resLIC[k,:] = self.L_IC[k].getNextVal(self.resLIC[k,0], self.resLIC[k,1], self.stim[t]*gauss(1.0, self.sigmaD) + IgapL + IsynL)
            self.VLIC[k,t] = self.resLIC[k,0]

            if t < self.tasyncdelay: # This is to delay the stim to right ICs so that they don't fire synchronously
                right_ON = 0
            else:
                right_ON = 1 #set to 0 if you want to prevent right side coiling
            
            self.resRIC[k,:] = self.R_IC[k].getNextVal(self.resRIC[k,0], self.resRIC[k,1], self.stim[t] * right_ON + IgapR + IsynR)
            self.VRIC[k,t] = self.resRIC[k,0]

