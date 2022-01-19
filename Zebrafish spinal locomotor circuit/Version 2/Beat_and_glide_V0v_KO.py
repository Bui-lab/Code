#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from random import gauss
from Beat_and_glide import Beat_and_glide_base

class Beat_and_glide_V0v_KO(Beat_and_glide_base):

    __tV0vKOstart = None #calculated at init
    __tV0vKOend = None #calculated at init
    _tshutoff = None
    
    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                          R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
        self.setTimeParameters() #to nitialize with default values

    def setTimeParameters(self, tmax_ms = 10000, tshutoff_ms = 50, tskip_ms = 1000, dt = 0.1, tV0vKOstart_ms = 6000, tV0vKOend_ms = 11000):
        super().setTimeParameters(tmax_ms, tshutoff_ms, tskip_ms, dt)
        self.__tV0vKOstart = (tV0vKOstart_ms + tskip_ms) / dt
        self.__tV0vKOend = (tV0vKOend_ms + tskip_ms) / dt
        self._tshutoff = super().gettShutOff() #to prevent multiple function calls


    def calcV0vPotentialsandResidues(self, t):
        for k in range (0, self.nV0v):
            if t < self._tshutoff: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL = 0.0
                IsynR = 0.0
                IGapL = - sum(self.LSGap_V0v_V0v[k,:]) + sum(self.LSGap_V0v_V0v[:,k]) -sum(self.LSGap_V0v_MN[k,:])  + sum(self.LSGap_MN_V0v[:,k])
                IGapR = - sum(self.RSGap_V0v_V0v[k,:]) + sum(self.RSGap_V0v_V0v[:,k]) -sum(self.RSGap_V0v_MN[k,:])  + sum(self.RSGap_MN_V0v[:,k])
            elif (t > self.__tV0vKOstart and t < self.__tV0vKOend):
                IsynL = 0.0
                IsynR = 0.0
                IGapL = 0.0
                IGapR = 0.0
            else:
                IsynL = sum(self.LSyn_V2a_V0v[self.nV0v*l+k,0]*self.LW_V2a_V0v[l,k] for l in range (0, self.nV2a)) + sum(self.LSyn_V1_V0v[self.nV0v*l+k,0]*self.LW_V1_V0v[l,k]*self.R_str for l in range (0, self.nV1))
                IsynR = sum(self.RSyn_V2a_V0v[self.nV0v*l+k,0]*self.RW_V2a_V0v[l,k] for l in range (0, self.nV2a)) + sum(self.RSyn_V1_V0v[self.nV0v*l+k,0]*self.RW_V1_V0v[l,k]*self.R_str for l in range (0, self.nV1))
                IGapL = - sum(self.LSGap_V0v_V0v[k,:]) + sum(self.LSGap_V0v_V0v[:,k]) -sum(self.LSGap_V0v_MN[k,:])  + sum(self.LSGap_MN_V0v[:,k])
                IGapR = - sum(self.RSGap_V0v_V0v[k,:]) + sum(self.RSGap_V0v_V0v[:,k]) -sum(self.RSGap_V0v_MN[k,:])  + sum(self.RSGap_MN_V0v[:,k])

            self.resLV0v[k,:] = self.L_V0v[k].getNextVal(self.resLV0v[k,0], self.resLV0v[k,1], IGapL + IsynL)
            self.VLV0v[k,t] = self.resLV0v[k,0]
            self.resRV0v[k,:] = self.R_V0v[k].getNextVal(self.resRV0v[k,0], self.resRV0v[k,1], IGapR + IsynR)
            self.VRV0v[k,t] = self.resRV0v[k,0]
