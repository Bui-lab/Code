
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from Beat_and_glide import Beat_and_glide_base

class Beat_and_glide_V1_KO(Beat_and_glide_base):

    __tV1KOstart = None #calculated at init
    __tV1KOend = None #calculated at init
    _tshutoff = None

    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                          R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
        self.setTimeParameters() #to nitialize with default values

    def setTimeParameters(self, tmax_ms = 1000, tshutoff_ms = 50, tskip_ms = 1000, dt = 0.1, tV1KOstart_ms = 6000, tV1KOend_ms = 11000):
        super().setTimeParameters(tmax_ms, tshutoff_ms, tskip_ms, dt)
        self.__tV1KOstart = (tV1KOstart_ms + tskip_ms) / dt
        self.__tV1KOend = (tV1KOend_ms + tskip_ms) / dt
        self._tshutoff = super().gettShutOff() #to prevent multiple function calls

    def calcV1PotentialsandResidues(self, t):
        
        for k in range (0, self.nV1):
            #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
            if (t < self._tshutoff) or (t > self.__tV1KOstart and t < self.__tV1KOend): 
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.LSyn_V2a_V1[self.nV1*m+k,0] * self.LW_V2a_V1[m,k] for m in range (0, self.nV2a))
                IsynR = sum(self.RSyn_V2a_V1[self.nV1*m+k,0] * self.RW_V2a_V1[m,k] for m in range (0, self.nV2a))
            self.resLV1[k,:] = self.L_V1[k].getNextVal(self.resLV1[k,0], self.resLV1[k,1], IsynL)  
            self.VLV1[k,t] = self.resLV1[k,0]
            self.resRV1[k,:] = self.R_V1[k].getNextVal(self.resRV1[k,0], self.resRV1[k,1], IsynR)  
            self.VRV1[k,t] = self.resRV1[k,0]
            
