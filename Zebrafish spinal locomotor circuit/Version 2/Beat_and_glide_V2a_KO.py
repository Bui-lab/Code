#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from random import gauss
from Beat_and_glide import Beat_and_glide_base

class Beat_and_glide_V2a_KO(Beat_and_glide_base):

    __tV2aKOstart = None #calculated at init
    __tV2aKOend = None #calculated at init
    _tshutoff = None

    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                          R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
        self.setTimeParameters() #to nitialize with default values

    def setTimeParameters(self, tmax_ms = 10000, tshutoff_ms = 50, tskip_ms = 1000, dt = 0.1, tV2aKOstart_ms = 6000, tV2aKOend_ms = 11000):
        super().setTimeParameters(tmax_ms, tshutoff_ms, tskip_ms, dt)
        self.__tV2aKOstart = (tV2aKOstart_ms + tskip_ms) / dt
        self.__tV2aKOend = (tV2aKOend_ms + tskip_ms) / dt
        self._tshutoff = super().gettShutOff() #to prevent multiple function calls

    def calcV2aPotentialsandResidues(self, t):
        for k in range (0, self.nV2a):
            if t < self._tshutoff: #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL = 0.0
                IsynR = 0.0
                IGapL = - sum(self.LSGap_V2a_V2a[k,:]) + sum(self.LSGap_V2a_V2a[:,k]) - sum(self.LSGap_V2a_MN[k,:])+ sum(self.LSGap_MN_V2a[:,k])
                IGapR = - sum(self.RSGap_V2a_V2a[k,:]) + sum(self.RSGap_V2a_V2a[:,k]) - sum(self.RSGap_V2a_MN[k,:])+ sum(self.RSGap_MN_V2a[:,k])
            elif (t > self.__tV2aKOstart and t < self.__tV2aKOend):
                IsynL = 0.0
                IsynR = 0.0
                IGapL = 0.0
                IGapR = 0.0
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
                IGapR = - sum(self.RSGap_V2a_V2a[k,:]) + sum(self.RSGap_V2a_V2a[:,k]) - sum(self.RSGap_V2a_MN[k,:])+ sum(self.RSGap_MN_V2a[:,k])

            self.resLV2a[k,:] = self.L_V2a[k].getNextVal(self.resLV2a[k,0], self.resLV2a[k,1], IGapL + IsynL)         
            self.VLV2a[k,t] = self.resLV2a[k,0]
            self.resRV2a[k,:] = self.R_V2a[k].getNextVal(self.resRV2a[k,0], self.resRV2a[k,1], IGapR + IsynR)    
            self.VRV2a[k,t] = self.resRV2a[k,0]

        
