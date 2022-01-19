#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from random import gauss
from Beat_and_glide import Beat_and_glide_base

class Beat_and_glide_dI6_KO(Beat_and_glide_base):

    __tdI6KOstart = None #calculated at init
    __tdI6KOend = None #calculated at init
    _tshutoff = None

    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                          R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
        self.setTimeParameters() #to nitialize with default values

    def setTimeParameters(self, tmax_ms = 10000, tshutoff_ms = 50, tskip_ms = 1000, dt = 0.1, tdI6KOstart_ms = 6000, tdI6KOend_ms = 11000):
        super().setTimeParameters(tmax_ms, tshutoff_ms, tskip_ms, dt)
        self.__tdI6KOstart = (tdI6KOstart_ms + tskip_ms) / dt
        self.__tdI6KOend = (tdI6KOend_ms + tskip_ms) / dt
        self._tshutoff = super().gettShutOff() #to prevent multiple function calls

    def calcdI6PotentialsandResidues(self, t):
        for k in range (0, self.ndI6):
            if t < self._tshutoff: 
                IsynL = 0.0
                IsynR = 0.0
                IGapL = - sum(self.LSGap_dI6_dI6[k,:]) + sum(self.LSGap_dI6_dI6[:,k]) - sum(self.LSGap_dI6_MN[k,:]) + sum(self.LSGap_MN_dI6[:,k])
                IGapR = - sum(self.RSGap_dI6_dI6[k,:]) + sum(self.RSGap_dI6_dI6[:,k]) - sum(self.RSGap_dI6_MN[k,:]) + sum(self.RSGap_MN_dI6[:,k])
            elif (t > self.__tdI6KOstart and t < self.__tdI6KOend):
                IsynL = 0.0
                IsynR = 0.0
                IGapL = 0.0
                IGapR = 0.0
            else:
                IsynL = sum(self.LSyn_V2a_dI6[self.ndI6*l+k,0]*self.LW_V2a_dI6[l,k] for l in range (0, self.nV2a)) + sum(self.RSyn_dI6_dI6[self.ndI6*l+k,0]*self.LW_dI6_dI6[l,k]*self.R_str for l in range (0, self.ndI6)) + sum(self.LSyn_V1_dI6[self.ndI6*l+k,0]*self.LW_V1_dI6[l,k]*self.R_str for l in range (0, self.nV1))
                IsynR = sum(self.RSyn_V2a_dI6[self.ndI6*l+k,0]*self.RW_V2a_dI6[l,k] for l in range (0, self.nV2a)) + sum(self.LSyn_dI6_dI6[self.ndI6*l+k,0]*self.RW_dI6_dI6[l,k]*self.R_str for l in range (0, self.ndI6)) + sum(self.RSyn_V1_dI6[self.ndI6*l+k,0]*self.RW_V1_dI6[l,k]*self.R_str for l in range (0, self.nV1))
                IGapL = - sum(self.LSGap_dI6_dI6[k,:]) + sum(self.LSGap_dI6_dI6[:,k]) - sum(self.LSGap_dI6_MN[k,:]) + sum(self.LSGap_MN_dI6[:,k])
                IGapR = - sum(self.RSGap_dI6_dI6[k,:]) + sum(self.RSGap_dI6_dI6[:,k]) - sum(self.RSGap_dI6_MN[k,:]) + sum(self.RSGap_MN_dI6[:,k])

            self.resLdI6[k,:] = self.L_dI6[k].getNextVal(self.resLdI6[k,0],self.resLdI6[k,1], IGapL + IsynL)
            self.VLdI6[k,t] = self.resLdI6[k,0]
            self.resRdI6[k,:] = self.R_dI6[k].getNextVal(self.resRdI6[k,0],self.resRdI6[k,1], IGapR + IsynR)
            self.VRdI6[k,t] = self.resRdI6[k,0]
