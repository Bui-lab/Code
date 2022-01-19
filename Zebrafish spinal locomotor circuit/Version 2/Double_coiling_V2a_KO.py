#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from Double_coiling_model import Double_coil_base

class Double_coil_V2a_KO(Double_coil_base):

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0, E_glu = 0, E_gly = -70,
                  cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nV0v = 10, nV2a = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigma, E_glu, E_gly,
                  cv, nIC, nMN, nV0d, nV0v, nV2a, nMuscle)


    def calcV0vPotentialsandResidues(self, t):
        for k in range (0, self.nV0v):
            self.resLV0v[k,:] = self.L_V0v[k].getNextVal(self.resLV0v[k,0],self.resLV0v[k,1], - sum(self.LSGap_V0v_IC[k,:]) + sum(self.LSGap_IC_V0v[:,k]) - sum(self.LSGap_V0v_V0v[k,:]) + sum(self.LSGap_V0v_V0v[:,k]) -sum(self.LSGap_V0v_MN[k,:])  + sum(self.LSGap_MN_V0v[:,k]))
            self.VLV0v[k,t] = self.resLV0v[k,0]
            self.resRV0v[k,:] = self.R_V0v[k].getNextVal(self.resRV0v[k,0],self.resRV0v[k,1], - sum(self.RSGap_V0v_IC[k,:]) + sum(self.RSGap_IC_V0v[:,k]) - sum(self.RSGap_V0v_V0v[k,:]) + sum(self.RSGap_V0v_V0v[:,k]) -sum(self.RSGap_V0v_MN[k,:])  + sum(self.RSGap_MN_V0v[:,k]))
            self.VRV0v[k,t] = self.resRV0v[k,0]
    
