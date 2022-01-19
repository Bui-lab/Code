#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from Single_coiling_model import Single_coil_base

class Single_coil_V0d_KO(Single_coil_base):
    tV0dKOstart = 50000
    tV0dKOend = 100000

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0,
                 E_glu = 0, E_gly = -70, cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigma,
                       E_glu, E_gly, cv, nIC, nMN, nV0d, nMuscle)
        super().setWeightParameters(IC_IC_gap_weight = 0.001, IC_MN_gap_weight = 0.04, IC_V0d_gap_weight = 0.05,
                                    MN_MN_gap_weight = 0.1, V0d_V0d_gap_weight = 0.04, MN_V0d_gap_weight = 0.01,
                                    V0d_MN_syn_weight = 2.0, V0d_IC_syn_weight = 2.0, MN_Muscle_syn_weight = 0.015)
        super().setRangeParameters(rangeMin = 0.2, rangeIC_MN = 10, rangeIC_V0d = 10, rangeMN_MN = 6.5, rangeV0d_V0d = 3.5,
                           rangeMN_V0d = 1.5, rangeV0d_MN = 8, rangeV0d_IC = 20, rangeMN_Muscle = 1)


    def calcV0dPotentialsandResidues(self, t):
        for k in range (0, self.nV0d):
            if t > self.tV0dKOstart and t < self.tV0dKOend:
                self.resLV0d[k,:] = self.L_V0d[k].getNextVal(self.resLV0d[k,0],self.resLV0d[k,1], 0)
                self.resRV0d[k,:] = self.R_V0d[k].getNextVal(self.resRV0d[k,0],self.resRV0d[k,1], 0)
            else:
                IgapL = - sum(self.LSGap_V0d_IC[k,:]) + sum(self.LSGap_IC_V0d[:,k]) - sum(self.LSGap_V0d_V0d[k,:]) + sum(self.LSGap_V0d_V0d[:,k]) - sum(self.LSGap_V0d_MN[k,:]) + sum(self.LSGap_MN_V0d[:,k])
                IgapR = - sum(self.RSGap_V0d_IC[k,:]) + sum(self.RSGap_IC_V0d[:,k]) - sum(self.RSGap_V0d_V0d[k,:]) + sum(self.RSGap_V0d_V0d[:,k]) - sum(self.RSGap_V0d_MN[k,:]) + sum(self.RSGap_MN_V0d[:,k])
                
                self.resLV0d[k,:] = self.L_V0d[k].getNextVal(self.resLV0d[k,0],self.resLV0d[k,1], IgapL)
                self.resRV0d[k,:] = self.R_V0d[k].getNextVal(self.resRV0d[k,0],self.resRV0d[k,1], IgapR)

            self.VLV0d[k,t] = self.resLV0d[k,0]
            self.VRV0d[k,t] = self.resRV0d[k,0]


