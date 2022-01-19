#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from Single_coiling_model import Single_coil_base

class Single_coil_glycine_null(Single_coil_base):

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0,
                 E_glu = 0, E_gly = -70, cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigma,
                       E_glu, E_gly, cv, nIC, nMN, nV0d, nMuscle)
        super().setWeightParameters(IC_IC_gap_weight = 0.001, IC_MN_gap_weight = 0.04, IC_V0d_gap_weight = 0.05,
                                    MN_MN_gap_weight = 0.1, V0d_V0d_gap_weight = 0.04, MN_V0d_gap_weight = 0.01,
                                    V0d_MN_syn_weight = 2.0, V0d_IC_syn_weight = 2.0, MN_Muscle_syn_weight = 0.015)
        super().setRangeParameters(rangeMin = 0.2, rangeIC_MN = 10, rangeIC_V0d = 10, rangeMN_MN = 6.5, rangeV0d_V0d = 3.5,
                           rangeMN_V0d = 1.5, rangeV0d_MN = 8, rangeV0d_IC = 20, rangeMN_Muscle = 1)

    def calcICPotentialsandResidues(self, t):
        for k in range (0, self.nIC):
            IsynL= 0.0
            IsynR= 0.0

            self.resLIC[k,:] = self.L_IC[k].getNextVal(self.resLIC[k,0], self.resLIC[k,1], self.stim[t] + sum(self.LSGap_IC_IC[:,k]) - sum(self.LSGap_IC_IC[k,:]) + sum(self.LSGap_MN_IC[:,k]) - sum(self.LSGap_IC_MN[k,:]) + sum(self.LSGap_V0d_IC[:,k]) - sum(self.LSGap_IC_V0d[k,:]) + IsynL)
            self.VLIC[k,t] = self.resLIC[k,0]

            right_ON = 0

            self.resRIC[k,:] = self.R_IC[k].getNextVal(self.resRIC[k,0],self.resRIC[k,1], self.stim[t]*right_ON + sum(self.RSGap_IC_IC[:,k])  - sum(self.RSGap_IC_IC[k,:])  + sum(self.RSGap_MN_IC[:,k]) - sum(self.RSGap_IC_MN[k,:]) + sum(self.RSGap_V0d_IC[:,k]) - sum(self.RSGap_IC_V0d[k,:]) + IsynR)
            self.VRIC[k,t] = self.resRIC[k,0]


    def calcMNPotentialsandResidues(self, t):
        for k in range (0, self.nMN):
            IsynL= 0.0
            IsynR= 0.0
            #if k == 5: # this is to hyperpolarize a MN to observe periodic depolarizations and synaptic bursts
            #   IsynL = IsynL - 10

            self.resLMN[k,:] = self.L_MN[k].getNextVal(self.resLMN[k,0],self.resLMN[k,1], - sum(self.LSGap_MN_IC[k,:]) + sum(self.LSGap_IC_MN[:,k]) - sum(self.LSGap_MN_MN[k,:]) + sum(self.LSGap_MN_MN[:,k]) - sum(self.LSGap_MN_V0d[k,:]) + sum(self.LSGap_V0d_MN[:,k]) + IsynL)
            self.VLMN[k,t] = self.resLMN[k,0]

            self.resRMN[k,:] = self.R_MN[k].getNextVal(self.resRMN[k,0],self.resRMN[k,1], - sum(self.RSGap_MN_IC[k,:]) + sum(self.RSGap_IC_MN[:,k]) - sum(self.RSGap_MN_MN[k,:]) + sum(self.RSGap_MN_MN[:,k]) - sum(self.RSGap_MN_V0d[k,:]) + sum(self.RSGap_V0d_MN[:,k]) + IsynR)
            self.VRMN[k,t] = self.resRMN[k,0]

