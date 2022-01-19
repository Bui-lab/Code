#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017
@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from Double_coiling_model import Double_coil_base

class Double_coil_V0v_to_IC_null(Double_coil_base):

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0, E_glu = 0, E_gly = -70,
                  cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nV0v = 10, nV2a = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigma, E_glu, E_gly,
                  cv, nIC, nMN, nV0d, nV0v, nV2a, nMuscle)


    def calcICPotentialsandResidues(self, t):
        for k in range (0, self.nIC):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else: # V0v to contralateral IC synapse removed from Isyn calculations
                IsynL = sum(self.RSyn_V0d_IC[self.nIC*l+k,0]*self.LW_V0d_IC[l,k] for l in range (0, self.nV0d))
                IsynR = sum(self.LSyn_V0d_IC[self.nIC*l+k,0]*self.RW_V0d_IC[l,k] for l in range (0, self.nV0d))
            
            if t < self.tasyncdelay: #to ensure that double coils are not due to coincident coiling 
                right_side_delay=0
            else:
                right_side_delay=1

            self.resLIC[k,:] = self.L_IC[k].getNextVal(self.resLIC[k,0],self.resLIC[k,1], self.stim[t] + sum(self.LSGap_IC_IC[:,k]) - sum(self.LSGap_IC_IC[k,:]) + sum(self.LSGap_MN_IC[:,k]) - sum(self.LSGap_IC_MN[k,:]) + sum(self.LSGap_V0d_IC[:,k]) - sum(self.LSGap_IC_V0d[k,:]) + sum(self.LSGap_V0v_IC[:,k]) - sum(self.LSGap_IC_V0v[k,:]) + sum(self.LSGap_V2a_IC[:,k]) - sum(self.LSGap_IC_V2a[k,:]) + IsynL)
            self.VLIC[k,t] = self.resLIC[k,0]
        
            self.resRIC[k,:] = self.R_IC[k].getNextVal(self.resRIC[k,0],self.resRIC[k,1], self.stim[t]*right_side_delay + sum(self.RSGap_IC_IC[:,k])  - sum(self.RSGap_IC_IC[k,:])  + sum(self.RSGap_MN_IC[:,k]) - sum(self.RSGap_IC_MN[k,:]) + sum(self.RSGap_V0d_IC[:,k]) - sum(self.RSGap_IC_V0d[k,:])  + sum(self.RSGap_V0v_IC[:,k]) - sum(self.RSGap_IC_V0v[k,:]) + sum(self.RSGap_V2a_IC[:,k]) - sum(self.RSGap_IC_V2a[k,:]) + IsynR)
            self.VRIC[k,t] = self.resRIC[k,0]
