#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017
@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from Double_coiling_model import Double_coil_base

class Double_coil_no_excitatory_syn(Double_coil_base):

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0, E_glu = 0, E_gly = -70,
                  cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nV0v = 10, nV2a = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigma, E_glu, E_gly,
                  cv, nIC, nMN, nV0d, nV0v, nV2a, nMuscle)


    def calcICPotentialsandResidues(self, t):
        for k in range (0, self.nIC):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else: #input from V0v is removed
                IsynL = sum(self.RSyn_V0d_IC[self.nIC*l+k,0]*self.LW_V0d_IC[l,k] for l in range (0, self.nV0d))
                IsynR = sum(self.LSyn_V0d_IC[self.nIC*l+k,0]*self.RW_V0d_IC[l,k] for l in range (0, self.nV0d))
            
            if t < self.tasyncdelay: #to ensure that double coils are not due to coincident coiling 
                right_side_delay=0
            else:
                right_side_delay=1

            IgapL = sum(self.LSGap_IC_IC[:,k]) - sum(self.LSGap_IC_IC[k,:]) + sum(self.LSGap_MN_IC[:,k]) - sum(self.LSGap_IC_MN[k,:]) + sum(self.LSGap_V0d_IC[:,k]) - sum(self.LSGap_IC_V0d[k,:]) + sum(self.LSGap_V0v_IC[:,k]) - sum(self.LSGap_IC_V0v[k,:]) + sum(self.LSGap_V2a_IC[:,k]) - sum(self.LSGap_IC_V2a[k,:])
            IgapR = sum(self.RSGap_IC_IC[:,k]) - sum(self.RSGap_IC_IC[k,:]) + sum(self.RSGap_MN_IC[:,k]) - sum(self.RSGap_IC_MN[k,:]) + sum(self.RSGap_V0d_IC[:,k]) - sum(self.RSGap_IC_V0d[k,:])  + sum(self.RSGap_V0v_IC[:,k]) - sum(self.RSGap_IC_V0v[k,:]) + sum(self.RSGap_V2a_IC[:,k]) - sum(self.RSGap_IC_V2a[k,:])

            self.resLIC[k,:] = self.L_IC[k].getNextVal(self.resLIC[k,0],self.resLIC[k,1], self.stim[t] + IgapL + IsynL)
            self.VLIC[k,t] = self.resLIC[k,0]
        
            self.resRIC[k,:] = self.R_IC[k].getNextVal(self.resRIC[k,0],self.resRIC[k,1], self.stim[t] * right_side_delay + IgapR + IsynR)
            self.VRIC[k,t] = self.resRIC[k,0]

    def calcMNPotentialsandResidues(self, t):
        for k in range (0, self.nMN):
            if t < (self.tshutoff/self.dt): #Synaptic currents are shut off for the first 50 ms of the sims to let initial conditions subside
                IsynL= 0.0
                IsynR= 0.0
            else:
                IsynL = sum(self.RSyn_V0d_MN[self.nMN*l+k,0]*self.LW_V0d_MN[l,k] for l in range (0, self.nV0d)) 
                IsynR = sum(self.LSyn_V0d_MN[self.nMN*l+k,0]*self.RW_V0d_MN[l,k] for l in range (0, self.nV0d))
            if k == 4: # this is to hyperpolarize a MN to observe periodic depolarizations and synaptic bursts
                IsynL = IsynL - 10

            IgapL = - sum(self.LSGap_MN_IC[k,:]) + sum(self.LSGap_IC_MN[:,k]) - sum(self.LSGap_MN_MN[k,:]) + sum(self.LSGap_MN_MN[:,k]) - sum(self.LSGap_MN_V0d[k,:]) + sum(self.LSGap_V0d_MN[:,k]) - sum(self.LSGap_MN_V0v[k,:]) + sum(self.LSGap_V0v_MN[:,k]) - sum(self.LSGap_MN_V2a[k,:]) + sum(self.LSGap_V2a_MN[:,k])
            IgapR = - sum(self.RSGap_MN_IC[k,:]) + sum(self.RSGap_IC_MN[:,k]) - sum(self.RSGap_MN_MN[k,:]) + sum(self.RSGap_MN_MN[:,k]) - sum(self.RSGap_MN_V0d[k,:]) + sum(self.RSGap_V0d_MN[:,k]) - sum(self.RSGap_MN_V0v[k,:]) + sum(self.RSGap_V0v_MN[:,k]) - sum(self.LSGap_MN_V2a[k,:]) + sum(self.LSGap_V2a_MN[:,k])

            self.resLMN[k,:] = self.L_MN[k].getNextVal(self.resLMN[k,0],self.resLMN[k,1], IgapL + IsynL)  
            self.VLMN[k,t] = self.resLMN[k,0]
            
            self.resRMN[k,:] = self.R_MN[k].getNextVal(self.resRMN[k,0],self.resRMN[k,1], IgapR + IsynR) 
            self.VRMN[k,t] = self.resRMN[k,0]

    def calcV0vPotentialsandResidues(self, t):
        for k in range (0, self.nV0v):
            IsynL= 0.0
            IsynR= 0.0

            IgapL = - sum(self.LSGap_V0v_IC[k,:]) + sum(self.LSGap_IC_V0v[:,k]) - sum(self.LSGap_V0v_V0v[k,:]) + sum(self.LSGap_V0v_V0v[:,k]) -sum(self.LSGap_V0v_MN[k,:])  + sum(self.LSGap_MN_V0v[:,k])
            IgapR = - sum(self.RSGap_V0v_IC[k,:]) + sum(self.RSGap_IC_V0v[:,k]) - sum(self.RSGap_V0v_V0v[k,:]) + sum(self.RSGap_V0v_V0v[:,k]) -sum(self.RSGap_V0v_MN[k,:])  + sum(self.RSGap_MN_V0v[:,k])

            self.resLV0v[k,:] = self.L_V0v[k].getNextVal(self.resLV0v[k,0],self.resLV0v[k,1], IgapL + IsynL)
            self.VLV0v[k,t] = self.resLV0v[k,0]
            self.resRV0v[k,:] = self.R_V0v[k].getNextVal(self.resRV0v[k,0],self.resRV0v[k,1], IgapR + IsynR)
            self.VRV0v[k,t] = self.resRV0v[k,0]