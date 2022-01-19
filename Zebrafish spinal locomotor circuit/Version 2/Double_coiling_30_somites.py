#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from Double_coiling_model import Double_coil_base
from Izhikevich_class import Leaky_Integrator

class Double_coil_30_somites(Double_coil_base):

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0, E_glu = 0, E_gly = -70,
                  cv = 0.55, nIC = 5, nMN = 30, nV0d = 30, nV0v = 30, nV2a = 30, nMuscle = 30):
        super().__init__(dt, stim0, sigma, E_glu, E_gly, cv, nIC, nMN, nV0d, nV0v, nV2a, nMuscle)
        super().setWeightParameters(IC_IC_gap_weight = 0.0001, IC_MN_gap_weight = 0.05, 
                    IC_V0d_gap_weight = 0.05, IC_V0v_gap_weight = 0.00005, IC_V2a_gap_weight = 0.15, 
                    MN_MN_gap_weight =  0.07, MN_V0d_gap_weight = 0.0001, MN_V0v_gap_weight = 0.0001,  MN_Muscle_syn_weight = 0.02,        
                    V0d_IC_syn_weight = 0.3, V0d_MN_syn_weight = 0.3, V0d_V0d_gap_weight = 0.03, V0d_V2a_syn_weight = 0.3,
                    V0v_IC_syn_weight = 0.5, V0v_V0v_gap_weight = 0.044,
                    V2a_MN_gap_weight = 0.005, V2a_V0v_syn_weight = 0.0375, V2a_V2a_gap_weight = 0.005) 
        super().setRangeParameters(rangeMin = 0.2, rangeIC_MN = 30, rangeIC_V0d = 30, rangeIC_V0v = 30, rangeIC_V2a = 30,
                            rangeMN_MN = 6.5, rangeMN_V0d = 1.5, rangeMN_V0v = 1.5,
                            rangeV0d_IC = 20, rangeV0d_MN = 8, rangeV0d_V0d = 3.5, rangeV0d_V2a = 8,
                            rangeV0v_IC_min = 6, rangeV0v_IC_max = 20, rangeV0v_V0v = 3.5,
                            rangeV2a_MN = 3.5, rangeV2a_V0v_asc = 4, rangeV2a_V0v_desc = 10, rangeV2a_V2a = 3.5,
                            rangeMN_Muscle = 1)


    def initNeurons(self):
        super().initNeurons()
        self.L_Muscle = [ Leaky_Integrator(20.0, 10.0, self.dt, 5.0+1.6*i,-1) for i in range(self.nMuscle)]
        self.R_Muscle = [ Leaky_Integrator(20.0, 10.0, self.dt, 5.0+1.6*i, 1) for i in range(self.nMuscle)]
