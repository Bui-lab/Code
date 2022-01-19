#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from random import gauss
from Beat_and_glide import Beat_and_glide_base

class Beat_and_glide_30_Somites(Beat_and_glide_base):

    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 30, ndI6 = 30, nV0v = 30, nV2a = 30, nV1 = 30, nMuscle = 30, 
                          R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
    
    def initWeightParameters(self):
        super().initWeightParameters()
        self.V2a_dI6_syn_weight = 0.65       
        self.V2a_V0v_syn_weight = 0.25 
        self.V1_V2a_syn_weight = 0.75    

    def noiseMultiplier(self):
        return gauss(1, self.sigma)
