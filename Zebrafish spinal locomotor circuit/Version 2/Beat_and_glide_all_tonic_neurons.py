#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from random import gauss
from Beat_and_glide import Beat_and_glide_base
from Izhikevich_class import Izhikevich_9P

class Beat_and_glide_all_tonic(Beat_and_glide_base):

    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                          R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
        self.setWeightParameters(V2a_V0v_syn_weight = 0.25, V1_V2a_syn_weight = 0.6)

    def initNeurons(self):
        super().initNeurons()
        _dt = self.getdt()
        #V0v neurons are overwritten with different parameters
        self.L_V0v = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=_dt, x=5.1+1.6*i*gauss(1, self.sigma),y=-1) for i in range(self.nV0v)]
        self.R_V0v = [ Izhikevich_9P(a=0.1,b=0.002,c=-55, d=4, vmax=10, vr=-60, vt=-54, k=0.3, Cm = 10, dt=_dt, x=5.1+1.6*i*gauss(1, self.sigma),y=1) for i in range(self.nV0v)]

