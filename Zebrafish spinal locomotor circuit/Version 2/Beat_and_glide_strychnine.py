#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from random import gauss
from Beat_and_glide import Beat_and_glide_base

class Beat_and_glide_strychnine(Beat_and_glide_base):

    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, E_glu = 0, E_gly = -70, cv = 0.80,
                          nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                          R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
        self.setTimeParameters() #to nitialize with default values

    def setTimeParameters(self, tmax_ms = 10000, tshutoff_ms = 50, tskip_ms = 1000, dt = 0.1, tStrystart_ms = 6000, tStryend_ms = 11000):
        super().setTimeParameters(tmax_ms, tshutoff_ms, tskip_ms, dt)
        self.__tStrystart = (tStrystart_ms + tskip_ms) / dt
        self.__tStrayend = (tStryend_ms + tskip_ms) / dt

    def getStimulus(self, t):
        if t > self.__tStrystart and t < self.__tStrayend:
            self.R_str = 0.0
        else:
            self.R_str = 1.0

        if t > 2000: # Let the initial conditions dissipate for the first 200 ms
            return self.stim0
        return 0

            
        
