#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:47:19 2018

@author: Yann Roussel and Tuan Bui
Editted by: Emine Topcu on Oct 2021
"""
from random import gauss
from Beat_and_glide import Beat_and_glide_base
from Izhikevich_class import Izhikevich_9P, Leaky_Integrator

class Beat_and_glide_with_sigmas(Beat_and_glide_base):

    sigmaD = 0
    sigmaL = 0
    sigmaP = 0
    sigmaW = 0
    
    def __init__ (self, stim0 = 2.89, sigma = 0, sigma_LR = 0.1, sigmaD = 0, sigmaL = 0, sigmaP = 0, sigmaW = 0,
                        E_glu = 0, E_gly = -70, cv = 0.80,
                        nMN = 15, ndI6 = 15, nV0v = 15, nV2a = 15, nV1 = 15, nMuscle = 15, 
                        R_str = 1.0):
        super().__init__(stim0, sigma, sigma_LR, E_glu, E_gly, cv,
                          nMN, ndI6, nV0v, nV2a, nV1, nMuscle, R_str)
        self.sigmaD = sigmaD
        self.sigmaL = sigmaL
        self.sigmaP = sigmaP
        self.sigmaW = sigmaW

    def initNeurons(self):
        ## Declare Neuron Types

        self.L_MN = [ Izhikevich_9P(a = 0.5*gauss(1, self.sigmaP),
                                    b = 0.01*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP), 
                                    d = 100*gauss(1, self.sigmaP), 
                                    vmax = 10*gauss(1, self.sigmaP), 
                                    vr = -65*gauss(1, self.sigmaP), 
                                    vt = -58*gauss(1, self.sigmaP), 
                                    k = 0.5*gauss(1, self.sigmaP), 
                                    Cm = 20*gauss(1, self.sigmaP), 
                                    dt = self.getdt(), 
                                    x = 5.0+1.6*i*gauss(1, self.sigma),
                                    y = -1) for i in range(self.nMN)]
        self.R_MN = [ Izhikevich_9P(a = 0.5*gauss(1, self.sigmaP),
                                    b = 0.01*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP),
                                    d = 100*gauss(1, self.sigmaP),
                                    vmax = 10*gauss(1, self.sigmaP),
                                    vr = -65*gauss(1, self.sigmaP),
                                    vt = -58*gauss(1, self.sigmaP),
                                    k = 0.5*gauss(1, self.sigmaP),
                                    Cm = 20*gauss(1, self.sigmaP),
                                    dt = self.getdt(),
                                    x = 5.0+1.6*i*gauss(1, self.sigma),
                                    y = 1) for i in range(self.nMN)]

        self.L_dI6 = [ Izhikevich_9P(a = 0.1*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP), 
                                    d = 4*gauss(1, self.sigmaP), 
                                    vmax = 10*gauss(1, self.sigmaP),
                                    vr = -60*gauss(1, self.sigmaP), 
                                    vt = -54*gauss(1, self.sigmaP), 
                                    k = 0.3*gauss(1, self.sigmaP), 
                                    Cm = 10*gauss(1, self.sigmaP), 
                                    dt = self.getdt(), 
                                    x = 5.1+1.6*i*gauss(1, self.sigma),
                                    y = -1) for i in range(self.ndI6)]
        self.R_dI6 = [ Izhikevich_9P(a = 0.1*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP), 
                                    d = 4*gauss(1, self.sigmaP), 
                                    vmax = 10*gauss(1, self.sigmaP), 
                                    vr = -60*gauss(1, self.sigmaP), 
                                    vt = -54*gauss(1, self.sigmaP), 
                                    k = 0.3*gauss(1, self.sigmaP), 
                                    Cm = 10*gauss(1, self.sigmaP), 
                                    dt = self.getdt(), 
                                    x = 5.1+1.6*i*gauss(1, self.sigma),
                                    y = 1) for i in range(self.ndI6)]

        self.L_V0v = [ Izhikevich_9P(a = 0.01*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP), 
                                    d = 2*gauss(1, self.sigmaP), 
                                    vmax = 8*gauss(1, self.sigmaP), 
                                    vr = -60*gauss(1, self.sigmaP), 
                                    vt = -54*gauss(1, self.sigmaP), 
                                    k = 0.3*gauss(1, self.sigmaP), 
                                    Cm = 10*gauss(1, self.sigmaP), 
                                    dt = self.getdt(), 
                                    x = 5.1+1.6*i*gauss(1, self.sigma),
                                    y = -1) for i in range(self.nV0v)]
        self.R_V0v = [ Izhikevich_9P(a = 0.01*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP), 
                                    d = 2*gauss(1, self.sigmaP), 
                                    vmax = 8*gauss(1, self.sigmaP), 
                                    vr = -60*gauss(1, self.sigmaP), 
                                    vt = -54*gauss(1, self.sigmaP), 
                                    k = 0.3*gauss(1, self.sigmaP), 
                                    Cm = 10*gauss(1, self.sigmaP), 
                                    dt = self.getdt(),
                                    x = 5.1+1.6*i*gauss(1, self.sigma),
                                    y = 1) for i in range(self.nV0v)]

        self.L_V2a = [ Izhikevich_9P(a = 0.1*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP),
                                    d = 4*gauss(1, self.sigmaP), 
                                    vmax = 10*gauss(1, self.sigmaP),
                                    vr = -60*gauss(1, self.sigmaP),
                                    vt = -54*gauss(1, self.sigmaP),
                                    k = 0.3*gauss(1, self.sigmaP),
                                    Cm = 10*gauss(1, self.sigmaP),
                                    dt = self.getdt(),
                                    x = 5.1+1.6*i*gauss(1, self.sigma),
                                    y = -1) for i in range(self.nV2a)]
        self.R_V2a = [ Izhikevich_9P(a = 0.1*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP),
                                    d = 4*gauss(1, self.sigmaP),
                                    vmax = 10*gauss(1, self.sigmaP),
                                    vr = -60*gauss(1, self.sigmaP),
                                    vt = -54*gauss(1, self.sigmaP),
                                    k = 0.3*gauss(1, self.sigmaP),
                                    Cm = 10*gauss(1, self.sigmaP),
                                    dt = self.getdt(),
                                    x = 5.1+1.6*i*gauss(1, self.sigma),
                                    y = 1) for i in range(self.nV2a)]
        
        self.L_V1 = [ Izhikevich_9P(a = 0.1*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP),
                                    d = 4*gauss(1, self.sigmaP), 
                                    vmax = 10*gauss(1, self.sigmaP),
                                    vr = -60*gauss(1, self.sigmaP),
                                    vt = -54*gauss(1, self.sigmaP),
                                    k = 0.3*gauss(1, self.sigmaP),
                                    Cm = 10*gauss(1, self.sigmaP),
                                    dt = self.getdt(), 
                                    x = 7.1+1.6*i*gauss(1, self.sigma),
                                    y = -1) for i in range(self.nV1)]
        self.R_V1 = [ Izhikevich_9P(a = 0.1*gauss(1, self.sigmaP),
                                    b = 0.002*gauss(1, self.sigmaP),
                                    c = -55*gauss(1, self.sigmaP),
                                    d = 4*gauss(1, self.sigmaP),
                                    vmax = 10*gauss(1, self.sigmaP),
                                    vr = -60*gauss(1, self.sigmaP),
                                    vt = -54*gauss(1, self.sigmaP),
                                    k = 0.3*gauss(1, self.sigmaP),
                                    Cm = 10*gauss(1, self.sigmaP),
                                    dt = self.getdt(),
                                    x = 7.1+1.6*i*gauss(1, self.sigma),
                                    y = 1) for i in range(self.nV1)]
        
        self.L_Muscle = [ Leaky_Integrator(1.0, 3.0, self.getdt(), 5.0+1.6*i,-1) for i in range(self.nMuscle)]
        self.R_Muscle = [ Leaky_Integrator(1.0, 3.0, self.getdt(), 5.0+1.6*i, 1) for i in range(self.nMuscle)]
    
    def getStimulus(self, t):
        if t > 2000: # Let the initial conditions dissipate for the first 200 ms
            return self.stim0 * gauss(1, self.sigmaD)
        return 0

    def rangeNoiseMultiplier(self):
        return gauss(1, self.sigmaL)

    def weightNoiseMultiplier(self):
        return gauss(1, self.sigmaW)
        
