#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""
from random import gauss
from Izhikevich_class import Izhikevich_9P, Leaky_Integrator

from Double_coiling_model import Double_coil_base

class Double_coil_with_sigmas(Double_coil_base):

    sigmaD = 0
    sigmaL = 0
    sigmaP = 0
    sigmaW = 0
    
    def __init__(self, dt = 0.1, stim0 = 8, sigmaD = 0, sigmaL = 0, sigmaP = 0, sigmaW = 0,
                    E_glu = 0, E_gly = -70,
                  cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nV0v = 10, nV2a = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigmaD, E_glu, E_gly,
                  cv, nIC, nMN, nV0d, nV0v, nV2a, nMuscle)
        self.sigmaD = sigmaD
        self.sigmaL = sigmaL
        self.sigmaP = sigmaP
        self.sigmaW = sigmaW


    def initNeurons(self):
        ## Declare Neuron Types

        self.L_IC = [ Izhikevich_9P(a=0.0002*gauss(1, self.sigmaP),b=0.5*gauss(1, self.sigmaP),c=-40*gauss(1, self.sigmaP), d=5*gauss(1, self.sigmaP), vmax=0*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.3*gauss(1, self.sigmaP), Cm = 50*gauss(1, self.sigmaP), dt = self.dt, x=1.0,y=-1) for i in range(self.nIC)]
        self.R_IC = [ Izhikevich_9P(a=0.0002*gauss(1, self.sigmaP),b=0.5*gauss(1, self.sigmaP),c=-40*gauss(1, self.sigmaP), d=5*gauss(1, self.sigmaP), vmax=0*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.3*gauss(1, self.sigmaP), Cm = 50*gauss(1, self.sigmaP), dt = self.dt, x=1.0,y=1) for i in range(self.nIC)]

        self.L_MN = [ Izhikevich_9P(a=0.5*gauss(1, self.sigmaP),b=0.1*gauss(1, self.sigmaP),c=-50*gauss(1, self.sigmaP), d=100*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-50*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.0+1.6*i,y=-1) for i in range(self.nMN)]
        self.R_MN = [ Izhikevich_9P(a=0.5*gauss(1, self.sigmaP),b=0.1*gauss(1, self.sigmaP),c=-50*gauss(1, self.sigmaP), d=100*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-50*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.0+1.6*i,y=1) for i in range(self.nMN)]

        self.L_V0d = [ Izhikevich_9P(a=0.02*gauss(1, self.sigmaP),b=0.1*gauss(1, self.sigmaP),c=-30*gauss(1, self.sigmaP), d=3.75*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.0+1.6*i,y=-1) for i in range(self.nV0d)]
        self.R_V0d = [ Izhikevich_9P(a=0.02*gauss(1, self.sigmaP),b=0.1*gauss(1, self.sigmaP),c=-30*gauss(1, self.sigmaP), d=3.75*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.0+1.6*i,y=1) for i in range(self.nV0d)]

        self.L_V0v = [ Izhikevich_9P(a=0.02*gauss(1, self.sigmaP), b=0.1*gauss(1, self.sigmaP), c=-30*gauss(1, self.sigmaP), d=11.6*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.1+1.6*i,y=-1) for i in range(self.nV0v)]
        self.R_V0v = [ Izhikevich_9P(a=0.02*gauss(1, self.sigmaP), b=0.1*gauss(1, self.sigmaP), c=-30*gauss(1, self.sigmaP), d=11.6*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.1+1.6*i,y=1) for i in range(self.nV0v)]

        self.L_V2a = [ Izhikevich_9P(a=0.5*gauss(1, self.sigmaP), b=0.1*gauss(1, self.sigmaP), c=-50*gauss(1, self.sigmaP), d=100*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.1+1.6*i,y=-1) for i in range(self.nV2a)]
        self.R_V2a = [ Izhikevich_9P(a=0.5*gauss(1, self.sigmaP), b=0.1*gauss(1, self.sigmaP), c=-50*gauss(1, self.sigmaP), d=100*gauss(1, self.sigmaP), vmax=10*gauss(1, self.sigmaP), vr=-60*gauss(1, self.sigmaP), vt=-45*gauss(1, self.sigmaP), k=0.05*gauss(1, self.sigmaP), Cm = 20*gauss(1, self.sigmaP), dt = self.dt, x=5.1+1.6*i,y=1) for i in range(self.nV2a)]

        self.L_Muscle = [ Leaky_Integrator(50.0*gauss(1, self.sigmaP), 5.0*gauss(1, self.sigmaP), self.dt, 5.0+1.6*i,-1) for i in range(self.nMuscle)]
        self.R_Muscle = [ Leaky_Integrator(50.0*gauss(1, self.sigmaP), 5.0*gauss(1, self.sigmaP), self.dt, 5.0+1.6*i, 1) for i in range(self.nMuscle)]

    def rangeNoiseMultiplier(self):
        return gauss(1, self.sigmaL)

    def gapWeightNoiseMultiplier(self):
        return gauss(1, self.sigmaW)

    def synWeightNoiseMultiplier(self):
        return gauss(1, self.sigmaW)

    def getStimulus(self):
        return self.stim0 * gauss(1, self.sigmaD)

    def printParameters(self):
        super().printParameters()
        print("sigmaD: " + str(self.sigmaP) + "; sigmaL: " + str(self.sigmaL) + "; sigmaP: " + str(self.sigmaP) + "; sigmaW: " + str(self.sigmaW))


