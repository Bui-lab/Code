#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 08:41:10 2017

@author: Yann Roussel and Tuan Bui
Edited by: Emine Topcu on Sep 2021
"""

from Double_coiling_model import Double_coil_base

class Double_coil_glycine_null(Double_coil_base):

    def __init__(self, dt = 0.1, stim0 = 8, sigma = 0, E_glu = 0, E_gly = -70,
                  cv = 0.55, nIC = 5, nMN = 10, nV0d = 10, nV0v = 10, nV2a = 10, nMuscle = 10):
        super().__init__(dt, stim0, sigma, E_glu, E_gly,
                  cv, nIC, nMN, nV0d, nV0v, nV2a, nMuscle)
        super().setWeightParameters(V0d_IC_syn_weight=0, V0d_MN_syn_weight=0, V0d_V2a_syn_weight=0)
