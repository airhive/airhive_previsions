#!/usr/bin/env python
# -*- coding: utf-8 -*-
#cython: language_level=3

cimport numpy as np

def controlla_anomalie(np.ndarray[double, ndim=1] df, double media, double dev_std, double numero_deviazioni):
    """Calcolo i V-values"""
    # Accetta una sola colonna del df
    V_values = (df - media) / dev_std
    return V_values > numero_deviazioni