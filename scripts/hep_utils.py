import matplotlib.pyplot as plt
import seaborn as sns
import mplhep as hep
import awkward as ak
import pandas as pd
import numpy as np
import uproot
import yaml
import math

from numba import njit

def invariantMass3(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2, pt3, eta3, phi3, m3):
    # Calcular p_x, p_y, p_z para cada partícula
    def calculate_momentum(pt, eta, phi):
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        return px, py, pz
    
    # Energía de cada partícula
    def calculate_energy(pt, pz, mass):
        return np.sqrt(pt**2 + pz**2 + mass**2)
    
    # Cálculos para la partícula 1
    px1, py1, pz1 = calculate_momentum(pt1, eta1, phi1)
    E1 = calculate_energy(pt1, pz1, m1)
    
    # Cálculos para la partícula 2
    px2, py2, pz2 = calculate_momentum(pt2, eta2, phi2)
    E2 = calculate_energy(pt2, pz2, m2)
    
    # Cálculos para la partícula 3
    px3, py3, pz3 = calculate_momentum(pt3, eta3, phi3)
    E3 = calculate_energy(pt3, pz3, m3)
    
    # Sumar los cuatro-momentos totales
    E_total = E1 + E2 + E3
    px_total = px1 + px2 + px3
    py_total = py1 + py2 + py3
    pz_total = pz1 + pz2 + pz3
    
    # Calcular la masa invariante
    mass_invariant = np.sqrt(E_total**2 - (px_total**2 + py_total**2 + pz_total**2))
    
    return mass_invariant

def invariantMass2(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    # Calcular componentes px, py y pz de cada leptón
    px1 = pt1 * math.cos(phi1)
    py1 = pt1 * math.sin(phi1)
    pz1 = pt1 * math.sinh(eta1)
    E1 = math.sqrt(pt1**2 + pz1**2 + m1**2)
    
    px2 = pt2 * math.cos(phi2)
    py2 = pt2 * math.sin(phi2)
    pz2 = pt2 * math.sinh(eta2)
    E2 = math.sqrt(pt2**2 + pz2**2 + m2**2)
    
    # Calcular el cuadrimomento total (E, px, py, pz)
    E_total = E1 + E2
    px_total = px1 + px2
    py_total = py1 + py2
    pz_total = pz1 + pz2
    
    # Calcular la masa invariante
    arg = E_total**2 - px_total**2 - py_total**2 - pz_total**2
    if arg < 0:
        return 0

    M = math.sqrt(arg)
    
    return M

def transverseMass(pt, Emiss, dphi):
    return np.sqrt( 2 * pt * Emiss * (1 - np.cos(dphi)) )
