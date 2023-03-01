__all__ = ['wn2Hz',
           'Hz2wn',
           'Hs',
           'delta_ij',
           'pulse',
           'fid',
           'ket_abs',
           'ket_emis',
           'bra_abs',
           'bra_emis']

# imports
import numpy as np

# useful constants
c = 2.998e8  # in m/s

# useful functions

def wn2Hz(wn):
    return wn*c*100*2*np.pi

def Hz2wn(hz):
    return hz/c/100/2/np.pi


# modeling functions

def Hs(x):
    return np.heaviside(x, 0)


def delta_ij(omega_ij, gamma_ij):
    return omega_ij - 1J*gamma_ij


def pulse(ti, tf, t):
    return Hs(t-ti) - Hs(t-tf)


def fid(rho0_ij,
        omega_ij,
        gamma_ij,
        t):
    return rho0_ij*np.exp(-1J * delta_ij(omega_ij, gamma_ij) * t)


def ket_abs(rabi_ik,
            omega,
            omega_kj,
            omega_ij,
            gamma_ij,
            ti,
            tf,
            t):
    return 0.5*rabi_ik*((np.exp(-1J * (omega_kj+omega) * (t-ti)) - np.exp(-1J * delta_ij(omega_ij, gamma_ij) * (t-ti)))
                        / (delta_ij(omega_ij, gamma_ij) - omega - omega_kj)) * pulse(ti, tf, t)


def ket_emis(rabi_ik,
             omega,
             omega_kj,
             omega_ij,
             gamma_ij,
             ti,
             tf,
             t):
    return 0.5*rabi_ik*((np.exp(-1J * (omega_kj-omega) * (t-ti)) - np.exp(-1J * delta_ij(omega_ij, gamma_ij) * (t-ti)))
                               / (delta_ij(omega_ij, gamma_ij) + omega - omega_kj)) * pulse(ti, tf, t)


def bra_abs(rabi_jk,
            omega,
            omega_ik,
            omega_ij,
            gamma_ij,
            ti,
            tf,
            t):
    return -0.5*rabi_jk*((np.exp(-1J * (omega_ik-omega) * (t-ti)) - np.exp(-1J * delta_ij(omega_ij, gamma_ij) * (t-ti)))
                           / (delta_ij(omega_ij, gamma_ij) + omega - omega_ik)) * pulse(ti, tf, t)


def bra_emis(rabi_jk,
             omega,
             omega_ik,
             omega_ij,
             gamma_ij,
             ti,
             tf,
             t):
    return -0.5*rabi_jk*((np.exp(-1J * (omega_ik+omega) * (t-ti)) - np.exp(-1J * delta_ij(omega_ij, gamma_ij) * (t-ti)))
                           / (delta_ij(omega_ij, gamma_ij) - omega - omega_ik)) * pulse(ti, tf, t)
