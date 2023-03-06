__all__ = ['Experiment']

from Transients import _transients as _trans
import numpy as np
from scipy.integrate import quadrature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button


class Experiment:
    def __init__(self, omegas, gammas, rabis, delays=[], pulse_widths=[], times=[]):
        self.omegas = list(map(_trans.wn2Hz, omegas))
        self.gammas = list(map(_trans.wn2Hz, gammas))
        self.rabis = list(map(_trans.wn2Hz, rabis))
        self.delays = delays
        self.pws = pulse_widths
        self.times = times

    def set_delays(self, delays):
        self.delays = delays

    def get_delays(self):
        return self.delays

    def set_pws(self, pws):
        self.pws = pws

    def get_pws(self):
        return self.pws

    def set_times(self, time_int=100):
        if len(self.pws) > len(self.delays) > 0:
            t0 = 0
            t1 = round(t0 + self.pws[0], 18)
            t2 = t1 + self.delays[0]
            t3 = round(t2 + self.pws[1], 18)
            t4 = t3 + self.delays[1]
            t5 = t4 + self.pws[2]
            self.times = [t0, t1, t2, t3, t4, t5]
        else:
            print('Invalid pulse width and/or delay inputs.')

    def get_times(self):
        return self.times

    def set_pulse_freqs(self, freqs):
        self.pulse_freqs = list(map(_trans.wn2Hz, freqs))

    def get_pulse_freqs(self):
        return self.pulse_freqs, list(map(_trans.Hz2wn, self.pulse_freqs))

    def set_scan_range(self, scan_ranges):
        if self.pulse_freqs:
            self.scan_range = np.array([self.pulse_freqs[0] - scan_ranges[0],
                                        self.pulse_freqs[0] + scan_ranges[0],
                                        self.pulse_freqs[1] - scan_ranges[1],
                                        self.pulse_freqs[1] + scan_ranges[1]])
        else:
            print('Not Completed. Specify pulse freqs first.')

    def get_scan_range(self):
        return self.scan_range

    def set_transitions(self, transitions):
        self.transitions = np.array(transitions)

    def get_transitions(self):
        return [t.__name__ for t in self.transitions]

    def set_pm(self, phasematching):
        self.pm = np.array(phasematching)
        self.signs, self.orderings = np.sign(self.pm), np.abs(self.pm) - 1

    def get_pm(self):
        return self.pm

    def _timedparams(self, t, where):
        if self.times[0] <= t <= self.times[1]:
            t1, fid1 = 1, 0
        elif self.times[1] <= t:
            t1, fid1 = 0, 1
        else:
            t1 = fid1 = 0

        if self.times[2] <= t <= self.times[3]:
            t2, fid2 = 1, 0
        elif self.times[3] <= t:
            t2, fid2 = 0, 1
        else:
            t2 = fid2 = 0

        if self.times[4] <= t <= self.times[5]:
            t3 = 1
        else:
            t3 = 0

        driven = [t1*self.signs[0]*self.pulse_freqs[self.orderings[0]],
                  t2*self.signs[1]*self.pulse_freqs[self.orderings[1]],
                  t3*self.signs[2]*self.pulse_freqs[self.orderings[2]]]
        fid = [fid1*self.signs[0]*self.omegas[self.orderings[0]],
               fid2*self.signs[1]*self.omegas[self.orderings[1]],
               0]
        return np.sum(driven, where=where)

    def compute(self):
        self.t1 = lambda t: self.transitions[0](self.rabis[self.orderings[0]],
                                           self.pulse_freqs[self.orderings[0]],
                                           0,
                                           self.signs[0]*self.omegas[self.orderings[0]],
                                           self.gammas[self.orderings[0]],
                                           self.times[0],
                                           self.times[1],
                                           t) * _trans.pulse(self.times[0], self.times[1], t) 
        self.fid1 = lambda t: _trans.fid(self.t1(self.times[1]),
                                        self.signs[0]*self.omegas[self.orderings[0]],
                                        self.gammas[self.orderings[0]],
                                        t-self.times[1]) * _trans.Hs(t-self.times[1])

        self.t2 = lambda t: self.transitions[1](self.rabis[self.orderings[1]],
                                                self.pulse_freqs[self.orderings[1]],
                                                self.signs[0]*self.omegas[self.orderings[0]],
                                                self.signs[0]*self.omegas[self.orderings[0]]
                                                + self.signs[1]*self.omegas[self.orderings[1]],
                                                self.gammas[self.orderings[1]],
                                                self.times[2],
                                                self.times[3],
                                                t) * _trans.pulse(self.times[2], self.times[3], t)
        self.fid2 = lambda t: _trans.fid(self.t2(self.times[3]),
                                         self.signs[0]*self.omegas[self.orderings[0]]
                                         + self.signs[1]*self.omegas[self.orderings[1]],
                                         self.gammas[self.orderings[1]],
                                         t-self.times[3]) * _trans.Hs(t-self.times[3])


        self.t3 = lambda t: self.transitions[2](self.rabis[self.orderings[2]],
                                                self.pulse_freqs[self.orderings[2]],
                                                self.signs[0]*self.omegas[self.orderings[0]]
                                                + self.signs[1]*self.omegas[self.orderings[1]],
                                                self.signs[0]*self.omegas[self.orderings[0]]
                                                + self.signs[1]*self.omegas[self.orderings[1]]
                                                + self.signs[2]*self.omegas[self.orderings[2]],
                                                self.gammas[self.orderings[2]],
                                                self.times[4],
                                                self.times[5],
                                                t)
        return np.real(quadrature(lambda t: (self.t1(t)+self.fid1(t))*(self.t2(t)+self.fid2(t))*self.t3(t) \
                                  * np.conjugate((self.t1(t)+self.fid1(t))*(self.t2(t)+self.fid2(t))*self.t3(t)),
                                  self.times[4], self.times[5])[0])

    def draw(self, spacing=1, **kwargs):
        times = np.linspace(self.times[0], self.times[5], int((self.times[5]-self.times[0])*10e15))
        amps1 = [self.t1(i)+self.fid1(i) for i in times]
        amps21 = [(self.t1(i)+self.fid1(i))*(self.t2(i)+self.fid2(i)) for i in times]
        amps321 = [(self.t1(i)+self.fid1(i))*(self.t2(i)+self.fid2(i))*self.t3(i) for i in times]
        plt.plot(times, amps1, color='k', alpha=0.50)
        plt.plot(times, amps21, color='r', alpha=0.75)
        plt.plot(times, amps321, color='b', alpha=0.75)
        plt.show()
