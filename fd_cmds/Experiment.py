__all__ = ['Experiment']

import _transientsv3 as _trans
import numpy as np


class Experiment:
    def __init__(self, omegas, gammas, rabis, delays=[], pws=[], times=[]):
        self.omegas = list(map(_trans.wntohz, omegas))
        self.gammas = list(map(_trans.wntohz, gammas))
        self.rabis = list(map(_trans.wntohz, rabis))
        self.delays = delays
        self.pws = pws
        self.times = times

    def set_delays(self, delays):
        self.delays = delays
    
    def get_delays(self):
        return self.delays

    def set_pws(self, pws):
        self.pws = pws

    def get_pws(self):
        return self.pws

    def set_times(self):
        if len(self.pws)>len(self.delays)>0:
            t0 = 0
            t1 = round(t0+self.pws[0], 18)
            t2 = t1+self.delays[0]
            t3 = round(t2+self.pws[1], 18)
            t4 = t3+self.delays[1]
            t5 = t4+self.pws[2]
            self.times = [t0,t1,t2,t3,t4,t5]
        else: 
            print('Invalid pulse width and/or delay inputs.')

    def get_times(self):
        return self.times

    def set_pulse_freqs(self, freqs):
        self.pulse_freqs = list(map(_trans.wntohz, freqs))

    def get_pulse_freqs(self):
        return self.pulse_freqs

    def set_scan_range(self, scan_ranges):
        if self.pulse_freqs:
            self.scan_range = [self.pulse_freqs[0]-scan_ranges[0],
                               self.pulse_freqs[0]+scan_ranges[0],
                               self.pulse_freqs[1]-scan_ranges[1],
                               self.pulse_freqs[1]+scan_ranges[1]]
        else: 
            print('Not Completed. Specify pulse freqs first.')

    def get_scan_range(self):
        return self.scan_range

    def set_transitions(self, transitions):
        self.transitions = transitions

    def get_transitions(self):
        return [t.__name__ for t in self.transitions]
    
    def set_pm(self, phasematching):
        self.pm = phasematching    
    
    def get_pm(self):
        return self.pm
    
    def compute(self, time_int=100e15):
        """
        :param time_int:
        :return:
        run the calculation
        :return: output coherence amplitude
        """

        signs, orderings = np.sign(self.pm), np.abs(self.pm)-1
        GROUND_STATE_GAMMA = _trans.wntohz(1e-18)

        t1 = self.transitions[0](self.rabis[orderings[0]],
                               _trans.delta_ij(0, GROUND_STATE_GAMMA),
                               _trans.delta_ij(signs[0]*self.omegas[orderings[0]], self.gammas[orderings[0]]), #note: these are coherence gammas
                               self.pulse_freqs[orderings[0]],
                               signs[0]*self.omegas[orderings[0]],
                               GROUND_STATE_GAMMA,
                               self.gammas[orderings[0]],
                               self.times[1])
        fid1 = _trans.fid(1,
                          _trans.delta_ij(self.omegas[orderings[0]], self.gammas[orderings[0]]),
                          self.times[3]-self.times[1])

        t2 = self.transitions[1](self.rabis[orderings[1]],
                               _trans.delta_ij(signs[0]*self.omegas[orderings[0]],
                                               self.gammas[orderings[0]]),
                               _trans.delta_ij(signs[0]*self.omegas[orderings[0]]
                                               + signs[1]*self.omegas[orderings[1]], self.gammas[orderings[1]]),
                               signs[0]*self.pulse_freqs[orderings[0]] \
                               + signs[1]*self.pulse_freqs[orderings[1]],
                               signs[0]*self.omegas[orderings[0]] + signs[1]*self.omegas[orderings[1]],
                               self.gammas[orderings[0]],
                               self.gammas[orderings[1]],
                               self.times[3] - self.times[2])
        fid2 = _trans.fid(1,
                          _trans.delta_ij(signs[0]*self.omegas[orderings[0]]
                                          + signs[1]*self.omegas[orderings[1]], self.gammas[orderings[1]]),
                          np.linspace(self.times[4] - self.times[3],
                                      self.times[5] - self.times[3],
                                      int((self.times[5] - self.times[4]) * time_int)+1))
        t3 = self.transitions[2](self.rabis[orderings[2]],
                               _trans.delta_ij(signs[0]*self.omegas[orderings[0]]
                                               + signs[1]*self.omegas[orderings[1]], self.gammas[orderings[1]]),
                               _trans.delta_ij(signs[0]*self.omegas[orderings[0]]
                                               + signs[1]*self.omegas[orderings[1]]
                                               + signs[2]*self.omegas[orderings[2]], self.gammas[orderings[2]]),
                               signs[0]*self.pulse_freqs[orderings[0]]
                               + signs[1]*self.pulse_freqs[orderings[1]]
                               + signs[2]*self.pulse_freqs[orderings[2]],
                               signs[0]*self.omegas[orderings[0]]
                               + signs[1]*self.omegas[orderings[1]]
                               + signs[2]*self.omegas[orderings[2]],
                               self.gammas[orderings[1]],
                               self.gammas[orderings[2]],
                               np.linspace(0, self.times[5]-self.times[4],
                                           int((self.times[5]-self.times[4])*time_int)+1))
                                           
        coeff = t1*fid1*t2*fid2
        out_field = coeff*t3
        return np.sum(np.real(out_field*np.conjugate(out_field)))