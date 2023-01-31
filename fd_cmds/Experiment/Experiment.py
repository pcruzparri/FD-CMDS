__all__ = ['Experiment']

from Transients import _transients as _trans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Experiment:
    def __init__(self, omegas, gammas, rabis, delays=[], pulse_widths=[], times=[]):
        self.omegas = list(map(_trans.wntohz, omegas))
        self.gammas = list(map(_trans.wntohz, gammas))
        self.rabis = list(map(_trans.wntohz, rabis))
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

    def set_times(self):
        if len(self.pws)>len(self.delays)>0:
            t0 = 0
            t1 = round(t0+self.pws[0], 18)
            t2 = t1+self.delays[0]
            t3 = round(t2+self.pws[1], 18)
            t4 = t3+self.delays[1]
            t5 = t4+self.pws[2]
            self.times = [t0, t1, t2, t3, t4, t5]
        else: 
            print('Invalid pulse width and/or delay inputs.')

    def get_times(self):
        return self.times

    def set_pulse_freqs(self, freqs):
        self.pulse_freqs = list(map(_trans.wntohz, freqs))

    def get_pulse_freqs(self):
        return self.pulse_freqs, list(map(_trans.hztown, self.pulse_freqs))

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
    
    def compute(self, time_int=100):
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
                                 _trans.delta_ij(signs[0] * self.omegas[orderings[0]],
                                                 self.gammas[orderings[0]]), #note: these are coherence gammas
                                 signs[0] * self.pulse_freqs[orderings[0]],
                                 signs[0] * self.omegas[orderings[0]],
                                 GROUND_STATE_GAMMA,
                                 self.gammas[orderings[0]],
                                 self.times[1])
        fid1 = _trans.fid(t1,
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
                                 self.times[3] - self.times[2]) * fid1
        fid2 = _trans.fid(t2,
                          _trans.delta_ij(signs[0]*self.omegas[orderings[0]]
                                          + signs[1]*self.omegas[orderings[1]], self.gammas[orderings[1]]),
                          np.linspace(self.times[4] - self.times[3],
                                      self.times[5] - self.times[3],
                                      int((self.times[5] - self.times[4]) * time_int * 1e15)+1))
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
                                             int((self.times[5]-self.times[4]) * time_int * 1e15)+1)) * fid2
                                           
        coeff = t1*fid1*t2*fid2
        out_field = t3
        return np.sum(np.real(out_field*np.conjugate(out_field)))

    def draw(self, spacing=1):
        spacing *= 1e-15
        fig, ax = plt.subplots(layout='constrained')

        if self.times and self.pulse_freqs:
            time1 = np.arange(self.times[0], self.times[1], spacing)
            delay1 = np.arange(self.times[1]+1e-15, self.times[2], spacing)
            time2 = np.arange(self.times[2]+1e-15, self.times[3], spacing)
            delay2 = np.arange(self.times[3]+1e-15, self.times[4], spacing)
            time3 = np.arange(self.times[4]+1e-15, self.times[5], spacing)
            time_axis = np.concatenate((time1, delay1, time2, delay2, time3))

            signs, orderings = np.sign(self.pm), np.abs(self.pm) - 1
            GROUND_STATE_GAMMA = _trans.wntohz(1e-18)

            t1 = self.transitions[0](self.rabis[orderings[0]],
                                     _trans.delta_ij(0, GROUND_STATE_GAMMA),
                                     _trans.delta_ij(signs[0] * self.omegas[orderings[0]],
                                                     self.gammas[orderings[0]]),  # note: these are coherence gammas
                                     signs[0] * self.pulse_freqs[orderings[0]],
                                     signs[0] * self.omegas[orderings[0]],
                                     GROUND_STATE_GAMMA,
                                     self.gammas[orderings[0]],
                                     time1)
            fid1 = _trans.fid(t1[-1],
                              _trans.delta_ij(signs[0] * self.omegas[orderings[0]], self.gammas[orderings[0]]),
                              time_axis-time1[-1]) * _trans.hs(time_axis-time1[-1])
            t2 = self.transitions[1](self.rabis[orderings[1]],
                                     _trans.delta_ij(signs[0] * self.omegas[orderings[0]],
                                                     self.gammas[orderings[0]]),
                                     _trans.delta_ij(signs[0] * self.omegas[orderings[0]]
                                                     + signs[1] * self.omegas[orderings[1]],
                                                     self.gammas[orderings[1]]),
                                     signs[0] * self.pulse_freqs[orderings[0]]
                                     + signs[1] * self.pulse_freqs[orderings[1]],
                                     signs[0] * self.omegas[orderings[0]] + signs[1] * self.omegas[orderings[1]],
                                     self.gammas[orderings[0]],
                                     self.gammas[orderings[1]],
                                     time2-time2[0]) * fid1[len(time1)+len(delay1):
                                                            len(time1)+len(delay1)+len(time2)]
            fid2 = _trans.fid(t2[-1],
                              _trans.delta_ij(signs[0] * self.omegas[orderings[0]]
                                              + signs[1] * self.omegas[orderings[1]], self.gammas[orderings[1]]),
                              time_axis-time2[-1]) * _trans.hs(time_axis-time2[-1])

            t3 = self.transitions[2](self.rabis[orderings[2]],
                                     _trans.delta_ij(signs[0] * self.omegas[orderings[0]]
                                                     + signs[1] * self.omegas[orderings[1]],
                                                     self.gammas[orderings[1]]),
                                     _trans.delta_ij(signs[0] * self.omegas[orderings[0]]
                                                     + signs[1] * self.omegas[orderings[1]]
                                                     + signs[2] * self.omegas[orderings[2]],
                                                     self.gammas[orderings[2]]),
                                     signs[0] * self.pulse_freqs[orderings[0]]
                                     + signs[1] * self.pulse_freqs[orderings[1]]
                                     + signs[2] * self.pulse_freqs[orderings[2]],
                                     signs[0] * self.omegas[orderings[0]]
                                     + signs[1] * self.omegas[orderings[1]]
                                     + signs[2] * self.omegas[orderings[2]],
                                     self.gammas[orderings[1]],
                                     self.gammas[orderings[2]],
                                     time3-time3[0]) * fid2[len(time1)+len(delay1)+len(time2)+len(delay2):
                                                            len(time_axis)]
            plt.plot(time1, t1, color='b', alpha=0.5)
            plt.plot(time_axis, fid1, color='b', alpha=0.5)
            plt.plot(time2, t2, color='g', alpha=0.5)
            plt.plot(time_axis, fid2, color='g', alpha=0.5)
            plt.plot(time3, t3, color='r', alpha=0.5)

            box_height = np.max(np.abs(t1))
            ax.add_patch(Rectangle((time1[0], -box_height/2),
                                   self.times[1] - self.times[0],
                                   box_height,
                                   alpha=0.2,
                                   color='b'))
            ax.add_patch(Rectangle((time2[0], -box_height/2),
                                   self.times[3] - self.times[2],
                                   box_height,
                                   alpha=0.2,
                                   color='g'))
            ax.add_patch(Rectangle((time3[0], -box_height/2),
                                   self.times[5] - self.times[4],
                                   box_height,
                                   alpha=0.2,
                                   color='r'))
            ax.legend(('Pulse 1', 'Pulse 2', 'Pulse 3'))

        else:
            raise AttributeError('Experiment times and/or pulse frequencies have to be defined.')
        plt.show()
