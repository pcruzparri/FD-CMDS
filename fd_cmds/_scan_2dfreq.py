__all__ = ['Experiment', 'CMDSProcess']

import numpy as np
import matplotlib.pyplot as plt
import _transientsv3 as _trans
import time





class Experiment():
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
    
    def get_pm(self)
        return self.pm
    
    def compute(self, time_int=100e15):
        """
        :param time_int:
        :return:
        run the calculation
        :return: output coherence amplitude
        """

        signs, orderings = np.sign(self.transitions), np.abs(self.transitions)
        GROUND_STATE_GAMMA = _trans.wntohz(1e-18)

        t1 = self.transitions[0](self.ExpConds.rabis[orderings[0]],
                               _trans.delta_ij(0, GROUND_STATE_GAMMA),
                               _trans.delta_ij(signs[0]*self.ExpConds.omegas[orderings[0]], self.ExpConds.gammas[0]), #note: these are coherence gammas
                               self.ExpConds.pulse_freqs[orderings[0]],
                               signs[0]*self.ExpConds.omegas[orderings[0]],
                               GROUND_STATE_GAMMA,
                               self.ExpConds.gammas[orderings[0]],
                               self.ExpConds.times[1])
        fid1 = _trans.fid(1,
                          _trans.delta_ij(self.ExpConds.omegas[orderings[0]], self.ExpConds.gammas[orderings[0]]),
                          self.ExpConds.times[3]-self.ExpConds.times[1])

        t2 = self.transitions[1](self.ExpConds.rabis[orderings[1]],
                               _trans.delta_ij(signs[0]*self.ExpConds.omegas[orderings[0]],
                                               self.ExpConds.gammas[orderings[1]]),
                               _trans.delta_ij(signs[0]*self.ExpConds.omegas[orderings[0]]
                                               + signs[1]*self.ExpConds.omegas[orderings[1]], self.ExpConds.gammas[orderings[1]]),
                               signs[0]*self.ExpConds.pulse_freqs[orderings[0]] \
                               + signs[1]*self.ExpConds.pulse_freqs[orderings[1]],
                               signs[0]*self.ExpConds.omegas[orderings[0]] + signs[1]*self.ExpConds.omegas[orderings[1]],
                               self.ExpConds.gammas[orderings[0]],
                               self.ExpConds.gammas[orderings[1]],
                               self.ExpConds.times[3] - self.ExpConds.times[2])
        fid2 = _trans.fid(1,
                          _trans.delta_ij(signs[0]*self.ExpConds.omegas[orderings[0]]
                                          + signs[1]*self.ExpConds.omegas[orderings[1]], self.ExpConds.gammas[orderings[1]]),
                          np.linspace(self.ExpConds.times[4] - self.ExpConds.times[3],
                                      self.ExpConds.times[5] - self.ExpConds.times[3],
                                      int((self.ExpConds.times[5] - self.ExpConds.times[4]) * time_int) + 1))
        t3 = self.transitions[2](self.ExpConds.rabis[orderings[2]],
                               _trans.delta_ij(signs[0]*self.ExpConds.omegas[orderings[0]]
                                               + signs[1]*self.ExpConds.omegas[orderings[1]], self.ExpConds.gammas[orderings[1]]),
                               _trans.delta_ij(signs[0]*self.ExpConds.omegas[orderings[0]]
                                               + signs[1]*self.ExpConds.omegas[orderings[1]]
                                               + signs[2]*self.ExpConds.omegas[orderings[2]], self.ExpConds.gammas[orderings[2]]),
                               signs[0]*self.ExpConds.pulse_freqs[orderings[0]]
                               + signs[1]*self.ExpConds.pulse_freqs[orderings[1]]
                               + signs[2]*self.ExpConds.pulse_freqs[orderings[2]],
                               signs[0]*self.ExpConds.omegas[orderings[0]]
                               + signs[1]*self.ExpConds.omegas[orderings[1]]
                               + signs[2]*self.ExpConds.omegas[orderings[2]],
                               self.ExpConds.gammas[orderings[1]],
                               self.ExpConds.gammas[orderings[2]],
                               np.linspace(0, self.ExpConds.times[5]-self.ExpConds.times[4],
                                           int((self.ExpConds.times[5]-self.ExpConds.times[4])*1e15*time_int)))
        coeff = t1*fid1*t2*fid2
        out_field = coeff*t3
        return np.sum(np.real(out_field*np.conjugate(out_field)))

    def dove_ir_1_freq_scan(self, scan_freqs, npts, queue=None, time_int=100):
        # scan parameters
        w1_center = scan_freqs[0]
        w2_center = scan_freqs[1]
        w1_range = scan_freqs[2]
        w2_range = scan_freqs[3]

        w1_scan_range = np.linspace(w1_center-w1_range, w1_center+w1_range, npts[0])
        w2_scan_range = np.linspace(w2_center-w2_range, w2_center+w2_range, npts[1])
        self.w1_scan_range = w1_scan_range
        self.w2_scan_range = w2_scan_range

        self.scan = np.zeros((len(w2_scan_range), len(w1_scan_range)))

        scan_start = time.time()
        remaining = len(w1_scan_range)*len(w2_scan_range)
        last_speed = [0]*5

        for ind2, w2 in enumerate(w2_scan_range):  # y axis
            for ind1, w1 in enumerate(w1_scan_range):  # x axis
                self.set_pulse_freqs([w1, w2, self.omegas[2]])

                time1 = time.time()

                ground_gamma = _trans.wntohz(1e-18)
                T1 = _trans.ket_abs(self.rabis[1],
                                        _trans.delta_ij(0, ground_gamma),
                                        _trans.delta_ij(_trans.wntohz(3164), self.gammas[1]),
                                        _trans.wntohz(w2),
                                        _trans.wntohz(3164),
                                        ground_gamma,
                                        self.gammas[1],
                                        self.times[1])
                FID1 = _trans.fid(1, _trans.delta_ij(_trans.wntohz(3164), self.gammas[1]),
                                        self.times[3]-self.times[1])
                T2 = _trans.bra_abs(self.rabis[0],
                                        _trans.delta_ij(_trans.wntohz(3164), self.gammas[1]),
                                        _trans.delta_ij(_trans.wntohz(3164-2253), self.gammas[1]),
                                        _trans.wntohz(w2-w1),
                                        self.omegas[1],
                                        self.gammas[1],
                                        self.gammas[0],
                                        self.times[3]-self.times[2])  # trans1, driven, 0 to t1
                FID2 = _trans.fid(1, _trans.delta_ij(self.omegas[1]-self.omegas[0], self.gammas[0]),
                                        np.linspace(self.times[4]-self.times[3], \
                                        self.times[5]-self.times[3], \
                                        int((self.times[5]-self.times[4])*1e15*time_int)+1))
                T3 = _trans.ket_abs(self.rabis[2],
                                        _trans.delta_ij(_trans.wntohz(3164-2253), self.gammas[1]),
                                        _trans.delta_ij(_trans.wntohz(3164-2253+9800), self.gammas[2]),
                                        _trans.wntohz(w2-w1+9800),
                                        _trans.wntohz(3164-2253+9800),
                                        self.gammas[0],
                                        self.gammas[2],
                                        np.linspace(0, self.times[5]-self.times[4], int((self.times[5]-self.times[4])*1e15*time_int)+1))

                coeff = T1*FID1*T2*FID2
                test = np.array(coeff*T3)




                self.scan[ind2][ind1] = np.sum(np.real((test * np.conjugate(test))))
                remaining -= 1

                time2 = time.time()
                round_time = round(time2-time1, 2)
                last_speed = last_speed[1:]+[round_time]
                print(f'Finished w1={w1} and w2={w2} | '
                    f'Calc. time was {round_time} s | '
                    f'Time remaining is {round(np.average(last_speed)*remaining/60, 2)} min')

        scan_end = time.time()
        print(f'Total calc. time was {scan_end-scan_start}s')
        if queue:
            queue.put((w2_center, self.scan))
            return self.scan
        else:
            return self.scan


    def dove_ir_2_freq_scan(self, scan_freqs, npts, queue=None, time_int=100):
        # scan parameters
        w1_center = scan_freqs[0]
        w2_center = scan_freqs[1]
        w1_range = scan_freqs[2]
        w2_range = scan_freqs[3]

        w1_scan_range = np.linspace(w1_center-w1_range, w1_center+w1_range, npts[0])
        w2_scan_range = np.linspace(w2_center-w2_range, w2_center+w2_range, npts[1])
        self.w1_scan_range = w1_scan_range
        self.w2_scan_range = w2_scan_range

        self.scan = np.zeros((len(w2_scan_range), len(w1_scan_range)))

        scan_start = time.time()
        remaining = len(w1_scan_range)*len(w2_scan_range)
        last_speed = [0]*5

        for ind2, w2 in enumerate(w2_scan_range):  # y axis
            for ind1, w1 in enumerate(w1_scan_range):  # x axis
                self.set_pulse_freqs([w1, w2, self.omegas[2]])

                time1 = time.time()

                ground_gamma = _trans.wntohz(1-18)
                T1 = _trans.bra_abs(self.rabis[0],
                                        _trans.delta_ij(0, ground_gamma),
                                        _trans.delta_ij(self.omegas[0], self.gammas[0]),
                                        self.pulse_freqs[0],
                                        self.omegas[0],
                                        ground_gamma,
                                        self.gammas[0],
                                        self.times[1])  # trans1, driven, 0 to t1
                FID1 = _trans.fid(1, _trans.delta_ij(self.omegas[0], self.gammas[0]), self.times[3]-self.times[1])
                T2 = _trans.ket_abs(self.rabis[1],
                                        _trans.delta_ij(self.omegas[0], self.gammas[0]),
                                        _trans.delta_ij(_trans.wntohz(self.omegas[1]-self.omegas[0]), self.gammas[1]),
                                        _trans.wntohz(w2-w1),
                                        _trans.wntohz(self.omegas[1]-self.omegas[0]),
                                        self.gammas[0],
                                        self.gammas[1],
                                        self.times[3]-self.times[2])
                FID2 = _trans.fid(1, _trans.delta_ij(_trans.wntohz(self.omegas[1]-self.omegas[0]), self.gammas[1]),
                                        np.linspace(self.times[4]-self.times[3], \
                                        self.times[5]-self.times[3], \
                                        int((self.times[5]-self.times[4])*1e15*time_int)+1))
                T3 = _trans.ket_abs(self.rabis[2],
                                        _trans.delta_ij(_trans.wntohz(self.omegas[1]-self.omegas[0]), self.gammas[1]),
                                        _trans.delta_ij(_trans.wntohz(self.omegas[1]-self.omegas[0])+self.omegas[2], self.gammas[2]),
                                        _trans.wntohz(w2-w1+self.omegas[2]),
                                        _trans.wntohz(self.omegas[1]-self.omegas[0]+self.omegas[2]),
                                        self.gammas[1],
                                        self.gammas[2],
                                        np.linspace(0, self.times[5]-self.times[4], int((self.times[5]-self.times[4])*1e15*time_int)+1))

                coeff = T1*FID1*T2*FID2
                test = np.array(coeff*T3)




                self.scan[ind2][ind1] = np.sum(np.real((test * np.conjugate(test))))
                remaining -= 1

                time2 = time.time()
                round_time = round(time2-time1, 2)
                last_speed = last_speed[1:]+[round_time]
                print(f'Finished w1={w1} and w2={w2} | '
                    f'Calc. time was {round_time} s | '
                    f'Time remaining is {round(np.average(last_speed)*remaining/60, 2)} min')

        scan_end = time.time()
        print(f'Total calc. time was {scan_end-scan_start}s')
        if queue:
            queue.put((w2_center, self.scan))
            return self.scan
        else:
            return self.scan


    def plot(self):
        if self.scan.any():
            # scan = gaussian_filter(scan, sigma=2)
            plt.imshow(self.scan, cmap='bwr', origin='lower', extent=(min(self.w1_scan_range),
                                                                 max(self.w1_scan_range),
                                                                 min(self.w2_scan_range),
                                                                 max(self.w2_scan_range)))
            plt.colorbar()
            plt.show()



class CMDSProcess():
    def __init__(self, ExperimentObject, transition_sequence, phase_matching):
        """
        :param transitions: list of transient function objects (required)
        DOVE-IR-II example: [bra_abs, ket_abs, ket_abs]

        :param pm: list of phase matching and ordering indices of the pulses as given in ExpObj. (required)
        DOVE-IR-II example: [-2, 1, 3]
        This example assumes pulse frequency list in ExpObj is something like:
        ["combination/overtone freq", "fundamental freq", "elect. freq"]
        This example reads as "the first pulse (based on index) is the first pulse frequency in ExpObj, out of phase.
        The second pulse is the second pulse frequency in ExpObj, in phase.
        The third pulse is the third pulse frequency in ExpObj, in phase."
        """
        self.ExpConds = ExperimentObject
        self.transitions = transition_sequence
        self.pm = phase_matching

    






