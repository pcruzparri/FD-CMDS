import _transientsv3 as _trans
from Experiment import Experiment
import matplotlib.pyplot as plt
import numpy as np
import timeit
import os
from multiprocessing import Process, Queue
import time


omegas = [2253, 3164, 77000] #in wn for the individual states.
gammas = [20, 50, 1000] #in wn for the coherences, not the individual states.
rabis = [5, 8, 1000] #in wn for the coherences, by definition. 
exp = Experiment(omegas, gammas, rabis)

d1 = 200e-15
d2 = 200e-15
delays = [d1, d2]
exp.set_delays(delays)

pw1 = 100e-15
pw2 = 500e-15
pw3 = 500e-15
pws = [pw1, pw2, pw3]
exp.set_pws(pws)
exp.set_times()
exp.set_pulse_freqs([2253, 3164, omegas[2]])
exp.set_transitions([_trans.bra_abs, _trans.ket_abs, _trans.ket_abs])
exp.set_pm([-1, 2, 3])
#exp.draw(spacing=0.01)

e = exp.compute()

def f(w1,w2):
    omegas = [2253, 3164, 77000] #in wn for the individual states.
    gammas = [20, 50, 1000] #in wn for the coherences, not the individual states.
    rabis = [5, 8, 1000] #in wn for the coherences, by definition. 
    exp = Experiment(omegas, gammas, rabis)

    d1 = 200e-15
    d2 = 200e-15
    delays = [d1, d2]
    exp.set_delays(delays)

    pw1 = 500e-15
    pw2 = 500e-15
    pw3 = 500e-15
    pws = [pw1, pw2, pw3]
    exp.set_pws(pws)
    exp.set_times()
    exp.set_pulse_freqs([w1*30+1900, w2*30+2850, omegas[2]])
    exp.set_transitions([_trans.ket_abs, _trans.bra_abs, _trans.ket_abs])
    exp.set_pm([2, -1, 3])
    print(w1, w2)
    return exp.compute()



"""
n=400
t = timeit.Timer(lambda: f(2200,3300)).timeit(n)
print(t/n)
"""

"""
matrix = np.fromfunction(np.vectorize(f), (20, 20), dtype=int)
plt.imshow(np.transpose(matrix), cmap='bwr', origin='lower')
plt.colorbar()
plt.show()
"""

"""
w1_scan = np.linspace(1800, 2600, 81)
w2_scan = np.linspace(2800, 3600, 81)
out_scan = []
for i in w1_scan:
    print(i)
    exp.set_pulse_freqs([i, 3164, omegas[2]])
    out_scan.append(exp.compute())

plt.plot(w1_scan, out_scan)
plt.show()
"""
