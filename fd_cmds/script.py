from Transients import _transients as _trans
from Experiment import Experiment
from Scan import Scan


omegas = [2253, 3164, 77000] #in wn for the individual states.
gammas = [20, 50, 1000] #in wn for the coherences, not the individual states.
rabis = [50, 80, 1000] #in wn for the coherences, by definition. 
exp = Experiment(omegas, gammas, rabis)

d1 = -5000e-15
d2 = -5000e-15
exp.set_delays([d1, d2])

pw1 = 5000e-15
pw2 = 5000e-15
pw3 = 5000e-15
exp.set_pws([pw1, pw2, pw3])
exp.set_times()
exp.set_pulse_freqs([2253, 3164, omegas[2]])
exp.set_transitions([_trans.ket_abs, _trans.bra_abs, _trans.ket_abs])
exp.set_pm([2, -1, 3])
print(exp.times)
print(exp.compute())
exp.draw()


#s1 = Scan(exp)
#s1.scan_1d_freq(0, (2100, 2300), 31, 3164)
#s1.scan_1d_freq(1, (3100, 3200), 21, 2254)
#s1.scan_1d_delay(0, (0, 500), 26, 100)
#s1.scan_1d_delay(1, (0, 500), 26, 100)


s2 = Scan(exp)
s2.scan_2d_freq((2100, 2400), (3050, 3300), (61, 41))
#s2.scan_2d_delay((0, 400), (0,400), (21, 21))
