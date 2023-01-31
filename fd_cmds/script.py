from Transients import _transients as _trans
from Experiment import Experiment
from Scan import Scan

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
exp.set_pulse_freqs([2253, 3164, omegas[2]])
exp.set_transitions([_trans.bra_abs, _trans.ket_abs, _trans.ket_abs])
exp.set_pm([-1, 2, 3])
#exp.draw(spacing=0.01)
#exp.compute()
#e


#s1 = Scan(exp)
#s1.scan_1d(0, (2100, 2300), 31, 3164)
#s1.scan_1d(1, (3100, 3200), 21, 2254)


#s2 = Scan(exp)
#s2.scan_2d((2100, 2400), (3100, 3300), (31, 21))
