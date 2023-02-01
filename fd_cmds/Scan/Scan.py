import numpy as np
import matplotlib.pyplot as plt
from Transients import _transients as _trans


class Scan:
    def __init__(self, ExperimentObject):
         self.ExpObj = ExperimentObject

    def scan_1d(self, axis, scan_range, npts, fixed_at):
        def f(point):
            freqs = np.zeros(2)
            freqs[axis], freqs[axis-1] = scan_range[0]+np.diff(scan_range)/npts*point, fixed_at
            freqs = np.append(freqs, _trans.hztown(self.ExpObj.omegas[2]))
            self.ExpObj.set_pulse_freqs(freqs)
            print(point)
            return self.ExpObj.compute()
            
        out = np.fromfunction(np.vectorize(f), (npts,), dtype=int)
        plt.plot(np.linspace(*scan_range, npts), out)
        plt.xlabel(f'w{axis+1} (wn)')
        plt.ylabel('Intensity')
        plt.show()

    def scan_2d(self, scan_range1, scan_range2, size):
        def f(point1, point2):
            w1 = scan_range1[0]+np.diff(scan_range1)/size[0]*point1
            w2 = scan_range2[0]+np.diff(scan_range2)/size[0]*point2
            self.ExpObj.set_pulse_freqs([w1, w2, _trans.hztown(self.ExpObj.omegas[2])])
            print(point1, point2)
            return self.ExpObj.compute()
            
        out = np.fromfunction(np.vectorize(f), size, dtype=int)
        plt.imshow(np.transpose(out), cmap='bwr', origin='lower', extent=(*scan_range1, *scan_range2))
        plt.xlabel("w1 (wn)")
        plt.ylabel("w2 (wn)")
        plt.colorbar(ticks=np.linspace(out.min(), out.max(), 7))
        plt.show()