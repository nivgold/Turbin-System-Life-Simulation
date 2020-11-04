import numpy as np
import chaospy

class ExponentialDist:

    def __init__(self, lmd):
        self.lmd = lmd

    def estimate_params(self, n=500, is_halton=False, with_sample=False):
        dist_randoms = []

        if is_halton:
            uniform = chaospy.Uniform(0, 1)
            halton_sequence = uniform.sample(n, rule='halton')
            for i in range(len(halton_sequence)):
                ksi = halton_sequence[i]
                t = self._generate_numbers(ksi)
                dist_randoms.append(t)
        else:
            for i in range(n):
                ksi = np.random.uniform(0, 1)
                t = self._generate_numbers(ksi)
                dist_randoms.append(t)

        dist_randoms = np.array(dist_randoms)

        lmd_estimator = n / np.sum(dist_randoms)

        if with_sample:
            return lmd_estimator, dist_randoms
        else:
            return lmd_estimator

    def _generate_numbers(self, ksi):
        t = - (np.log(ksi) / self.lmd)

        return t

    def generate_sample(self, size=500):
        dist_randoms = []

        for i in range(size):
            ksi = np.random.uniform(0, 1)
            t = self._generate_numbers(ksi)
            dist_randoms.append(t)

        return dist_randoms