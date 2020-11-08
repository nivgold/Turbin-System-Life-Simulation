import numpy as np
import chaospy

class WeibullDist:

    def __init__(self, ni, m):
        self.m = m
        self.ni = ni

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

        ni_estimator = np.power(np.sum(np.power(dist_randoms, self.m) / len(dist_randoms)), 1 / self.m)
        m_estimator = 1 / ((np.sum(np.power(dist_randoms, self.m) * np.log(dist_randoms)) / np.sum(np.power(dist_randoms, self.m))) - np.sum(np.log(dist_randoms)) / len(dist_randoms))

        if with_sample:
            return ni_estimator, m_estimator, dist_randoms
        else:
            return ni_estimator, m_estimator


    @staticmethod
    def estimate_from_sample(dist_randoms):
        n = len(dist_randoms)
        dist_randoms = np.array(dist_randoms)

        from scipy import stats
        m_estimator, loc, ni_estimator = stats.weibull_min.fit(dist_randoms, floc=0)

        # ni_estimator = np.power(np.sum(np.power(dist_randoms, self.m) / len(dist_randoms)), 1 / self.m)
        # m_estimator = 1 / ((np.sum(np.power(dist_randoms, self.m) * np.log(dist_randoms)) / np.sum(np.power(dist_randoms, self.m))) - np.sum(np.log(dist_randoms)) / len(dist_randoms))

        return ni_estimator, m_estimator

    def _generate_numbers(self, ksi):
        t = self.ni * np.power(-np.log(ksi), (1 / self.m))

        return t

    def generate_sample(self, size=500):
        dist_randoms = []

        for i in range(size):
            ksi = np.random.uniform(0, 1)
            t = self._generate_numbers(ksi)
            dist_randoms.append(t)

        return dist_randoms