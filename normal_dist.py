import numpy as np
import chaospy

class NormalDist:

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def estimate_params(self, n=500, is_halton=False, with_sample=False):
        dist_randoms = []

        if is_halton:
            uniform = chaospy.Uniform(0, 1)
            halton_sequence = uniform.sample(n/2, rule='halton')
            for i in range(len(halton_sequence)):
                ksi1 = halton_sequence[i]
                ksi2 = halton_sequence[i]
                t1, t2 = self._generate_numbers(ksi1, ksi2)
                dist_randoms.append(t1)
                dist_randoms.append(t2)
        else:
            for i in range(int(n/2)):
                ksi1 = np.random.uniform(0, 1)
                ksi2 = ksi1
                t1, t2 = self._generate_numbers(ksi1, ksi2)
                dist_randoms.append(t1)
                dist_randoms.append(t2)

        dist_randoms = np.array(dist_randoms)

        mu_estimator = np.sum(dist_randoms) / len(dist_randoms)
        sigma_estimator = np.sqrt((np.sum(np.power(dist_randoms - mu_estimator, 2))) / (len(dist_randoms)))

        if with_sample:
            return mu_estimator, sigma_estimator, dist_randoms
        else:
            return mu_estimator, sigma_estimator

    def _generate_numbers(self, ksi1, ksi2):
        t1 = np.sqrt(-2*np.log(ksi1)) * np.sin(2*np.pi*ksi2)
        t1 = t1 * self.sigma + self.mu

        t2 = np.sqrt(-2*np.log(ksi1)) * np.cos(2*np.pi*ksi2)
        t2 = t2 * self.sigma + self.mu

        return t1, t2

    def generate_sample(self, size=500):
        dist_randoms = []

        for i in range(int(size / 2)):
            ksi1 = np.random.uniform(0, 1)
            ksi2 = ksi1
            t1, t2 = self._generate_numbers(ksi1, ksi2)
            dist_randoms.append(t1)
            dist_randoms.append(t2)

        return dist_randoms