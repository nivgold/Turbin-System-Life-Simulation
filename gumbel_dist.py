import numpy as np
import chaospy

class GumbelDist:

    def __init__(self, mu, beta):
        self.mu = mu
        self.beta = beta

    def estimate_params(self, n=500, is_halton=False, witH_sample=False):
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

        beta_estimator = (np.sum(dist_randoms) / len(dist_randoms)) - (np.sum(dist_randoms * np.power(np.e, -(dist_randoms / self.beta))) / np.sum(np.power(np.e, -(dist_randoms / self.beta))))
        mu_estimator = -self.beta * np.log((np.sum(np.power(np.e, -(dist_randoms / self.beta)))) / (len(dist_randoms)))

        if witH_sample:
            return mu_estimator, beta_estimator, dist_randoms
        else:
            return mu_estimator, beta_estimator


    @staticmethod
    def estimate_from_sample(dist_randoms):
        n = len(dist_randoms)
        dist_randoms = np.array(dist_randoms)

        from scipy import stats
        mu_estimator, beta_estimator = stats.gumbel_r.fit(dist_randoms)

        #
        # beta_estimator = (np.sum(dist_randoms) / len(dist_randoms)) - (np.sum(dist_randoms * np.power(np.e, -(dist_randoms / self.beta))) / np.sum(np.power(np.e, -(dist_randoms / self.beta))))
        # mu_estimator = -self.beta * np.log((np.sum(np.power(np.e, -(dist_randoms / self.beta)))) / (len(dist_randoms)))

        return mu_estimator, beta_estimator

    def _generate_numbers(self, ksi):
        t = self.mu - self.beta * np.log(-np.log(ksi))
        return t

    def generate_sample(self, size=500):
        dist_randoms = []

        for i in range(size):
            ksi = np.random.uniform(0, 1)
            t = self._generate_numbers(ksi)
            dist_randoms.append(t)

        return dist_randoms