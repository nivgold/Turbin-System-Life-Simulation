import numpy as np
from time import time
from collections import defaultdict
from scipy.stats import chisquare
from scipy import stats


from exponential_dist import ExponentialDist
from gumbel_dist import GumbelDist
from log_normal_dist import LogNormalDist
from normal_dist import NormalDist
from weibull_dist import WeibullDist

blade = {'mu': 42000, 'sigma': 663}
gearbox = {'mu': 11, 'sigma': 1.2}
generator = {'ni': 76000, 'm': 1.2}
yaw = {'mu': 65000, 'beta': 370}
pitch = {'mu': 84534, 'sigma': 506}
brake = {'lmd': 1/120000}
lubrication = {'ni': 66000, 'm': 1.3}
electrical = {'ni': 35000, 'm': 1.5}
frequency = {'lmd': 1/45000}

components_params = {'blade': blade, 'gearbox': gearbox, 'generator': generator, 'yaw': yaw, 'pitch': pitch, 'brake': brake, 'lubrication': lubrication, 'electrical': electrical, 'frequency': frequency}

def print_estimators(**kwargs):
    print('-' * 100)
    print('Blade\nNormal Distribution params estimations:\nReal mu:%s, sigma:%s\nEstimated mu:%s, sigma:%s\ntime:%s' % kwargs['blade'])

    print('-' * 100)
    print('Gearbox\nLogarithmic Normal Distribution params estimations:\nReal params mu:%s, sigma:%s\nEstimated mu:%s, sigma:%s\ntime:%s' % kwargs['gearbox'])

    print('-' * 100)
    print('Generator\nWeibull Distribution params estimations:\nReal ni:%s, m:%s\nEstimated ni:%s, m:%s\ntime:%s' % kwargs['generator'])

    print('-' * 100)
    print('Yaw\nGumbel Distribution params estimations:\nReal mu:%s, beta:%s\nEstimated mu:%s, beta:%s\ntime:%s' % kwargs['yaw'])

    print('-' * 100)
    print('Pitch\nNormal Distribution params estimations:\nReal mu:%s, sigma:%s\nEstimated mu:%s, sigma:%s\ntime:%s' % kwargs['pitch'])

    print('-' * 100)
    print('Brake\nExponential Distribution params estimations:\nReal lambda:%s\nEstimated lambda:%s\ntime:%s' % kwargs['brake'])

    print('-' * 100)
    print('Lubrication\nWeibull Distribution params estimations:\nReal ni:%s, m:%s\nEstimated ni:%s, m:%s\ntime:%s' % kwargs['lubrication'])

    print('-' * 100)
    print('Electrical\nWeibull Distribution params estimations:\nReal ni:%s, m:%s\nEstimated ni:%s, m:%s\ntime:%s' % kwargs['electrical'])

    print('-' * 100)
    print('Frequency\nExponential Distribution params estimations:\nReal lambda:%s\nEstimated lambda:%s\ntime:%s' % kwargs['frequency'])
    print('-' * 100)

def create_simulation_estimators(size, halton=False, to_print=True, calculate_minimum_dist=False, **kwargs):
    components_estimators = {}
    estimators_print = {}

    blade_dist = NormalDist(**kwargs['blade'])
    start_time = time()
    if calculate_minimum_dist:
        return_tuple = blade_dist.estimate_params(size, halton, with_sample=True)
        blade_estimators = return_tuple[0], return_tuple[1]
        blade_sample = return_tuple[2]
    else:
        blade_estimators = blade_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['blade'] = tuple(kwargs['blade'].values()) + blade_estimators + (end_time - start_time, )
    components_estimators['blade'] = blade_estimators

    gearbox_dist = LogNormalDist(**kwargs['gearbox'])
    start_time = time()
    if calculate_minimum_dist:
        return_tuple = gearbox_dist.estimate_params(size, halton, with_sample=True)
        gearbox_estimator = return_tuple[0], return_tuple[1]
        gearbox_sample = return_tuple[2]
    else:
        gearbox_estimator = gearbox_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['gearbox'] = tuple(kwargs['gearbox'].values()) + gearbox_estimator + (end_time-start_time,)
    components_estimators['gearbox'] = gearbox_estimator

    generator_dist = WeibullDist(**kwargs['generator'])
    start_time = time()
    if calculate_minimum_dist:
        return_tuple = generator_dist.estimate_params(size, halton, with_sample=True)
        generator_estimator = return_tuple[0], return_tuple[1]
        generator_sample = return_tuple[2]
    else:
        generator_estimator = generator_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['generator'] = tuple(kwargs['generator'].values()) + generator_estimator + (end_time - start_time,)
    components_estimators['generator'] = generator_estimator

    yaw_dist = GumbelDist(**kwargs['yaw'])
    start_time = time()
    if calculate_minimum_dist:
        return_tuple = yaw_dist.estimate_params(size, halton, witH_sample=True)
        yaw_estimator = return_tuple[0], return_tuple[1]
        yaw_sample = return_tuple[2]
    else:
        yaw_estimator = yaw_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['yaw'] = tuple(kwargs['yaw'].values()) + yaw_estimator + (end_time - start_time,)
    components_estimators['yaw'] = yaw_estimator

    pitch_dist = NormalDist(**kwargs['pitch'])
    start_time = time()
    if calculate_minimum_dist:
        return_tuple = pitch_dist.estimate_params(size, halton, with_sample=True)
        pitch_estimator = return_tuple[0], return_tuple[1]
        pitch_sample = return_tuple[2]
    else:
        pitch_estimator = pitch_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['pitch'] = tuple(kwargs['pitch'].values()) + pitch_estimator + (end_time - start_time,)
    components_estimators['pitch'] = pitch_estimator

    brake_dist = ExponentialDist(**kwargs['brake'])
    start_time = time()
    if calculate_minimum_dist:
        brake_estimator, brake_sample = brake_dist.estimate_params(size, halton, with_sample=True)
    else:
        brake_estimator = brake_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['brake'] = list(kwargs['brake'].values())[0], brake_estimator, end_time - start_time
    components_estimators['brake'] = brake_dist.estimate_params(size, halton)

    lubrication_dist = WeibullDist(**kwargs['lubrication'])
    start_time = time()
    if calculate_minimum_dist:
        return_tuple = lubrication_dist.estimate_params(size, halton, with_sample=True)
        lubrication_estimator = return_tuple[0], return_tuple[1]
        lubrication_sample = return_tuple[2]
    else:
        lubrication_estimator = lubrication_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['lubrication'] = tuple(kwargs['lubrication'].values()) + lubrication_estimator + (end_time - start_time,)
    components_estimators['lubrication'] = lubrication_estimator

    electrical_dist = WeibullDist(**kwargs['electrical'])
    start_time = time()
    if calculate_minimum_dist:
        return_tuple = electrical_dist.estimate_params(size, halton, with_sample=True)
        electrical_estimator = return_tuple[0], return_tuple[1]
        electrical_sample = return_tuple[2]
    else:
        electrical_estimator = electrical_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['electrical'] = tuple(kwargs['electrical'].values()) + electrical_estimator + (end_time - start_time,)
    components_estimators['electrical'] = electrical_estimator

    frequency_dist = ExponentialDist(**kwargs['frequency'])
    start_time = time()
    if calculate_minimum_dist:
        frequency_estimator, frequency_sample = frequency_dist.estimate_params(size, halton, with_sample=True)
    else:
        frequency_estimator = frequency_dist.estimate_params(size, halton)
    end_time = time()
    estimators_print['frequency'] = list(kwargs['frequency'].values())[0], frequency_estimator, end_time-start_time
    components_estimators['frequency'] = frequency_estimator

    if to_print:
        print_estimators(**estimators_print)

    if calculate_minimum_dist:
        # calculating the minimum distribution
        concat = np.concatenate(
            [blade_sample, gearbox_sample, generator_sample, yaw_sample, pitch_sample, brake_sample, lubrication_sample,
             electrical_sample, frequency_sample], axis=0).reshape(9, size)
        minimum_dist = np.min(concat, axis=0)
        return components_estimators, minimum_dist

    return components_estimators


def empiric_confidence(**kwargs):
    for sample_type in kwargs:
        sample = kwargs[sample_type]
        lower_10th_decile = np.percentile(sample, 10)
        upper_10th_decile = np.percentile(sample, 90)
        print('Empiric Confidence: ', end='')
        print("P{%s <= %s <= %s} = 0.9" % (lower_10th_decile, sample_type, upper_10th_decile))

def normalic_confidence(**kwargs):
    z_val = 0.8289
    for sample_type in kwargs:
        sample = kwargs[sample_type]
        mu_estimate = np.sum(sample) / len(sample)
        sigma_estimate = np.sqrt(np.sum(np.power(sample - mu_estimate, 2)) / len(sample))
        left = mu_estimate - z_val * (sigma_estimate / np.sqrt(len(sample)))
        right = mu_estimate + z_val * (sigma_estimate / np.sqrt(len(sample)))
        print("Normalic Confidence: ", end='')
        print("P{%s <= %s <= %s} = 0.9" % (left, sample_type, right))


def ex1_b():
    print('+'+'-'*98+'+')
    print('|'+' '*46+'ex. 1b'+' '*46+'|')
    print('+'+'-'*98+'+')

    # setting the seed
    seed = 1
    np.random.seed(seed)
    create_simulation_estimators(500, **components_params)

def ex1_c():
    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 40 + 'ex. 1c - 500 seed 1' + ' ' * 39 + '|')
    print('+' + '-' * 98 + '+')

    # setting the seed
    seed = 1
    np.random.seed(seed)
    create_simulation_estimators(500, **components_params)

    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 40 + 'ex. 1c - 500 seed 5' + ' ' * 39 + '|')
    print('+' + '-' * 98 + '+')

    # setting the seed
    seed = 5
    np.random.seed(seed)
    create_simulation_estimators(500, **components_params)

    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 39 + 'ex. 1c - 10000 seed 3' + ' ' * 38 + '|')
    print('+' + '-' * 98 + '+')

    # setting the seed
    seed = 3
    np.random.seed(seed)
    create_simulation_estimators(10000, **components_params)

def ex1_d():
    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 46 + 'ex. 1d' + ' ' * 46 + '|')
    print('+' + '-' * 98 + '+')

    blade_estimators = defaultdict(list)
    gearbox_estimators = defaultdict(list)
    generator_estimators = defaultdict(list)
    yaw_estimators = defaultdict(list)
    pitch_estimators = defaultdict(list)
    brake_estimators = defaultdict(list)
    lubrication_estimators = defaultdict(list)
    electrical_estimators = defaultdict(list)
    frequency_estimators = defaultdict(list)

    for i in range(100):
        # setting the seed
        seed = i
        np.random.seed(seed)
        components_estimators = create_simulation_estimators(500, halton=False, to_print=False, **components_params)

        blade_estimators['mu'].append(components_estimators['blade'][0])
        blade_estimators['sigma'].append(components_estimators['blade'][1])

        gearbox_estimators['mu'].append(components_estimators['gearbox'][0])
        gearbox_estimators['sigma'].append(components_estimators['gearbox'][1])

        generator_estimators['ni'].append(components_estimators['generator'][0])
        generator_estimators['m'].append(components_estimators['generator'][1])

        yaw_estimators['mu'].append(components_estimators['yaw'][0])
        yaw_estimators['beta'].append(components_estimators['yaw'][1])

        pitch_estimators['mu'].append(components_estimators['pitch'][0])
        pitch_estimators['sigma'].append(components_estimators['pitch'][1])

        brake_estimators['lmd'].append(components_estimators['brake'])

        lubrication_estimators['ni'].append(components_estimators['lubrication'][0])
        lubrication_estimators['m'].append(components_estimators['lubrication'][1])

        electrical_estimators['ni'].append(components_estimators['electrical'][0])
        electrical_estimators['m'].append(components_estimators['electrical'][1])

        frequency_estimators['lmd'].append(components_estimators['frequency'])

    print('-' * 100)
    print('Blade Confidences:')
    empiric_confidence(**blade_estimators)
    normalic_confidence(**blade_estimators)

    print('-' * 100)
    print('Gearbox Confidences:')
    empiric_confidence(**gearbox_estimators)
    normalic_confidence(**gearbox_estimators)

    print('-' * 100)
    print('Generator Confidences:')
    empiric_confidence(**generator_estimators)
    normalic_confidence(**generator_estimators)

    print('-' * 100)
    print("Yaw Confidences:")
    empiric_confidence(**yaw_estimators)
    normalic_confidence(**yaw_estimators)

    print('-' * 100)
    print("Pitch Confidences:")
    empiric_confidence(**pitch_estimators)
    normalic_confidence(**pitch_estimators)

    print('-' * 100)
    print("Brake Confidences:")
    empiric_confidence(**brake_estimators)
    normalic_confidence(**brake_estimators)

    print('-' * 100)
    print("Lubrication Confidences:")
    empiric_confidence(**lubrication_estimators)
    normalic_confidence(**lubrication_estimators)

    print('-' * 100)
    print("Frequency Confidences:")
    empiric_confidence(**frequency_estimators)
    normalic_confidence(**frequency_estimators)

def ex1_e():
    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 43 + 'ex. 1e - 50' + ' ' * 44 + '|')
    print('+' + '-' * 98 + '+')

    create_simulation_estimators(50, halton=True, **components_params)

    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 43 + 'ex. 1e - 200' + ' ' * 43 + '|')
    print('+' + '-' * 98 + '+')

    create_simulation_estimators(200, halton=True, **components_params)

    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 43 + 'ex. 1e - 500' + ' ' * 43 + '|')
    print('+' + '-' * 98 + '+')

    create_simulation_estimators(500, halton=True, **components_params)

def ex2_a():
    np.random.seed(987654321)
    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 46 + 'ex. 2a' + ' ' * 46 + '|')
    print('+' + '-' * 98 + '+')

    # estimating the parameters from 500 sample
    components_estimators, minimum_dist = create_simulation_estimators(500, halton=True, to_print=False, calculate_minimum_dist=True, **components_params)
    return components_estimators, minimum_dist

def ex2_b(components_estimators, minimum_dist):
    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 46 + 'ex. 2b' + ' ' * 46 + '|')
    print('+' + '-' * 98 + '+')

    components_samples_dict = {}

    blade_dist = NormalDist(components_estimators['blade'][0], components_estimators['blade'][1])
    components_samples_dict[('Blade', 'Normal')] = blade_dist.generate_sample(500)

    gearbox_dist = LogNormalDist(components_estimators['gearbox'][1], components_estimators['gearbox'][1])
    components_samples_dict[('Gearbox', 'Logarithmic Normal')] = gearbox_dist.generate_sample(500)

    generator_dist = WeibullDist(components_estimators['generator'][0], components_estimators['generator'][1])
    components_samples_dict[('Generator', 'Weibull')] = generator_dist.generate_sample(500)

    yaw_dist = GumbelDist(components_estimators['yaw'][0], components_estimators['yaw'][1])
    components_samples_dict[('Yaw', 'Gumbel')] = yaw_dist.generate_sample(500)

    pitch_dist = NormalDist(components_estimators['pitch'][0], components_estimators['pitch'][1])
    components_samples_dict[('Pitch', 'Normal')] = pitch_dist.generate_sample(500)

    brake_dist = ExponentialDist(components_estimators['brake'])
    components_samples_dict[('Brake', 'Exponential')] = brake_dist.generate_sample(500)

    lubrication_dist = WeibullDist(components_estimators['lubrication'][0], components_estimators['lubrication'][1])
    components_samples_dict[('Lubrication', 'Weibull')] = lubrication_dist.generate_sample(500)

    electrical_dist = WeibullDist(components_estimators['electrical'][0], components_estimators['electrical'][1])
    components_samples_dict[('Electrical', 'Weibull')] = electrical_dist.generate_sample(500)

    frequency_dist = ExponentialDist(components_estimators['frequency'])
    components_samples_dict[('Frequency', 'Exponential')] = frequency_dist.generate_sample(500)

    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 41 + 'Chi-Square Tests' + ' ' * 41 + '|')
    print('+' + '-' * 98 + '+')

    for component, component_sample in components_samples_dict.items():
        print(f'Chi-Square Test Between Minimum Distribution and {component[0]} {component[1]} Distribution:')
        print(chisquare(component_sample, minimum_dist))
        print()

    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 37 + 'Kolmogorov–Smirnov Tests' + ' ' * 37 + '|')
    print('+' + '-' * 98 + '+')

    for component, component_sample in components_samples_dict.items():
        print(f'Kolmogorov–Smirnov Test Between Minimum Distribution and {component[0]} {component[1]} Distribution:')
        print(stats.kstest(component_sample, minimum_dist))
        print()

    print('+' + '-' * 98 + '+')
    print('|' + ' ' * 38 + 'Anderson–Darling Tests' + ' ' * 38 + '|')
    print('+' + '-' * 98 + '+')

    for component, component_sample in components_samples_dict.items():
        print(f'Anderson–Darling Test On {component[0]} {component[1]} Distribution:')
        print(stats.anderson(component_sample))
        print()

if __name__ == '__main__':
    ex1_b()
    ex1_c()
    ex1_d()
    ex1_e()
    ex2_b(*ex2_a())