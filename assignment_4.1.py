import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Function to compute relative error
def get_rel_err(approx, ref):
    return np.abs(approx - ref) / ref

# Standard Monte Carlo Sampling
def std_mcs(func, n_samples):
    samples = np.random.uniform(0, 1, n_samples)
    values = func(samples)
    return np.mean(values)

# Control Variate
def control_variate(func, cv, integral_cv_eval, n_samples):
    samples = np.random.uniform(0, 1, n_samples)
    values = func(samples)
    cv_values = cv(samples)
    cv_mean = np.mean(cv_values)
    func_mean = np.mean(values)
    # Adjusted mean
    adjusted_mean = func_mean - (cv_mean - integral_cv_eval)
    return adjusted_mean

# Importance Sampling
def importance_sampling(func, a, b, n_samples):
    samples = np.random.beta(a, b, size=n_samples)
    values = func(samples) * (samples**(a-1) * (1-samples)**(b-1)) / beta.pdf(samples, a, b)
    return np.mean(values)

if __name__ == '__main__':
    # Declare the function to integrate
    func = lambda x: np.exp(x)
    reference_integral = np.exp(1) - 1  # e - 1

    # Declare vector with number of samples
    N = [10, 100, 1000, 10000, 100000]

    # Declare the control variates
    cv = [lambda x: x, lambda x: 1 + x]
    integral_cv = [0.5, 1.5]  # Integrals of x over [0, 1] and 1 +

    # Parameters for the beta distribution
    a = [5, 0.5]
    b = [1, 0.5]

    # Arrays to store the relative errors
    rel_err_mcs = []
    rel_err_cv = [[] for _ in integral_cv]
    rel_err_ip = [[] for _ in range(len(a))]

    # Perform Monte Carlo integration for each sample size
    for n in N:
        # Standard Monte Carlo
        mcs_estimate = std_mcs(func, n)
        rel_err_mcs.append(get_rel_err(mcs_estimate, reference_integral))
        
        # Control Variate
        for i, cv_func in enumerate(cv):
            cv_estimate = control_variate(func, cv_func, integral_cv[i], n)
            rel_err_cv[i].append(get_rel_err(cv_estimate, reference_integral))
        
        # Importance Sampling
        for j in range(len(a)):
            ip_estimate = importance_sampling(func, a[j], b[j], n)
            rel_err_ip[j].append(get_rel_err(ip_estimate, reference_integral))

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.loglog(N, rel_err_mcs, 'o-', label='Standard MC')
    for i, errors in enumerate(rel_err_cv):
        plt.loglog(N, errors, 's--', label=f'Control Variate {i+1}')
    for j, errors in enumerate(rel_err_ip):
        plt.loglog(N, errors, 'x:', label=f'Importance Sampling {j+1}')
    
    plt.xlabel('Number of samples')
    plt.ylabel('Relative error')
    plt.title('Relative Error of Monte Carlo Methods')
    plt.legend()
    plt.grid(True)
    plt.show()
