import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Function to integrate
    func = lambda x: np.sin(x)

    # Number of samples at different magnitudes
    N = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

    # Initialize arrays to store results
    I_hat = np.zeros(len(N))  # Approximate Monte Carlo integral results
    I = 2  # Exact integral of sin(x) from 0 to pi
    est_std_dev = np.zeros(len(N))  # Monte Carlo standard error
    rms = np.zeros(len(N))  # Root Mean Square error between estimated and true value

    # Integration bounds
    a, b = 0, np.pi

    # Monte Carlo integration process
    for i in range(len(N)):
        # Generate random samples from uniform distribution between a and b
        samples = np.random.uniform(a, b, N[i])
        # Evaluate the function at these sample points
        evaluations = func(samples)
        # Calculate the integral estimate
        I_hat[i] = (b - a) * np.mean(evaluations)
        # Calculate the standard deviation of the samples
        est_std_dev[i] = (b - a) * np.std(evaluations) / np.sqrt(N[i])
        # Calculate RMS error
        rms[i] = np.sqrt((I_hat[i] - I) ** 2)

    # Plotting results
    plt.figure(figsize=(10, 5))

    # Plotting the estimates
    plt.subplot(1, 2, 1)
    plt.loglog(N, I_hat, 'o-', label='Monte Carlo Estimate')
    plt.axhline(y=I, color='r', linestyle='-', label='Exact Integral')
    plt.xlabel('Number of Samples')
    plt.ylabel('Integral Estimate')
    plt.title('Monte Carlo Integration Estimates')
    plt.legend()

    # Plotting the standard deviation and RMS error
    plt.subplot(1, 2, 2)
    plt.loglog(N, est_std_dev, 'o-', label='Estimated Std Dev')
    plt.loglog(N, rms, 's-', label='RMS Error')
    plt.xlabel('Number of Samples')
    plt.ylabel('Error')
    plt.title('Error Analysis')
    plt.legend()

    plt.tight_layout()
    plt.show()
