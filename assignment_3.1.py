import numpy as np
import matplotlib.pyplot as plt

N = [10, 100, 1000, 10000]
mean = np.array([-0.4, 1.1])
V = np.array([[1.0, 0.4], [0.4, 1.0]])

mean_errors = []
cov_errors = []

for n in N:
    # Sample from multivariate normal
    samples = np.random.multivariate_normal(mean, V, n)
    
    # Computing mean and covariance via Monte Carlo
    mc_mean = np.mean(samples, axis=0)
    mc_cov = np.cov(samples, rowvar=False)
    
    # Computing errors
    mean_errors.append(np.linalg.norm(mc_mean - mean))
    cov_errors.append(np.linalg.norm(mc_cov - V))

plt.figure()

# Plot mean errors
plt.subplot(1, 2, 1)
plt.loglog(N, mean_errors, marker='o')
plt.title('Mean Errors')
plt.xlabel('Number of samples')
plt.ylabel('Error')
plt.grid(True)

# Plot for covariance errors
plt.subplot(1, 2, 2)
plt.loglog(N, cov_errors, marker='o')
plt.title('Covariance Errors')
plt.xlabel('Number of samples')
plt.ylabel('Error')
plt.grid(True)

plt.tight_layout()
plt.show()