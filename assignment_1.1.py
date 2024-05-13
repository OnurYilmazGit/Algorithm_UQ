import numpy as np
import random
import numpy as np

g = [1.3, 1.7, 1.0, 2.0, 1.3, 1.7, 2.0, 2.3, 2.0, 1.7, 1.3, 1.0, 2.0, 1.7, 1.7, 1.3, 2.0]

#TODO: Compute Mean and Variance using numpy and your own implementation and compare

# Compute mean using numpy
mean_np = np.mean(g)

# Compute variance using numpy
var_np = np.var(g)

# Compute mean using own implementation
mean_own = sum(g) / len(g)

# Compute variance using own implementation
var_own = sum((x - mean_own) ** 2 for x in g) / len(g)

# Compare the results
print("Mean (numpy):", mean_np)
print("Mean (own):", mean_own)

print("Variance (numpy):", var_np)
print("Variance (own):", var_own)