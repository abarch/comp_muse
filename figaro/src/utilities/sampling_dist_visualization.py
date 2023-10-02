import matplotlib.pyplot as plt
import numpy as np
import pickle

save_name = "training_saves/vae_hidden_sampling_dist.npy"

with open(save_name, "rb") as f:
    means, log_vars, y = pickle.load(f)

print(len(y))

unique_y = sorted(list(set(y)))
mean_y = [[] for _ in range(len(unique_y))]
log_var_y = [[] for _ in range(len(unique_y))]
for i in range(0,len(y)):
    label = y[i]
    mean_y[unique_y.index(label)].append(means[i])
    log_var_y[unique_y.index(label)].append(log_vars[i])


for i in range(len(unique_y)):
    plt.plot(list(range(len(mean_y[i]))), np.array(mean_y[i])[:,:])
    plt.title(f"mean for y={unique_y[i]}")
    plt.show()

    plt.plot(list(range(len(log_var_y[i]))), np.array(log_var_y[i])[:,:])
    plt.title(f"log_var for y={unique_y[i]}")
    plt.show()
