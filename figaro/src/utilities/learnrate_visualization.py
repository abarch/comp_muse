import matplotlib.pyplot as plt
import numpy as np

save_name = "training_saves/vae_hidden"

a = np.load(f'{save_name}_loss.npy', mmap_mode='r')

flat = a.flatten()

plt.plot(list(range(len(flat))), flat)
plt.plot([i * 68 + 68 for i in range(len(a))], np.average(a, axis=1), 'ro-')
plt.show()

print('Hello')
