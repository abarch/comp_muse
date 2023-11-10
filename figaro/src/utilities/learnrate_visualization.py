import matplotlib.pyplot as plt
import numpy as np

save_name = "training_saves/figaro_vae_hidden"

a = np.load(f'{save_name}_loss.npy', mmap_mode='r')

print(a.shape)

a = a[0:20]

flat = a.flatten()
flat = flat[15:]

plt.plot(list(range(len(flat))), flat)
plt.plot([i * a.shape[1] + a.shape[1] for i in range(len(a[1:]))], np.average(a[1:], axis=1), 'ro-')
plt.show()
