import numpy as np
import torch
from utilities.dataset_encoder import EncoderDataSet
from torch.utils.data import DataLoader
import os
import pickle

from models.cvae import VAE  # this VAE is actually a cVAE


dataset = EncoderDataSet("./samples/figaro/encoder_hidden", classes=["Q1", "Q2", "Q3", "Q4"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_name = "training_saves"
os.makedirs(dir_name, exist_ok=True)
save_name = dir_name + "/vae_hidden"

# model parameter
encoding_size = [512, 128, 32, 16]
decoding_size = [16, 32, 128, 512]

model = VAE(encoding_size, 8, decoding_size, conditional=True, num_labels=dataset.num_labels())
model.to(device)

# define hyperparameter

# each item has about 200 bars. In one epoch only one bar is addressed. Within 2000 epochs we
# see each bar only once
epochs = 20
lr = 0.001
batch_size = 16
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# loss
def loss_fn(recon_x, x, mean, log_var):
    huber = torch.nn.functional.huber_loss(
        recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

    return huber + KLD

loss_training = []

sample_save_train = [[], [], []]

for epoch in range(epochs):
    loss_epoch = []
    print(f'starting with epoch {epoch + 1}')
    for iteration, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        recon_x, mean, log_var, z = model(x, y)

        sample_save_train[0] += mean.tolist()
        sample_save_train[1] += log_var.tolist()
        sample_save_train[2] += y.tolist()

        loss = loss_fn(recon_x, x, mean, log_var)

        loss_epoch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    loss_training.append(loss_epoch)
    print(f'average loss: {np.mean(loss_epoch)}')

loss_arr = np.array(loss_training)
with open(f'{save_name}_loss.npy', 'wb') as f:
    np.save(f, loss_arr)

with open(f'{save_name}_sampling_dist.npy', 'wb') as f:
    pickle.dump(sample_save_train, f)

torch.save(model.state_dict(), f'{save_name}_model.ckpt')
