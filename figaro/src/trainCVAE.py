import numpy as np
import torch
from utilities.dataset_encoder import EncoderDataSet
from models.cvae import VAE  # this VAE is actually a cVAE

from torch.utils.data import DataLoader

dataset = EncoderDataSet("./samples/encoder_hidden")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_name = "vae_hidden"

# model parameter
encoding_size = [512, 128, 32, 16]
decoding_size = [16, 32, 128, 512]

model = VAE(encoding_size, 8, decoding_size, conditional=True, num_labels=4)
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
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)

loss_training = []

for epoch in range(epochs):
    loss_epoch = []
    for iteration, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        recon_x, mean, log_var, z = model(x, y)

        loss = loss_fn(recon_x, x, mean, log_var)

        loss_epoch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    loss_training.append(loss_epoch)

loss_arr = np.array(loss_training)
with open(f'{save_name}_loss.npy', 'wb') as f:
    np.save(f, loss_arr)

torch.save(model.state_dict(), f'{save_name}_loss.npy')
