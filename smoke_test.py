import torch
import torch.nn as nn
from notebook_units import definitions
torch.set_num_threads(2)
scope=definitions(['VAE'], {'torch':torch,'nn':nn})
model=scope['VAE']()
x=torch.randn(8,30)
recon,mu,lv=model(x)
assert recon.shape==x.shape and mu.shape==lv.shape==(8,2)
loss=(recon-x).square().mean() - .5*(1+lv-mu.square()-lv.exp()).mean()
loss.backward()
assert torch.isfinite(loss)
assert model.fc_mu.weight.grad.abs().sum()>0
assert model.dec_out.weight.grad.abs().sum()>0
print('VAE: reconstruction/latent shapes and encoder/decoder gradients passed.')
