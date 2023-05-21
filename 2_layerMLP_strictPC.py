#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Wed Apr 12 13:29:09 2023. Updated: May 19.
Skeletal implementation of strict PC alg from Rosenbaum (2022).
"On the relationship between predictive coding and backpropagation,"
PLoS ONE, 17(3): e0266102.

Implements Table 8 from unpublished manuscript by Maida.
Works for only one training sample of MNIST to assess convergence.
Seems to work. Desired values of v2 are y = [0,0,0,0,0,1,0,0,0,0].
Plot "v2 values" shows the v2 outputs approach y but don't overshoot.
@author: maida
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F                # one_hot()
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
import pc_plot_utils

"""
Helper functions
"""
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)
    
def relu_deriv(x): # not used
    return torch.where(x>0, torch.ones_like(x), torch.zeros_like(x))

def layer1(W0,v0):
    return torch.matmul(W0, v0)

def layer2(W1, v1):
    return torch.matmul(W1, relu(v1))
"""
Load MNIST data
"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081))])

trainset = torchvision.datasets.MNIST(root='.data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.MNIST(root='.data', train=False, download=True, transform=transform)
"""
Get 1st image as input.
Check if it's reasonable.
"""
image, label = trainset[0]
image2D_np = image.squeeze().numpy()   # cvts from (1,28,28) to (28,28) & cvrts to np

plt.imshow(image2D_np, cmap='gray')
plt.title(f"Label: {label}")
plt.show()

"""
Initializations
"""
torch.manual_seed(1) # for reproducibility to test effect of code changes
eta         = 0.02                         # rate param for v update
image2D     = torch.from_numpy(image2D_np) # cvrts from np to tensor
num_hiddens = 10
num_classes = 10
y           = F.one_hot(torch.tensor(label), num_classes = num_classes) # 1-hot label
gamma       = 0.002 # Rate param for wt update. Not used in paper, but is in original code.
equi_steps  = 50    # "n" in the original paper
wt_eps      = 50    # epochs of wt training

"""
Allocate wts, do forward sweep
"""
v0 = torch.flatten(image2D)  # Flatten tensor. v1: 1x784. Also see nn.Flatten()
W0 = torch.normal(0, 0.01, size=(num_hiddens,784), requires_grad=False)
v1 = layer1(W0, v0)  # num_hiddens
W1 = torch.normal(0, 0.01, size=(num_classes, num_hiddens), requires_grad=False)
v2 = layer2(W1, v1)
v1_at_t0 = v1
v2_at_t0 = v2

def print_sanity_checks(v0, v1, v2, W1):
    print(f"torch.t(v0) size: {torch.t(v0).size()}")
    print(f"v1 size: {v1.size()}")
    print(f"relu(v1) size: {relu(v1).size()}")
    print(f"v2 size: {v2.size()}")
    print(f"W1 size: {W1.size()}")
print_sanity_checks(v0, v1, v2, W1)

"""
Solo equilibrium calc for 1 epoch, to get feet wet.
Results, used to generate "Equilibrium Only" plot.
"""
for i in range(equi_steps):
    v2       = torch.matmul(W1, relu(v1))  # redundant on 1st iter b/c of alreaded computed by forward sweep
    e2     = v2 - y
    e1     = v1 - torch.matmul(W0, v0)
    delta_v1 = -e1 + torch.matmul(e2, jacobian(layer2, (W1,v1))[1])
    v1 = v1 + eta * delta_v1
        
v1_at_end = v1  # v2 will be reset later on
v2_at_end = v2  # v2 will be reset later on

pc_plotter = pc_plot_utils.PcPlotUtils()
pc_plotter.plot_equilib_calcs('v1', v1_at_t0, v1_at_end, 10, equi_steps)
pc_plotter.plot_equilib_calcs('v2', v2_at_t0, v2_at_end, num_classes, equi_steps)

"""
Set up storage to save v2 values for each class as training proceeds.
Do full computation w equilibrium calculation and TRAINING for 1 sample.
Present the same sample for wt_eps.
"""
v2_values = [[] for _ in range(num_classes)] # Samuel's idea
v1_values = [[] for _ in range(10)]

# Redo forward sweep (wts haven't changed yet.)
v1 = layer1(W0, v0)  # num_hiddens x 1
v2 = layer2(W1, v1)

loss   = [None] * (wt_eps + 1)

torch.manual_seed(1) # for reproducibility to test effect of code changes
for ep in range(wt_eps):
    for i in range(equi_steps):
        v2       = torch.matmul(W1, relu(v1))  # redundant on 1st iter b/c already computed by forward sweep
        e2     = v2 - y
        e1     = v1 - torch.matmul(W0, v0)
        delta_v1 = -e1 + torch.matmul(e2, jacobian(layer2, (W1,v1))[1])
        v1 = v1 + eta * delta_v1
    # training in next 4 lines
    delta_W0 = torch.matmul(-e1, jacobian(layer1, (W0,v0))[0])
    W0       = W0 + gamma*delta_W0
    delta_W1 = torch.matmul(-e2, jacobian(layer2, (W1,v1))[0])
    W1       = W1 + gamma*delta_W1
    print(f"Finished {ep+1} epoch(s). \ne1: {e1}; \ne2: {e2}; \nv1: {v1}; \nv2: {v2}\n")
    loss[ep+1] = 2*torch.sum(y-v2)**2 # record loss at end of epoch
    for j in range(num_classes): # save data for plots
        v2_values[j].append(v2[j].item())    
    for j in range(10): # save data for plots
        v1_values[j].append(v1[j].item())      

pc_plotter.plot_convergence_process('v1', v1_values, 10, wt_eps, equi_steps)
pc_plotter.plot_convergence_process('v2', v2_values, num_classes, wt_eps, equi_steps, ylim1=-1, ylim2=2)

v1_at_end = layer1(W0, v0)
v2_at_end = layer2(W1, v1_at_end)

# This plot shows v2 values after training is complete on the
# single training example 5. It appears to be learning correctly.
pc_plotter.plot_unit_convergence("v2", v2_at_t0, v2_at_end, num_classes, 
                                 wt_eps, equi_steps, ylim1=-1, ylim2=2)
pc_plotter.plot_loss("", wt_eps, equi_steps, loss)





