#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: Sun Apr 16 14:28:51 2023
Uses ODEs to implement the equilibrium computation.
Does not yet use ODEs for weight update.
See Equation Sets 7 and 8.
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
Get 1st image from dataset as input.
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
eta         = 0.02                      # rate param for v update
image2D   = torch.from_numpy(image2D_np) # cvrts from np to tensor
num_classes = 10
num_hiddens = 10
y           = F.one_hot(torch.tensor(label), num_classes = num_classes) # 1-hot label
gamma       = 0.002
equi_steps  = 50
wt_eps      = 50

"""
Allocate wts, do forward sweep
"""
v0 = torch.flatten(image2D)  # Flatten tensor. v1: 1x784
W0 = torch.normal(0, 0.01, size=(num_hiddens,784), requires_grad=False)
v1 = layer1(W0, v0)
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
Solo ODE equilibrium calc for 1 epoch.
Results generate "Equilibrium Only" plot.
"""
delta_t    = 0.01                        # time step size
v2_hist    = [None] * (equi_steps + 1)   # save v vector for each t-step in a list 
v1_hist    = [None] * (equi_steps + 1)
e2_hist    = [None] * (equi_steps + 1)
e1_hist    = [None] * (equi_steps + 1)
v2_hist[0] = v2                 # init v2_hist[0] to output of forward pass
v1_hist[0] = v1
e2_hist[0] = torch.zeros(10)
e1_hist[0] = torch.zeros(num_hiddens)
tau_v      = 0.01  
tau_e      = 0.01
for t in range(equi_steps):
    v2_hist[t+1] = torch.matmul(W1,relu(v1_hist[t]))
    e2_hist[t+1] = (v2_hist[t] - y)
    e1_hist[t+1] = (v1_hist[t]-torch.matmul(W0, v0))
    delta_v1     = -e1_hist[t] + \
                   torch.matmul(e2_hist[t], jacobian(layer2, (W1, v1_hist[t]))[1])
    v1_hist[t+1] = v1_hist[t] + eta * delta_v1
        
pc_plotter = pc_plot_utils.PcPlotUtils()
# Plot below shows v1/v2 values before/after equilibrium calc.
pc_plotter.plot_equilib_calcs('ODE v1', v1_hist[0], v1_hist[equi_steps], 10, equi_steps)
pc_plotter.plot_equilib_calcs('ODE v2', v2_hist[0], v2_hist[equi_steps], num_classes, equi_steps)

"""
Do full computation w equilibrium and TRAINING for 1 sample.
"""
v2_values = [[] for _ in range(num_classes)] # Samuel's idea
v1_values = [[] for _ in range(10)]
e1_values = [[] for _ in range(10)]
e2_values = [[] for _ in range(num_classes)]


# Redo forward sweep (wts haven't been change yet.)
v1 = layer1(W0, v0)  # num_hiddens x 1 
v2 = layer2(W1, v1)
v2_hist    = [None] * (equi_steps + 1)   # save v vector for each t-step in a list 
v1_hist    = [None] * (equi_steps + 1)
e2_hist    = [None] * (equi_steps + 1)
e1_hist    = [None] * (equi_steps + 1)
v2_hist[0] = v2                 # init v2_hist[0] to output of forward pass
v1_hist[0] = v1
e2_hist[0] = torch.zeros(num_classes)
e1_hist[0] = torch.zeros(num_hiddens)

loss   = [None] * (wt_eps + 1)

torch.manual_seed(1) # for reproducibility to test effect of code changes
for ep in range(wt_eps):
    for t in range(equi_steps):
        # v2_hist[t+1] = torch.matmul(W1,relu(v1_hist[t]))
        # e2_hist[t+1] = v2_hist[t] - y
        # e1_hist[t+1] = v1_hist[t]-torch.matmul(W0, v0)
        # delta_v1     = -e1_hist[t] + \
        #                torch.matmul(e2_hist[t], jacobian(layer2, (W1, v1_hist[t]))[1])
        # v1_hist[t+1] = v1_hist[t] + eta * delta_v1
        v2_hist[t+1] = v2_hist[t] + \
                        delta_t * (1/tau_v) * (-v2_hist[t]+torch.matmul(W1,relu(v1_hist[t])))
        e2_hist[t+1] = e2_hist[t] + \
                        delta_t * (1/tau_e) * (-e2_hist[t] + (v2_hist[t] - y))
        e1_hist[t+1] = e1_hist[t] + \
                        delta_t * (1/tau_e) * (-e1_hist[t]+ (v1_hist[t]-torch.matmul(W0, v0)))
        v1_hist[t+1] = v1_hist[t] + delta_t * (1/tau_v) * \
                        (eta*(-e1_hist[t] + \
                              torch.matmul(e2_hist[t], \
                                          jacobian(layer2, (W1, v1_hist[t]))[1])))
    if ep == 0:
        pc_plotter.plot_equilib_calcs('ODE v1', v1_hist[0], v1_hist[equi_steps], 10, equi_steps)
        pc_plotter.plot_equilib_calcs('ODE v2', v2_hist[0], v2_hist[equi_steps], num_classes, equi_steps)
    # training in next 4 lines
    e1 = e1_hist[equi_steps]
    e2 = e2_hist[equi_steps]
    v1 = v1_hist[equi_steps]
    v2 = v2_hist[equi_steps]
    delta_W0 = torch.matmul(-e1, jacobian(layer1, (W0,v0))[0])
    W0       = W0 + gamma*delta_W0
    delta_W1 = torch.matmul(-e2, jacobian(layer2, (W1,v1))[0])
    W1       = W1 + gamma*delta_W1
    print(f"Finished {ep+1} epoch(s). \ne1: {e1}; \ne2: {e2}; \nv1: {v1}; \nv2: {v2}\n")
    loss[ep+1] = 2*torch.sum(y-v2)**2 # record loss at end of epoch
    for j in range(10): # save data for plots
        v2_values[j].append(v2_hist[ep+1][j].item())
        v1_values[j].append(v1_hist[ep+1][j].item())      
        e1_values[j].append(e1_hist[ep+1][j].item())
        e2_values[j].append(e2_hist[ep+1][j].item())
    e1_hist[0] = e1_hist[equi_steps]  # pass results from prev ep to start of next ep
    e2_hist[0] = e2_hist[equi_steps]  # pass results from prev ep to start of next ep
    v1_hist[0] = v1_hist[equi_steps]  # pass results from prev ep to start of next ep
    v2_hist[0] = v2_hist[equi_steps]  # pass results from prev ep to start of next ep
        
pc_plotter.plot_convergence_process('ODE v1', v1_values, 10, wt_eps, equi_steps)
pc_plotter.plot_convergence_process('ODE v2', v2_values, num_classes, wt_eps, equi_steps, ylim1=-1, ylim2=2)
pc_plotter.plot_convergence_process('ODE e1', e1_values, 10, wt_eps, equi_steps)
pc_plotter.plot_convergence_process('ODE e2', e2_values, 10, wt_eps, equi_steps)

v1_at_end = layer1(W0, v0)
v2_at_end = layer2(W1, v1_at_end)

# This plot shows v2 values after training is complete on the
# single training example 5. It appears to be learning correctly.
pc_plotter.plot_unit_convergence("ODE v2", v2_at_t0, v2_at_end, num_classes, 
                                 wt_eps, equi_steps, ylim1=-1, ylim2=2)

pc_plotter.plot_loss("ODE", wt_eps, equi_steps, loss)

# plt.title('ODE {0} Epochs Wt Training'.format(wt_eps))
# plt.plot(range(num_classes), v2_hist[0], label='0 steps')
# plt.plot(range(num_classes), v2_at_end, label='{0} steps'.format(wt_eps))
# plt.xlabel('v2 component')
# plt.ylabel('v2 value')
# plt.legend()
# plt.savefig('test.pdf', format='pdf')
# plt.show()





