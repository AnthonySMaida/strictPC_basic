# strictPC_basic
Skeleton for strict predictive coding for 2-layer MLP. Tested in Python 3.11 and PyTorch 2.0.

Main file is "2_layerMLP_strictPC.py". Helper file with plot functions is "pc_plot_utilities.py". This is probably what you want to experiment with.

The file "2_layerMLP_ODE.py" mimics the computation in "2_layerMLP_strictPC.py" but uses an ODE framework. It's purpose is to set the stage to use multicompartment neurons.
