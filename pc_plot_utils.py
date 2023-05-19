#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:03:20 2023

@author: maida
"""
import matplotlib.pyplot as plt

class PcPlotUtils:
    def __init__(self):
        self.dummy = 1
    
    # plot equilibrium calculations
    # strng specifies layer
    def plot_equilib_calcs(self, strng, start, end, x_range, n):
        fig = plt.figure()
        ax  = fig.add_subplot(1,1,1)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.title(strng + ' Equilibrium', fontsize = 16)
        plt.plot(range(x_range), start, label = '0 steps')
        plt.plot(range(x_range), end, label = '{0} steps'.format(n))
        plt.xlabel(strng + ' component', fontsize=16)
        plt.ylabel(strng + ' value', fontsize=16)
        plt.legend(fontsize=12)
        fig.subplots_adjust(bottom=0.2)
        plt.savefig('{0}_equilib_converge{1}eqio.pdf'.format(strng, n), format='pdf')
        plt.show()

        
    def plot_convergence_process(self, strng, values, x_range, epochs, n, ylim1=-10, ylim2=10):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.gca().set_xlim(0, epochs)
        plt.gca().set_ylim(ylim1, ylim2)
        for j in range(x_range):
            ax.plot(range(epochs), values[j], label=f"{strng}[{j}]", lw=3.0)
        ax.set_title("{0} vals {1} Eps Training {2} Units".format(strng,epochs,x_range), fontsize= 16)
        plt.ylabel('{0} value'.format(strng), fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        #ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.legend(loc = 'right', fontsize=12)
        fig.subplots_adjust(bottom=0.2)
        plt.savefig('z4_{0}_convergence_process{1}equi{2}eps.pdf'.format(strng,n,epochs), format='pdf')
        plt.show()
        
    def plot_unit_convergence(self, strng, start, end, x_range, epochs, n, ylim1=-10, ylim2=10):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.gca().set_ylim(ylim1, ylim2)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.title("{0} {1} Epochs Wt Training".format(strng, epochs), fontsize=16)
        plt.plot(range(x_range), start, label='0 steps', linewidth=3.0)
        plt.plot(range(x_range), end, label=str(epochs)+' wt update steps', linewidth=3.0)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.xlabel(strng + " component", fontsize=16)
        plt.ylabel(strng + " value", fontsize=16)
        plt.legend(fontsize=12)
        fig.subplots_adjust(bottom=0.2)
        plt.savefig("z2_{0}_wt_converge{1}equi{2}eps.pdf".format(strng,n,epochs), format='pdf')
        plt.show()
        
    def plot_loss(self, epochs, n, loss):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.gca().set_xlim(0, epochs)
        #plt.gca().set_ylim(.7, .85)  # set plot bounds by hand
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        plt.title('Loss: {0} Epochs Wt Training'.format(epochs), fontsize=16)
        plt.plot(range(epochs), loss[1:], lw = 3.0)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
        fig.subplots_adjust(bottom=0.2)
        plt.savefig('z3_loss{0}equi{1}eps.pdf'.format(n,epochs), format='pdf')
        plt.show()
        
        
        
        
        
        