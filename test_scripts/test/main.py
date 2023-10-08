import matplotlib.pyplot as plt
import numdifftools as nd
from functions import*
from script_neuron import CustomNet
import progressbar
import numpy as np
import math
import time
def coth(x):
    return 1 / math.tanh(x)
plank_constant = 1
tau = 0
omega_1 = 0
omega_2 = 0
beta_1 = 0
beta_2 = 0
constant_part = coth(beta_1*plank_constant*omega_1)*math.sqrt(3)/(4*tau)

def Integral_function(self,t):  
  w_0 = self.forward_pass(t)
  w_1 = self.forward_pass_first_derivative(t)
  w_2 = self.forward_pass_second_derivative(t)
  return (abs( ( ( -3*(w_1**2) )/( 4*(w_0**2) ) ) + (w_2 / (2*w_0 )) ) /   w_0 ) 

def C_AB(self):
   Integral_part = simpsons_3_8_rule_integration(self.Integral_function , b = tau)
   return constant_part*Integral_part

def cost_function(self):
    return self.C_AB() +  self.P[1]*(abs(self.forward_pass(0) - omega_1)) +\
                          self.P[2]*(abs(self.forward_pass_first_derivative(0))) +\
                          self.P[3]*(abs(self.forward_pass_second_derivative(0))) +\
                          self.P[4]*(abs(self.forward_pass(tau) - omega_2)) +\
                          self.P[5]*(abs(self.forward_pass_first_derivative(tau))) +\
                          self.P[6]*(abs(self.forward_pass_second_derivative(tau))) +\
                          self.P[7]*function.reLU(self.forward_pass(0)-self.forward_pass(tau))    

def main():
    P_val = np.array([
    [1, 1, 1, 1, 1, 1, 16],
    [2, 2, 2, 2, 2, 2, 16],
    [4, 4, 4, 4, 4, 4, 16],
    [8, 8, 8, 8, 8, 8, 16]
])
    bar = progressbar.ProgressBar(maxval=1000, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for initial_conditions in range(1000):
        bar.update(initial_conditions+1)
        # NN = Neural_network(W1, W2, W3, W4, b1, b2, b3, b4 ,gradient_W1 ,gradient_W2 ,gradient_W3 ,gradient_W4,gradient_b1,gradient_b2,gradient_b3,gradient_b4)
        # NN.tau = 20
        # NN.omega_1 = 0.1
        # NN.omega_2 = 0.5
        # NN.beta_1 = 1
        # NN.beta_2 = 0.75
        # del W1, W2, W3, W4, b1, b2, b3, b4 ,gradient_W1 ,gradient_W2 ,gradient_W3 ,gradient_W4,gradient_b1,gradient_b2,gradient_b3,gradient_b4 \
        net = CustomNet(input_size=1)
        for pass_idx in range(4):
            for epoch in range(1000):

                gradients_zeroth_derivative , gradients_first_derivative , gradients_second_derivative = net.calculate_gradients_param(t)
            # for epoch in range (1000):
                
    
