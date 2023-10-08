# Final code 
from functions import*
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def coth(x):
    return 1 / math.tanh(x)
plank_constant = 1
tau = 0
omega_1 = 0
omega_2 = 0
beta_1 = 0
beta_2 = 0
constant_part = coth(beta_1*plank_constant*omega_1)*math.sqrt(3)/(4*tau)

 



    
start_time = time.time()



class CustomNet(nn.Module):
    def __init__(self, input_size):
        super(CustomNet, self).__init__()
        # Define the layers
        self.input_layer = nn.Linear(input_size, 100)
        self.hidden_layer1 = nn.Linear(100, 100)
        self.hidden_layer2 = nn.Linear(100, 100)
        self.hidden_layer3 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)

    def forward(self, x):
        # Define the forward pass
        x = torch.sigmoid(self.input_layer(x))
        x = torch.sigmoid(self.hidden_layer1(x))
        x = torch.sigmoid(self.hidden_layer2(x))
        x = torch.sigmoid(self.hidden_layer3(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
    def calculate_gradients_param(self,t):
        input_data = torch.tensor([[t]], dtype=torch.float32, requires_grad=True)
        output = net(input_data)
        parameters = list(net.parameters())
        gradients_zeroth_derivative = []
        for param in parameters:
            gradient_0th = torch.autograd.grad(output, param, retain_graph=True)
            gradients_zeroth_derivative.append(gradient_0th)
        grad_output = torch.autograd.grad(output, input_data, create_graph=True)[0] 
        gradients_first_derivative = []
        for param in parameters:
            gradient_1st = torch.autograd.grad(grad_output, param, retain_graph=True)
            gradients_first_derivative.append(gradient_1st)
        grad_grad_output = torch.autograd.grad(grad_output, input_data, create_graph=True)[0]
        gradients_second_derivative = []
        for param in parameters:
            gradient_2nd = torch.autograd.grad(grad_grad_output, param, retain_graph=True)
            gradients_second_derivative.append(gradient_2nd)
        return  gradients_zeroth_derivative , gradients_first_derivative , gradients_second_derivative
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


# Create an instance of the custom neural network with input size 1
# net = CustomNet(input_size=1).to(device)
net = CustomNet(input_size=1)


for i in range(1000):
    num = np.random.rand(1 , 1)[0][0] - 0.5

    # Example input tensor with shape (batch_size, input_size)
    # input_data = torch.tensor([[num]], dtype=torch.float32, requires_grad=True).to(device)
    input_data = torch.tensor([[num]], dtype=torch.float32, requires_grad=True)


    # Forward pass through the network
    output = net(input_data)
    #####
    parameters = list(net.parameters())

    gradients_zeroth_derivative = []
    for param in parameters:
        gradient_0th = torch.autograd.grad(output, param, retain_graph=True)
        gradients_zeroth_derivative.append(gradient_0th)



    #####

    grad_output = torch.autograd.grad(output, input_data, create_graph=True)[0]

    ####


    # Calculate the gradients of the derivative with respect to network parameters (weights and biases)
    gradients_first_derivative = []
    for param in parameters:
        gradient_1st = torch.autograd.grad(grad_output, param, retain_graph=True)
        gradients_first_derivative.append(gradient_1st)

    ####

    grad_grad_output = torch.autograd.grad(grad_output, input_data, create_graph=True)[0]

    # Compute the gradients of the second derivative with respect to network parameters (weights and biases)

    gradients_second_derivative = []
    for param in parameters:
        gradient_2nd = torch.autograd.grad(grad_grad_output, param, retain_graph=True)
        gradients_second_derivative.append(gradient_2nd)

    # Print the second derivative of the output with respect to the input
    end_time = time.time()

    # print("First Derivative of Output with Respect to Input:")
    # print(grad_output)
    # print("Second Derivative of Output with Respect to Input:")
    # print(grad_grad_output)

elapsed_time = end_time - start_time
print(f"Time taken to run the program: {elapsed_time} seconds")

