import matplotlib.pyplot as plt
import numdifftools as nd
import progressbar
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pickle

def sign_val(a):
    if(a>1e-4):
        return 1
    elif(a<-1e-4):
        return -1
    else:
        return 0
def coth(x):
    return 1 / math.tanh(x)
plank_constant = 1
tau = 5
omega_1 = 0.1
omega_2 = 0.5
beta_1 = 1.0
beta_2 = 0.75
adam_beta_1 = 0.9
adam_beta_2 = 0.999
adam_learning_rate = 0.05
adam_epsilon = 1e-8
constant_part = coth(beta_1*plank_constant*omega_1)*plank_constant*math.sqrt(3)/(4*tau)

def reLU(val):
  return np.maximum(0, val)
def reLU_derivative(z):
    if(z>0):
        return 1
    if(z<0):
        return 0
    if(z==0):
        print("z = 0 , NN is giving same value at both zero and tau ")
def simpsons_3_8_rule_integration(func, a = 0, b = 0, n=999):
    h = (b - a) / n
    integral_sum = func(a) + func(b)  
    
    for i in range(1, n):
        x_i = a + i * h
        if i % 3 == 0:
            integral_sum += 2 * func(x_i)  
        else:
            integral_sum += 3 * func(x_i) 
    
    integral_approximation = (3 * h / 8) * integral_sum
    return integral_approximation


def function_derivative_integrand(val,val_1,val_2,val_theta,val_1_theta,val_2_theta):
    val = torch.ones_like(val_theta) * val
    val_1 = torch.ones_like(val_theta) * val_1
    val_2 = torch.ones_like(val_theta) * val_2
    sign_integrand = (-0.75)*(val_1**2)/(val**2) + (0.5)*(val_2/val)
    sign = torch.where(sign_integrand > 0, torch.tensor(1), torch.tensor(-1))
    
    return ((-0.75) * ((2 * val_1 * (val ** (-3)) * val_1_theta) + (-3) * (val_1 ** 2) * ((val ** (-4)) * val_theta)) +
          (0.5) * (((val ** (-2)) * val_2_theta) + ((-2) * (val ** (-3)) * val_theta * val_2)))*sign
def Cab_derivative_wrt_param(integral_part_derivative_wrt_param):
    return constant_part*integral_part_derivative_wrt_param



class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define the layers
        
        self.input_layer = nn.Linear(10, 100)
        self.hidden_layer1 = nn.Linear(100, 100)
        self.hidden_layer2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)

    def forward(self, x):
        # Define the forward pass
        x = torch.sigmoid(self.input_layer(x))
        x = torch.sigmoid(self.hidden_layer1(x))
        x = torch.sigmoid(self.hidden_layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x
    
    def calculate_gradients_param(self,t):
        input_data = torch.tensor([[1,t,t**2,t**3,t**4,t**5,t**6,t**7,t**8,t**9]], dtype=torch.float32, requires_grad=True)
        output = self.forward(input_data)
        parameters = list(self.parameters())
        gradients_zeroth_derivative = []
        for param in parameters:
            gradient_0th = torch.autograd.grad(output, param, retain_graph=True , allow_unused = True)
            gradients_zeroth_derivative.append(gradient_0th[0])
        grad_output = torch.autograd.grad(output, input_data, create_graph=True,allow_unused = True)
        grad_output = grad_output[0][0][1] 
        gradients_first_derivative = []
        for param in parameters:
            gradient_1st = torch.autograd.grad(grad_output, param, retain_graph=True,allow_unused = True)
            gradients_first_derivative.append(gradient_1st[0])
        grad_grad_output = torch.autograd.grad(grad_output, input_data, create_graph=True,allow_unused = True)
        grad_grad_output = grad_grad_output[0][0][1]
        gradients_second_derivative = []
        for param in parameters:
            gradient_2nd = torch.autograd.grad(grad_grad_output, param, retain_graph=True,allow_unused = True)
            gradients_second_derivative.append(gradient_2nd[0])
        return  gradients_zeroth_derivative , gradients_first_derivative , gradients_second_derivative , output , grad_output , grad_grad_output
    def gradients_wrt_time(self,t):
        input_data = torch.tensor([[1,t,t**2,t**3,t**4,t**5,t**6,t**7,t**8,t**9]], dtype=torch.float32, requires_grad=True)
        output = self.forward(input_data)
        grad_output = torch.autograd.grad(output, input_data, create_graph=True,allow_unused = True)
        grad_output = grad_output[0][0][1] 
        grad_grad_output = torch.autograd.grad(grad_output, input_data, create_graph=True,allow_unused = True)
        grad_grad_output = grad_grad_output[0][0][1]
        return output,grad_output,grad_grad_output
    def C_AB(self,):
        Integral_part = simpsons_3_8_rule_integration(self.Integrand_function , b = tau)
        return constant_part*Integral_part
    def cost_function(self,P):
        output_time0,grad_output_time0,grad_grad_output_time0 = self.gradients_wrt_time(0)
        output_time0,grad_output_time0,grad_grad_output_time0 = output_time0.item(),grad_output_time0.item(),grad_grad_output_time0.item()
        output_timetau,grad_output_timetau,grad_grad_output_timetau = self.gradients_wrt_time(tau)
        output_timetau,grad_output_timetau,grad_grad_output_timetau = output_timetau.item(),grad_output_timetau.item(),grad_grad_output_timetau.item()
        return self.C_AB() + P[0]*(abs(output_time0 - omega_1)) +\
                          P[1]*(abs(grad_output_time0)) +\
                          P[2]*(abs(grad_grad_output_time0)) +\
                          P[3]*(abs(output_timetau - omega_2)) +\
                          P[4]*(abs(grad_output_timetau)) +\
                          P[5]*(abs(grad_grad_output_timetau)) +\
                          P[6]*reLU(output_time0-output_timetau)    
        # return  P[3]*(abs(output_timetau - omega_2)) + P[0]*(abs(output_time0 - omega_1))
        # return self.C_AB()
    def Integrand_function(self,t,sign=False):
        val , val_1 , val_2 = self.gradients_wrt_time(t)   
        val , val_1 , val_2 = val.item() , val_1.item() , val_2.item()
   
        return (abs( ( ( -3*(val_1**2) )/( 4*(val**2) ) ) + (val_2 / (2*val )) ) /   val ) 

def main():
    P_val = np.array([
        [0.01, 0.01, 0.01, 0.95, 0.01, 0.01, 2],
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 2],
        [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 2],
        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 2]
    ])
    cost_fun = []
    param_step = []
    bar = progressbar.ProgressBar(maxval=1000, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for initial_conditions in range(1):
        bar.update(initial_conditions+1)
        net = CustomNet()
        parameters = list(net.parameters())
        # for using the saved parameters again for training uncomment the following code with parameters saved as pickle file (parameters.pkl)
        # file_name = "parameters.pkl"
        # with open(file_name, "rb") as file:
        #     param_step = pickle.load(file) 
        # i = 0
        # for param in net.parameters():
        #   param.data = param_step[len(param_step)-1][i]
        #   i =  i + 1
        loss_func_param_gradient = [torch.zeros_like(param) for param in parameters]
        cost_fun.append([])

        for pass_idx in range(4):
            m_curr = [torch.zeros_like(param) for param in parameters]
            v_curr = [torch.zeros_like(param) for param in parameters]
            m_prev = [torch.zeros_like(param) for param in parameters]
            v_prev = [torch.zeros_like(param) for param in parameters]
            t = 0
            P = P_val[pass_idx]
            for step in range(500):
                t = t + 1
                n=99
                h = tau/n
                a = 0
                b = tau
                gradients_zeroth_derivative_a, gradients_first_derivative_a , gradients_second_derivative_a , output_a , grad_output_a , grad_grad_output_a = net.calculate_gradients_param(a)
                gradients_zeroth_derivative_b, gradients_first_derivative_b , gradients_second_derivative_b , output_b , grad_output_b , grad_grad_output_b  = net.calculate_gradients_param(b)

                for i in range(len(loss_func_param_gradient)):

                    loss_func_param_gradient[i] = function_derivative_integrand(output_a.item(),grad_output_a.item(),grad_grad_output_a.item(),gradients_zeroth_derivative_a[i],gradients_first_derivative_a[i],gradients_second_derivative_a[i]) + \
                    function_derivative_integrand(output_b.item(),grad_output_b.item(),grad_grad_output_b.item(),gradients_zeroth_derivative_b[i],gradients_first_derivative_b[i],gradients_second_derivative_b[i])
                for integration_step in range(n):
                    x_i = a + integration_step * h
                    gradients_zeroth_derivative, gradients_first_derivative , gradients_second_derivative , output , grad_output , grad_grad_output = net.calculate_gradients_param(x_i)
                    for i in range(len(loss_func_param_gradient)):    
                        if (integration_step+1) % 3 == 0:
                            loss_func_param_gradient[i] = loss_func_param_gradient[i] + 2*function_derivative_integrand(output.item(),grad_output.item(),grad_grad_output.item(),gradients_zeroth_derivative[i],gradients_first_derivative[i],gradients_second_derivative[i])
                        else:
                            loss_func_param_gradient[i] = loss_func_param_gradient[i] + 3*function_derivative_integrand(output.item(),grad_output.item(),grad_grad_output.item(),gradients_zeroth_derivative[i],gradients_first_derivative[i],gradients_second_derivative[i])

                for j in range(len(loss_func_param_gradient)):    
                    loss_func_param_gradient[j] = (3 * h / 8) * loss_func_param_gradient[j]  


                for j in range(len(loss_func_param_gradient)):  
                    term_1 = sign_val(output_a.item()-omega_1)*P[0]*(gradients_zeroth_derivative_a[j])
                    term_2 = sign_val(grad_output_a.item())*P[1]*(gradients_first_derivative_a[j])
                    term_3 = sign_val(grad_grad_output_a.item())*P[2]*(gradients_second_derivative_a[j])
                    term_4 = sign_val(output_b.item()-omega_2)*P[3]*(gradients_zeroth_derivative_b[j])

                    term_5 = sign_val(grad_output_b.item())*P[4]*(gradients_first_derivative_b[j])
                    term_6 = sign_val(grad_grad_output_b.item())*P[5]*(gradients_second_derivative_b[j])
                    term_7 = P[6]*(gradients_zeroth_derivative_a[j] - gradients_zeroth_derivative_b[j])*reLU_derivative(output_a.item()-output_b.item())
                    if(reLU_derivative(output_a.item()-output_b.item())):
                        term_7 = P[6]*(gradients_zeroth_derivative_a[j] - gradients_zeroth_derivative_b[j])
                    else:
                        term_7 = 0*(gradients_zeroth_derivative_a[j] - gradients_zeroth_derivative_b[j])
                    loss_func_param_gradient[j] = loss_func_param_gradient[j] + term_4 + term_1 + term_2 + term_3 + term_5 + term_6 + term_7



                for i in range(len(loss_func_param_gradient)):   

                    m_curr[i] = (adam_beta_1 * m_prev[i] + (1 - adam_beta_1) * loss_func_param_gradient[i] )
                    v_curr[i] = (adam_beta_2 * v_prev[i] + (1 - adam_beta_2) * ( loss_func_param_gradient[i]** 2))

                    m_hat = m_curr[i] / (1 - adam_beta_1**t)
                    v_hat = v_curr[i] / (1 - adam_beta_2**t)
                    m_prev[i] = m_curr[i]
                    v_prev[i] = v_curr[i]

                    # m_curr[i] = ((adam_beta_1 * m_prev[i])- (1 - adam_beta_1) * loss_func_param_gradient[i] )
                    # v_curr[i] = ((adam_beta_2 * v_prev[i]) + (1 - adam_beta_2) * ( loss_func_param_gradient[i]** 2))
                    # delta_w = (-adam_learning_rate)*m_curr[i]*loss_func_param_gradient[i] / (torch.sqrt(v_curr[i]) + adam_epsilon)
                    # print(v_curr[i])

                    # parameters[i] = parameters[i] - delta_w
                    parameters[i] = parameters[i] - (adam_learning_rate * m_hat / (torch.sqrt(v_hat) + adam_epsilon))
                    # parameters[i] = parameters[i] - (adam_learning_rate)*loss_func_param_gradient[i]
                i = 0    
                for param in net.parameters():
                    param.data = parameters[i]
                    i = i + 1
                cost_fun[initial_conditions].append(net.cost_function(P_val[0]))
        param_step.append(parameters)

    with open('parrot.pkl', 'wb') as f:
        pickle.dump(cost_fun, f)
    with open('parameters.pkl', 'wb') as f:
        pickle.dump(param_step, f)    


if __name__ == "__main__":
    main()
