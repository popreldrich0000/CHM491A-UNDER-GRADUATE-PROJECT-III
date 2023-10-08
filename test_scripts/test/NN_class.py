from functions import*
import numpy as np
import math
def coth(x):
    return 1 / math.tanh(x)
plank_constant = 1
tau = 0
omega_1 = 0
omega_2 = 0
beta_1 = 0
beta_2 = 0
constant_part = coth(beta_1*plank_constant*omega_1)*math.sqrt(3)/(4*tau)




class Neural_network:
  P = np.zeros((1,7))
  def __init__(self,W1, W2, W3, W4, b1, b2, b3, b4\
               ,gradient_W1_0 ,gradient_W2_0 ,gradient_W3_0 ,gradient_W4_0\
               ,gradient_b1_0,gradient_b2_0,gradient_b3_0,gradient_b4_0\
               ,gradient_W1_1 , gradient_W1_2 , gradient_W2_1 , gradient_W2_2\
               ,gradient_W3_1 , gradient_W3_2 , gradient_W4_1 , gradient_W4_2\
               ,gradient_b1_1 , gradient_b1_2 , gradient_b2_1 , gradient_b2_2\
               ,gradient_b3_1 , gradient_b3_2 , gradient_b4_1 , gradient_b4_2\
              ,A1,A2,A3,A4,A1_deri_1,A2_deri_1,A3_deri_1,A4_deri_1\
              ,layer1,layer2,layer3,layer4\
              ,layer1_deri_1,layer2_deri_1,layer3_deri_1,layer4_deri_1):
      self.W1 = W1 
      self.gradient_W1_0 = gradient_W1_0
      self.gradient_W1_1 = gradient_W1_1
      self.gradient_W1_2 = gradient_W1_2
      self.W2 = W2
      self.gradient_W2_0 = gradient_W2_0
      self.gradient_W2_1 = gradient_W2_1
      self.gradient_W2_2 = gradient_W2_2
      self.W3 = W3
      self.gradient_W3_0 = gradient_W3_0
      self.gradient_W3_1 = gradient_W3_1
      self.gradient_W3_2 = gradient_W3_2
      self.W4 = W4
      self.gradient_W4_0 = gradient_W4_0
      self.gradient_W4_1 = gradient_W4_1
      self.gradient_W4_2 = gradient_W4_2
      self.b1 = b1
      self.gradient_b1_0 = gradient_b1_0
      self.gradient_b1_1 = gradient_b1_1
      self.gradient_b1_2 = gradient_b1_2
      self.b2 = b2
      self.gradient_b2_0 = gradient_b2_0
      self.gradient_b2_1 = gradient_b2_1
      self.gradient_b2_2 = gradient_b2_2
      self.b3 = b3
      self.gradient_b3_0 = gradient_b3_0
      self.gradient_b3_1 = gradient_b3_1
      self.gradient_b3_2 = gradient_b3_2
      self.b4 = b4
      self.gradient_b4_0 = gradient_b4_0
      self.gradient_b4_1 = gradient_b4_1
      self.gradient_b4_2 = gradient_b4_2
      self.A1 = A1
      self.A2 = A2
      self.A3 = A3
      self.A4 = A4
      self.A1_deri_1 = A1_deri_1
      self.A2_deri_1 = A2_deri_1
      self.A3_deri_1 = A3_deri_1
      self.A4_deri_1 = A4_deri_1
      self.layer1 = layer1
      self.layer2 = layer2
      self.layer3 = layer3
      self.layer4 = layer4
      self.layer1_deri_1 = layer1_deri_1
      self.layer2_deri_1 = layer2_deri_1
      self.layer3_deri_1 = layer3_deri_1
      self.layer4_deri_1 = layer4_deri_1

  def forward_pass(self,t,cache_nodes = False):
    x = np.array([[t]])
    if(cache_nodes):
      self.A1 = np.add(np.matmul(self.W1,x) , self.b1)
      self.layer1 =  sigmoid(self.A1)
      self.A2 = np.add(np.matmul(self.W2,self.layer1) , self.b2)
      self.layer2 =  sigmoid(self.A2)
      self.A3 = np.add(np.matmul(self.W3,self.layer2) , self.b3)
      self.layer3 =  sigmoid(self.A3)
      self.A4 = np.add(np.matmul(self.W4,self.layer3) , self.b4)
      self.layer4 =  sigmoid(self.A4)
      
    else:
       return sigmoid(np.add(np.matmul(self.W4,sigmoid(np.add(np.matmul(self.W3,sigmoid(np.add(np.matmul(self.W2,sigmoid(np.add(np.matmul(self.W1,t) , self.b1))) , self.b2))) , self.b3))) , self.b4))[0][0]
  def forward_pass_first_derivative(self,t,cache_nodes = False):

    if(cache_nodes):
      self.A1_deri_1 = self.W1
      self.layer1_deri_1 =  sigmoid_first_derivative(self.A1) * self.A1_deri_1
      self.A2_deri_1 = np.matmul(self.W2,self.layer1_deri_1)
      self.layer2_deri_1 =  sigmoid_first_derivative(self.A2) * self.A2_deri_1
      self.A3_deri_1 = np.matmul(self.W3,self.layer2_deri_1)
      self.layer3_deri_1 =  sigmoid_first_derivative(self.A3) * self.A3_deri_1
      self.A4_deri_1= np.matmul(self.W4,self.layer3_deri_1)
      self.layer4_deri_1 =  sigmoid_first_derivative(self.A4)[0][0] * self.A4_deri_1[0][0]
    else:
      return (sigmoid_first_derivative(self.A4) * (np.matmul(self.W4 , (sigmoid_first_derivative(self.A3) * np.matmul(self.W3 , (sigmoid_first_derivative(self.A2) * np.matmul(self.W2,( sigmoid_first_derivative(self.A1) * self.W1 ) )))))))[0][0]
  def forward_pass_second_derivative(self,t,cache_nodes = False):
    if(cache_nodes):
      A1 = np.add(np.matmul(W1,t) , b1)
      layer1 =  sigmoid(A1)
      A2 = np.add(np.matmul(W2,layer1) , b2)
      layer2 =  sigmoid(A2)
      A3 = np.add(np.matmul(W3,layer2) , b3)
      layer3 =  sigmoid(A3)
      A4 = np.add(np.matmul(W4,layer3) , b4)
      output =  sigmoid(A4)
      return A1,layer1,A2,layer2,A3,layer3,A4,output
    else:
       return sigmoid(np.add(np.matmul(W4,sigmoid(np.add(np.matmul(W3,sigmoid(np.add(np.matmul(W2,sigmoid(np.add(np.matmul(W1,t) , b1))) , b2))) , b3))) , b4))
  
  def calculate_gradients_wrt_omega(self,t):
  # reverse autoderivative
      self.forward_pass(t,cache_nodes = True)
      # self.gradient_W4 = (layer3 * sigmoid_first_derivative(A4)[0][0]).transpose
      # self.gradient_b4 =  sigmoid_first_derivative(A4)
      # gradient_l3 = ((self.W4).transpose) * sigmoid_first_derivative(A4)[0][0]
      # self.gradient_W3 = gradient_l3 * np.matmul(sigmoid_first_derivative(A3),layer2.transpose)
      # self.gradient_b3 = gradient_l3 * sigmoid_first_derivative(A3)
      # gradient_l2 = np.matmul( ( sigmoid_first_derivative(A3) * (self.W3) ).transpose , gradient_l3 )
      # self.gradient_W2 = gradient_l2 * np.matmul(sigmoid_first_derivative(A2),layer1.transpose)
      # self.gradient_b2 = gradient_l2 * sigmoid_first_derivative(A2)
      # gradient_l1 = np.matmul( ( sigmoid_first_derivative(A2) * (self.W2) ).transpose , gradient_l2 )
      # self.gradient_b1 = gradient_l1 *  sigmoid_first_derivative(A1)  
      # self.gradient_W1 = (self.gradient_b1) * t
      
      
      self.gradient_W4 = (self.layer3 * sigmoid_first_derivative(self.A4)[0][0]).transpose
      self.gradient_b4 =  sigmoid_first_derivative(self.A4)
      gradient_l3 = ((self.W4).transpose) * sigmoid_first_derivative(self.A4)[0][0]

      self.gradient_b3 = gradient_l3 * sigmoid_first_derivative(self.A3)
      self.gradient_W3 =  np.matmul(self.gradient_b3,self.layer2.transpose)
      gradient_l2 = np.matmul( ( sigmoid_first_derivative(self.A3) * (self.W3) ).transpose , gradient_l3 )

      self.gradient_b2 = gradient_l2 * sigmoid_first_derivative(self.A2)
      self.gradient_W2 = np.matmul(self.gradient_b2,self.layer1.transpose)
      gradient_l1 = np.matmul( ( sigmoid_first_derivative(self.A2) * (self.W2) ).transpose , gradient_l2 )
      
      self.gradient_b1 = gradient_l1 *  sigmoid_first_derivative(self.A1)  
      self.gradient_W1 = (self.gradient_b1) * t
      
      
  def calculate_gradients_wrt_omega1(self,t):
     self.forward_pass_first_derivative(t,cache_nodes = True)
     self.gradient_1_W4 = np.add(sigmoid_second_derivative(self.A4)[0][0] * self.A4_deri_1[0][0] * self.layer3 , (sigmoid_first_derivative(self.A4)[0][0] * self.layer3_deri_1)).transpose
     self.gradient_1_b4 = sigmoid_second_derivative(self.A4)[0][0] * self.A4_deri_1[0][0]
     self.gradient_1_l3 = sigmoid_second_derivative(self.A4)[0][0] * self.A4_deri_1[0][0] * self.W4.transpose
  # def calculate_gradients_wrt_omega2(self,t):
     
  # def adam_optimizatoin():
     
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