import numpy as np
def init_params():
  W1 = np.random.rand(100 , 1) - 0.5
  gradient_W1 = np.zeros((100 , 1))
  W2 = np.random.rand(100, 100) - 0.5
  gradient_W2 = np.zeros((100 , 100))
  W3 = np.random.rand(100, 100) -0.5
  gradient_W3 = np.zeros((100 , 100))
  W4 = np.random.rand(1, 100) - 0.5
  gradient_W4 = np.zeros((1 , 100))
  b1 = np.random.rand(100, 1) - 0.5
  gradient_b1 = np.zeros((100 , 1))
  b2 = np.random.rand(100, 1) - 0.5
  gradient_b2 = np.zeros((100 , 1))
  b3 = np.random.rand(100, 1) -0.5
  gradient_b3 = np.zeros((100 , 1))
  b4 = np.random.rand(1,1) - 0.5
  gradient_b4 = np.zeros((1, 1))
  return W1, W2, W3, W4, b1, b2, b3, b4 ,gradient_W1 ,gradient_W2 ,gradient_W3 ,gradient_W4,gradient_b1,gradient_b2,gradient_b3,gradient_b4

def reLU(val):
  return np.maximum(0, val)


def sigmoid(val_array):
  return 1 / (1 + np.exp(-val_array))
def sigmoid_first_derivative(val_array):
    sigmoid_val = 1 / (1 + np.exp(-val_array))
    return sigmoid_val * (1 - sigmoid_val)
def sigmoid_second_derivative(val_array):
    sigmoid_val = 1 / (1 + np.exp(-val_array))
    return sigmoid_val * (1 - sigmoid_val) * (1 - 2 * sigmoid_val)

def simpsons_3_8_rule_integration(func, a = 0, b = 0, n=9999):
    if n % 3 != 0:
        raise ValueError("The number of subintervals (n) must be a multiple of 3.")
    
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

