# import numpy as np
# def simpsons_3_8_rule_integration(func, a, b, n):
#     if n % 3 != 0:
#         raise ValueError("The number of subintervals (n) must be a multiple of 3.")
    
#     h = (b - a) / n
#     integral_sum = func(a) + func(b)  
    
#     for i in range(1, n):
#         x_i = a + i * h
#         if i % 3 == 0:
#             integral_sum += 2 * func(x_i)  
#         else:
#             integral_sum += 3 * func(x_i) 
    
#     integral_approximation = (3 * h / 8) * integral_sum
#     return integral_approximation

# def my_function(x,c = True):
#     if (c) :
#         return np.exp((3*x) + 1)
#     else:
#         return x**2

# a = 0
# b = 2
# n = 99
# print(simpsons_3_8_rule_integration(my_function, a, b, n))
# def external_function(callback_function):
#     print("External function")
#     callback_function()
# class MyClass:
#     def my_method(self):
#         print("Inside my_method")
#     def my_other_method(self):
#         external_function(self.my_method)


# Create an instance of MyClass
# obj = MyClass()

# Pass the class method as an argument to the external function
# external_function(obj.my_method)
# def Integral_function(self,t):  
#       w_0 = self.forward_pass(t)
#       w_1 = self.forward_pass_first_derivative(t)
#       w_2 = self.forward_pass_second_derivative(t)
#       return (abs( ( ( -3*(w_1**2) )/( 4*(w_0**2) ) ) + (w_2 / (2*w_0 )) ) /   w_0 ) 
# A = np.array([[1], [2], [3], [4]])  # A is of shape (4, 1)
# B = np.array([[11, 12,  13,  14],
#               [20,  30,  40,  50],
#               [10,  20,  100,  200],
#               [ 10,  20,  100,  200]])  # B is of shape (4, 4)
# result  = A*B


import progressbar
from time import sleep
bar = progressbar.ProgressBar(maxval=1000, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for i in range(1000):
    bar.update(i+1)
    sleep(0.1)
bar.finish()
