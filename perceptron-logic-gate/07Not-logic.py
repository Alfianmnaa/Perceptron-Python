# Logic Gate NOT
# importing Python library
import numpy as np

# define Unit Step Function
def unitStep(v):
 if v >= 0:
  return 1
 else:
  return 0

# design Perceptron Model
def perceptronModel(x, w, b):
 v = np.dot(w, x) + b
 y = unitStep(v)
 return y

# NOT Logic Function
# w = -1, b = 0.5
def NOT_logicFunction(x):
 w = -1
 b = 0.5
 return perceptronModel(x, w, b)

# testing the Perceptron Model
test1 = np.array(1)
test2 = np.array(0)

print("NOT({}) = {}".format(1, NOT_logicFunction(test1)))
print("NOT({}) = {}".format(0, NOT_logicFunction(test2)))