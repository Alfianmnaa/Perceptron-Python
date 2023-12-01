# Logic Gate XNOR
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
# wNOT = -1, bNOT = 0.5
def NOT_logicFunction(x):
 wNOT = -1
 bNOT = 0.5
 return perceptronModel(x, wNOT, bNOT)

# AND Logic Function
# w1 = 1, w2 = 1, bAND = -1.5
def AND_logicFunction(x):
 w = np.array([1, 1])
 bAND = -1.5
 return perceptronModel(x, w, bAND)

# OR Logic Function
# here w1 = wOR1 = 1,
# w2 = wOR2 = 1, bOR = -0.5
def OR_logicFunction(x):
 w = np.array([1, 1])
 bOR = -0.5
 return perceptronModel(x, w, bOR)

# XNOR Logic Function
# with AND, OR and NOT
# function calls in sequence
def XNOR_logicFunction(x):
 y1 = OR_logicFunction(x)
 y2 = AND_logicFunction(x)
 y3 = NOT_logicFunction(y1)
 final_x = np.array([y2, y3])
 finalOutput = OR_logicFunction(final_x)
 return finalOutput

# testing the Perceptron Model
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])

print("XNOR({}, {}) = {}".format(0, 1, XNOR_logicFunction(test1)))
print("XNOR({}, {}) = {}".format(1, 1, XNOR_logicFunction(test2)))
print("XNOR({}, {}) = {}".format(0, 0, XNOR_logicFunction(test3)))
print("XNOR({}, {}) = {}".format(1, 0, XNOR_logicFunction(test4)))