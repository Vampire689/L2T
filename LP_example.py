#导入包
from scipy import optimize
import numpy as np

# #确定c,A,b,Aeq,beq
# c = np.array([-1, -2])
# A = np.array([[-0.5, 1], [4, 5], [7, 3]])
# b = np.array([11/5, 26, 32])
# Aeq = None
# beq = None
# y1_bounds = (0, 2)
# y2_bounds = (0, None)

# #求解
# res = optimize.linprog(c,A,b,Aeq,beq, bounds=[y1_bounds, y2_bounds])
# print(res)

# x1, x2 bounds
bounds = [(-1, 2), (-1, 1), (None, None), (None, None), (0, 0), (0, 1), (None, None), (None, None)]
# 变量定义
#             x1, x2, a^, b^, Za, Zb, a , b
# 目标函数
f = np.array([0., 0., 0., 0., 0., 0., 1, -1])
# 等式
k1, k2, ua, la = 2, 1, 5, -3
k3, k4, ub, lb = 1, 1, 3, -2
C = np.array([
             [k1, k2, -1, 0., 0., 0., 0., 0],       # a^ = k1*x1 + k2*x2
             [k3, k4, 0., -1, 0., 0., 0,  0],       # b^ = k3*x1 + k4*x2
])
D = np.array([0, 0])
# 不等式
Za, Zb = 1, 1
A = np.array([
             [0., 0., 0., 0., 0., 0.,-1,  0],       # a >= 0 --> -a <= 0
             [0., 0., 1., 0., 0., 0.,-1,  0],       # a >= a^ --> a^ - a <= 0
             [0., 0., 0., 0., 0., 0., 0, -1],       # b >= 0 --> -b <= 0
             [0., 0., 0., 1., 0., 0., 0, -1],       # b >= b^ --> b^ - b <= 0
            #  [0., 0., -ua/(ua-la), 0., 0., 0., 1, 0.],       # a<= ua(a^ - la)/(ua - la) --> 
            #  [0., 0., 0., -ub/(ub-lb), 0., 0., 0, 1.],       # b<= ub(b^ - lb)/(ub - lb) --> 
             [0., 0., 0., 0, -ua, 0., 1,  0],       # a<= ua*Za --> a - ua*Za <= 0
             [0., 0., 0., 0,  0, -ub, 0,  1],       # b<= ub*Zb --> b - ub*Zb <= 0
             [0., 0., -1, 0, -la, 0,  1,  0],       # a - ^a - la*Za <= -la
             [0., 0., 0, -1, 0,  -lb, 0,  1],       # b - ^b - lb*Zb <= -lb
])       
B = np.array([0, 0, 0, 0, 
            #  -ua*la/(ua-la), -ub*lb/(ub-lb),
             0, 0, -la, -lb
])


res = optimize.linprog(f, A, B, C, D, bounds=bounds)
print(res)