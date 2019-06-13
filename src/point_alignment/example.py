import numpy as np
from SimpleRigid import Rigid


X = np.random.randint(0, 100, size=(30, 4))
Y = np.random.randint(100, 200, size=(50, 4))

rigid = Rigid(X[1:], Y[1:])
rigid.register(callback=None)

Rotation_M = rigid.R
Translation_V = rigid.t
Transformed_Y = rigid.TY

print('R = ', Rotation_M)
print('t = ', Translation_V)
# print('TY = ', Transformed_Y)
