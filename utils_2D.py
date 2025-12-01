import numpy as np
from tensorflow.keras import backend as K
from numba import jit

def rmsre(y_true, y_pred, eps=1e-8):
    return K.sqrt( K.mean( K.square( (y_true - y_pred) / (y_true + eps) ), axis=-1) )

###########################################################
# D2Q9 stencil
Q = 9
D = 2

c = np.array([
    [ 0,  0],
    [ 1,  0],
    [ 0,  1],
    [-1,  0],
    [ 0, -1],
    [ 1,  1],
    [-1,  1],
    [-1, -1],
    [ 1, -1],
], dtype=np.int32)

w = np.array([
    4./9.,
    1./9., 1./9., 1./9., 1./9.,
    1./36., 1./36., 1./36., 1./36.,
], dtype=np.float64)

cs2 = 1./3.

K0 = 1./Q
K1 = 1./6.

###########################################################
# Function for the calculation of the equilibrium
@jit(nopython=True)
def compute_feq(feq, rho, ux, uy):

    uu = (ux**2 + uy**2) / cs2

    for ip in range(Q):
        cu = (c[ip, 0] * ux[:, :] + c[ip, 1] * uy[:, :]) / cs2
        feq[:, :, ip] = w[ip] * rho * (1.0 + cu + 0.5 * (cu * cu - uu))

    return feq

###########################################################

def LB_stencil():
    return c, w, cs2, compute_feq
