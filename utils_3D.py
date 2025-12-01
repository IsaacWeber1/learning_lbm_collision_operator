import numpy as np
from tensorflow.keras import backend as K
from numba import jit

def rmsre(y_true, y_pred, eps=1e-8):
    return K.sqrt( K.mean( K.square( (y_true - y_pred) / (y_true + eps) ), axis=-1) )

###########################################################
# D3Q19 stencil
Q = 19
D = 3

c = np.array([
    # rest
    [ 0,  0,  0],
    # face (paired: + then -)
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
    # edge xy-plane (++, +-, -+, --)
    [ 1,  1,  0],
    [ 1, -1,  0],
    [-1,  1,  0],
    [-1, -1,  0],
    # edge yz-plane (++, +-, -+, --)
    [ 0,  1,  1],
    [ 0,  1, -1],
    [ 0, -1,  1],
    [ 0, -1, -1],
    # edge xz-plane (++, +-, -+, --)
    [ 1,  0,  1],
    [ 1,  0, -1],
    [-1,  0,  1],
    [-1,  0, -1],
], dtype=np.int32)

w = np.array([
    1./3.,                                      # rest
    1./18., 1./18., 1./18., 1./18., 1./18., 1./18.,  # face
    1./36., 1./36., 1./36., 1./36.,             # edge xy
    1./36., 1./36., 1./36., 1./36.,             # edge yz
    1./36., 1./36., 1./36., 1./36.,             # edge xz
], dtype=np.float64)

cs2 = 1./3.

K0 = 1./Q
K1 = 1./10.

###########################################################
# Function for the calculation of the equilibrium
@jit(nopython=True)
def compute_feq(feq, rho, ux, uy, uz):

    uu = (ux**2 + uy**2 + uz**2) / cs2

    for ip in range(Q):
        cu = (c[ip, 0] * ux[:, :, :] + c[ip, 1] * uy[:, :, :] + c[ip, 2] * uz[:, :, :]) / cs2
        feq[:, :, :, ip] = w[ip] * rho * (1.0 + cu + 0.5 * (cu * cu - uu))

    return feq

###########################################################

def LB_stencil():
    return c, w, cs2, compute_feq
