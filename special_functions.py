import numpy as np
import math
from scipy import special

def spher_harm(N , theta , phi):


    ''' 
    
    This function calculates all the spherical harmonics we need
    
    Inputs:
        N = maximum order 
        theta = elevation angle in radians
        phi = azimuth angle in radians
        
    Outputs:
        Ynm_pos_re = real part of the Ynm if m >= 0
        Ynm_pos_im = imaginary part of the Ynm if m >= 0
        Ynm_neg_re = real part of the Ynm if m < 0
        Ynm_neg_im = imaginary part of the Ynm if m < 0
        
    NOTE: all outputs will be an array of size (N+1) x (N+1) and has the form as followed
    
    Ynm = [ Y_00 Y_10 Y_20 Y_30  ... ...         Ynm = [ 0      0       0       0     ... ...
             0   Y_11 Y_21 Y_31  ... ...                 0   Y_1(-1) Y_2(-1) Y_3(-1)  ... ...
             0     0  Y_22 Y_32  ... ...     or          0      0    Y_2(-2) Y_3(-2)  ... ...
             0     0    0  Y_33  ... ...                 0      0       0    Y_3(-3)  ... ...
             .     .    .    .   ... ...                 .      .       .       .     ... ...
             .     .    .    .   ... ...]                .      .       .       .     ... ...] 
    '''

    all_Pnm = special.lpmn(N, N, math.cos(theta)*1.0)[0]

    Ynm_pos_re = np.zeros((N+1 , N+1))
    Ynm_pos_im = np.zeros((N+1 , N+1))
    Ynm_neg_re = np.zeros((N+1 , N+1))
    Ynm_neg_im = np.zeros((N+1 , N+1))

    for n in range(N+1):
        for m in range(-n , n+1):

            Anm = np.sqrt((2 * n + 1) * (math.factorial(n - abs(m))) / (4 * math.pi * (math.factorial(n + abs(m))))) + 0j
            Pnm = all_Pnm[abs(m), n]
            exp = (math.e) ** (1j * abs(m) * phi)
            final = Anm * Pnm * exp

            if(m >= 0):

                Ynm_pos_re[abs(m) , n] = final.real
                Ynm_pos_im[abs(m) , n] = final.imag

            else:

                Ynm_neg_re[abs(m) , n] = final.real * ((-1)**abs(m))
                Ynm_neg_im[abs(m) , n] = -final.imag * ((-1)**abs(m))

    return Ynm_pos_re , Ynm_pos_im, Ynm_neg_re , Ynm_neg_im

def calc_Xn(N , kx):
    '''
    This function calculates the Xn = i^n * j_n(kx) for all n at kx
    
    Inputs:
        N = maximum order
        kx = product of the wavenumber and the distance of a specific point
    
    Output:
        Xn_re = array of the real part of Xn that corresponds to all n at the kx specified 
        Xn_im = array of the imaginary part of Xn that corresponds to all n at the kx specified 

    '''

    Xn_re = np.zeros((N+1,))
    Xn_im = np.zeros((N+1,))

    for n in range(N+1):
        Xn = ((1j)**n)*special.spherical_jn(n , kx , derivative=False)
        Xn_re[n] = Xn.real
        Xn_im[n] = Xn.imag

    return Xn_re , Xn_im


def XYZ_2_spher(x , y , z):
    '''
    This function converts an array of points in the cartesian coordinate system to the spherical coordinate system
    
    Inputs:
        x = A mesh grid of x coordinate of the points 
        y = A mesh grid of y coordinate of the points 
        z = A mesh grid of z coordinate of the points 
    
    Outputs:
        r = Distance to the origin  
        theta = Elevation angle
        phi = Azimuth angle
    '''
    r = np.zeros(x.shape)
    theta = np.zeros(x.shape)
    phi = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):

            r[i,j]=math.sqrt(x[i,j]**2 + y[i,j]**2 + z[i,j]**2)
            theta[i,j] = math.acos(z[i,j]/r[i,j])
            phi[i,j]=math.atan2(y[i,j],x[i,j])

    return r , theta , phi
