import numpy as np
import math
import special_functions as sf
import matplotlib.pyplot as plt


# Variables and constants
c = 340 # speed of sound in m/s  (assuming it's constant)
f = 500 # frequency of the narrow-band plane-wave in Hz
k = 2*math.pi*f/c # wave number
x0 = 2 # radius of our reproduction sphere in meters
Theta = math.pi/2 # Elevation angle of the incoming plane wave
Phi = math.pi/4 # Azimuth angle of the incoming plane wave

N = math.ceil(k*x0) # Assuming an Nth order reproduction (Using the rule of thumb of 4% error)
print(N)

# For the spherical harmonics
Ynm_pos_re , Ynm_pos_im, Ynm_neg_re , Ynm_neg_im = sf.spher_harm(N, Theta , Phi)
Ynm_pos = Ynm_pos_re + 1j*Ynm_pos_im
Ynm_neg = Ynm_neg_re + 1j*Ynm_neg_im

# Set up variables for the sound field plot
z = x0 / 2 # The height of the plane of interest
x_range = 2 # Half of the width of our x axis on the plot
y_range = 2 # Half of the width of our y axis on the plot

num_increm = 50 # Number of increments for our variables

X = np.linspace(-x_range , x_range , num_increm)
Y = np.linspace(y_range , -y_range , num_increm)

X_mesh , Y_mesh = np.meshgrid(X , Y)
Z_mesh = z * np.ones((num_increm,num_increm))

# Converting from XYZ to the spherical coordinate

r , theta , phi = sf.XYZ_2_spher(X_mesh , Y_mesh , Z_mesh)

# Matrix form for the desired sound field
first = np.ones((1,N+1))

# x_mesh , phi_mesh = np.meshgrid(x , phi)
des_sound_field = np.zeros(X_mesh.shape) + 0j

print(x_mesh.shape)

for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        Ynm_pos_re, Ynm_pos_im, Ynm_neg_re, Ynm_neg_im = sf.spher_harm(N, theta[i,j] , phi[i,j])
        Ynm_x_pos = Ynm_pos_re + 1j * Ynm_pos_im
        Ynm_x_neg = Ynm_neg_re + 1j * Ynm_neg_im
        middle_pos = np.multiply(np.conj(Ynm_pos), Ynm_x_pos)
        middle_neg = np.multiply(np.conj(Ynm_neg), Ynm_x_neg)

        Xn_re, Xn_im = sf.calc_Xn(N, k * r[i,j])
        Xn = Xn_re + 1j * Xn_im
        des_sound_field[i,j] = np.dot(np.dot(first, middle_pos), Xn) + np.dot(np.dot(first, middle_neg), Xn)

        
# Plotting out the desired sound field
plt.figure()
plt.imshow(des_sound_field.real, extent=(X[0], X[len(X)-1], Y[len(Y)-1], Y[0]) )
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Sound Field Simulation")
plt.colorbar()
plt.show()






