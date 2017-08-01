import numpy as np
import math
import special_functions as sf



# Variables and constants
c = 340 # speed of sound in m/s  (assuming it's constant)
f = 500 # frequency of the narrow-band plane-wave in Hz
k = 2*math.pi*f/c # wave number
x0 = 5 # radius of our reproduction sphere in meters
Theta = math.pi/2 # Elevation angle of the incoming plane wave
Phi = math.pi/6 # Azimuth angle of the incoming plane wave

N = math.ceil(k*x0) # Assuming an Nth order reproduction (Using the rule of thumb of 4% error)
print(N)

# For the spherical harmonics
Ynm_pos_re , Ynm_pos_im, Ynm_neg_re , Ynm_neg_im = sf.spher_harm(N, Theta , Phi)
Ynm_pos = Ynm_pos_re + 1j*Ynm_pos_im
Ynm_neg = Ynm_neg_re + 1j*Ynm_neg_im

# Set up variables for the sound field plot
z = x0 / 2 # The height of the plane of interest

num_increm = 50 # Number of increments for our variables

X = np.linspace(-10 , 10 , num_increm)
Y = np.linspace(-10 , 10 , num_increm)

X_mesh , Y_mesh = np.meshgrid(X , Y)

# x = np.linspace(z , x0 , num_increm)
# theta = np.arccos(z/x)
# phi = np.linspace(0 , 2*math.pi , num_increm)

# Matrix form for the desired sound field
first = np.ones((1,N+1))

# x_mesh , phi_mesh = np.meshgrid(x , phi)
des_sound_field = np.zeros(X_mesh.shape) + 0j

print(x_mesh.shape)

for i in range(x_mesh.shape[0]):
    for j in range(x_mesh.shape[1]):
        Ynm_pos_re, Ynm_pos_im, Ynm_neg_re, Ynm_neg_im = sf.spher_harm(N, np.arccos(z/np.sqrt(X_mesh[i, j]**2 + Y_mesh[i,j]**2 + z**2)), np.arctan2(Y_mesh[i,j],X_mesh[i, j]))
        Ynm_x_pos = Ynm_pos_re + 1j * Ynm_pos_im
        Ynm_x_neg = Ynm_neg_re + 1j * Ynm_neg_im
        middle_pos = np.multiply(np.conj(Ynm_pos), Ynm_x_pos)
        middle_neg = np.multiply(np.conj(Ynm_neg), Ynm_x_neg)

        Xn_re, Xn_im = sf.calc_Xn(N, k * x_mesh[i,j])
        Xn = Xn_re + 1j * Xn_im
        des_sound_field[i,j] = np.dot(np.dot(first, middle_pos), Xn) + np.dot(np.dot(first, middle_neg), Xn)


print(des_sound_field)


plt.figure()
plt.imshow(z+10, extent=(np.amin(eta1), np.amax(eta1), np.amin(eta2), np.amax(eta2)) , clim=(1, 1000) , cmap=cm.hot, norm=LogNorm())
plt.xlabel("$\eta_1$")
plt.ylabel("$\eta_2$")
plt.title("|| y($\eta_1$ , $\eta_2$) - $y$|$|^2$")
plt.colorbar()
plt.show()






