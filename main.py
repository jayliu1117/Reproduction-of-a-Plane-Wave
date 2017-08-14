import numpy as np
import math
import special_functions as sf
import matplotlib.pyplot as plt
from scipy import special

# Flag for different operations: 1 = calculate/show only the desired sound field
#                                2 = calculate/show only the speakers sound field
#                                3 = calculate/show for both the desired sound field and the speakers sound field                               
flag = 3

# Variables and constants
c = 340 # speed of sound in m/s  (assuming it's constant)
f = 300 # frequency of the narrow-band plane-wave in Hz
k = 2*math.pi*f/c # wave number
x0 = 0.5 # radius of our reproduction sphere in meters
Theta = math.pi/2 # Elevation angle of the incoming plane wave
Phi = math.pi/3 # Azimuth angle of the incoming plane wave

# Set up variables for the sound field plot
z = x0 / 2  # The height of the plane of interest
x_range = 2  # Half of the width of our x axis on the plot
y_range = 2  # Half of the width of our y axis on the plot

num_increm = 50  # Number of increments for our variables

X = np.linspace(-x_range, x_range, num_increm)
Y = np.linspace(y_range, -y_range, num_increm)

X_mesh, Y_mesh = np.meshgrid(X, Y)
Z_mesh = z * np.ones((num_increm, num_increm))

# Converting from XYZ to the spherical coordinate

r, theta, phi = sf.XYZ_2_spher(X_mesh, Y_mesh, Z_mesh)

# For the drawing of our reproduction sphere

r_array = x0*np.ones((num_increm,1))
theta_array = math.acos(z/x0)*np.ones((num_increm,1))
phi_array = np.linspace(0,2*math.pi,num_increm).reshape(num_increm,1)

circle_x, circle_y, _ = sf.Spher_2_XYZ(r_array, theta_array, phi_array)


def desired_sound_field():
    N = 15
    print('The order of our actual desired sound field is  %i' % (N))
    # N = math.ceil(k*x0) # Assuming an Nth order reproduction (Using the rule of thumb of 4% error)
    # print(N)

    # For the spherical harmonics
    Ynm_pos_re , Ynm_pos_im, Ynm_neg_re , Ynm_neg_im = sf.spher_harm(N, Theta , Phi)
    Ynm_pos = Ynm_pos_re + 1j*Ynm_pos_im
    Ynm_neg = Ynm_neg_re + 1j*Ynm_neg_im

    # Matrix form for the desired sound field
    first = np.ones((1,N+1))

    # x_mesh , phi_mesh = np.meshgrid(x , phi)
    des_sound_field = np.zeros(X_mesh.shape) + 0j

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
    plt.plot(circle_x,circle_y,'k')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Sound Field Simulation - Desired Sound Field")
    plt.colorbar()
    if flag == 3:
        return 0
    else:
        plt.show()

    return 0


def speakers_sound_field():
    # For calculating the sound field produce by L speakers in a spherical array
    # Constants and variables
    rad = x0 # radius of the spherical array
    kr = k*rad
    N_speaker = int(math.ceil(k*x0)) # order of the speaker reproduction system
    print('The order of the speaker is  %i' % (N_speaker))
    L = 4 # number of speakers
    theta_speaker = np.ones((L,)) * (math.pi/2)  # Elevation angles for the speakers
    phi_speaker = np.arange(L) * (2*math.pi/L)   # Azimuth angels for the speakers

    pnm = np.empty((1,L),dtype=object)
    em = np.zeros((2*N_speaker+1,L)) + 0j
    Rn = sf.calc_Rn(N_speaker,kr)
    P = np.zeros(((N_speaker+1)**2,L)) + 0j
    b = np.zeros(((N_speaker+1)**2,1)) + 0j

    # Construct all the em into one matrix first
    for m in range(-N_speaker,N_speaker+1):
        for j in range(L):
            em[m+N_speaker,j] = math.e**(-1j*m*phi_speaker[j])

    # Construct all the pnm (with different speaker_theta) into one object array
    for l in range(L):
        pnm[0,l] = special.lpmn(N_speaker, N_speaker, math.cos(theta_speaker[l]) * 1.0)[0]

    # Construct the pnm for the incoming plane wave
    pnm_wave = special.lpmn(N_speaker, N_speaker, math.cos(Theta) * 1.0)[0]

    # Contruct the P matrix and the b matrix
    p = np.zeros((L,)) + 1j  # Initialize the pnm used in the matrix calculation
    count = 0
    for n in range(N_speaker+1):
        for m in range(-n,n+1):


            for l in range(L):
                p[l] = pnm[0, l][abs(m), n]

            e = em[N_speaker+m,0:L]
            P[count , 0:L] = Rn[n,n] * np.multiply(p, e)
            b[count, 0] = pnm_wave[abs(m), n] * (math.e ** (-1j * m * Phi))
            count += 1


    # Now solve for the weights a
    if(L < (N_speaker+1)**2):
        a = sf.truncate_solve(P, b)
    elif(L > (N_speaker+1)**2):
        a = sf.min_a_solve(P, b)

    print(a)

    # Now construct the sound field reproduced by our loudspeaker array
    # Write out the sound field equation in the matrix form Ysum x Rn x Xn
    Ysum = np.zeros((1,N_speaker)) + 0j


    # For the spherical harmonics of the loud speakers
    Ynm_speak_pos = np.empty((L,),dtype=object)
    Ynm_speak_neg = np.empty((L,),dtype=object)
    for l in range(L):
        Ynm_pos_re, Ynm_pos_im, Ynm_neg_re, Ynm_neg_im = sf.spher_harm(N_speaker, theta_speaker[l], phi_speaker[l])
        Ynm_speak_pos[l] = Ynm_pos_re + 1j * Ynm_pos_im
        Ynm_speak_neg[l] = Ynm_neg_re + 1j * Ynm_neg_im

    # For the product sum of the two spherical harmonics (Ynm^*)Ynm
    multi_sum_pos = np.empty((L,), dtype=object)
    multi_sum_neg = np.empty((L,), dtype=object)

    # Initialize the sound field
    actual_sound_field = np.zeros(X_mesh.shape) + 0j

    # Now calculate for the overall sound field
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            Ynm_pos_re, Ynm_pos_im, Ynm_neg_re, Ynm_neg_im = sf.spher_harm(N_speaker, theta[i, j], phi[i, j])
            Ynm_x_pos = Ynm_pos_re + 1j * Ynm_pos_im
            Ynm_x_neg = Ynm_neg_re + 1j * Ynm_neg_im

            for l in range(L):
                multi_pos = np.multiply(np.conj(Ynm_speak_pos[l]), Ynm_x_pos)
                multi_neg = np.multiply(np.conj(Ynm_speak_neg[l]), Ynm_x_neg)
                multi_sum_pos[l] = sf.column_sum(multi_pos)
                multi_sum_neg[l] = sf.column_sum(multi_neg)

                Ysum = a[l]*multi_sum_neg[l] + a[l]*multi_sum_pos[l]

            Xn_re, Xn_im = sf.calc_Xn(N_speaker, k * r[i, j])
            Xn = Xn_re + 1j * Xn_im

            actual_sound_field[i,j] = np.dot(np.dot(Ysum,Rn) , Xn)

    # Plotting out the actual sound field
    plt.figure()
    plt.imshow(actual_sound_field.real, extent=(X[0], X[len(X) - 1], Y[len(Y) - 1], Y[0]))
    plt.plot(circle_x,circle_y,'k')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Sound Field Simulation - Actual Sound Field using %i speakers" % L)
    plt.colorbar()
    plt.show()



    # print(P)

    return 0

if(flag == 1):
    desired_sound_field()
elif(flag == 2):
    speakers_sound_field()
elif(flag == 3):
    desired_sound_field()
    speakers_sound_field()
