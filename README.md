# Reproduction of a Plane Wave

The implementation is based on the paper titled "Reproduction of a Plane Wave Sound Field Using an Array of Loudspeakers" by Ward, D. et al.

In the main.py file, the desired soundfield and the approximated soundfield are both calculated using spherical harmonics as bases. The detailed computation and assumptions can be viewed in the specialfunc.py file.

The main idea here is to compute the corresponding base coefficients for the loudspeakers such that the resulting soundfield induced by the loudspeakers are approximately the same as a propagating plane wave.
