import numpy as np
from image_loading import read_pgm, read_all_faces
from faces_places import *
import matplotlib.pyplot as plt


# Read the image into a numpy array
#x = read_pgm('face_data', 1, 1)
x = read_pgm('changed')

k = 40  # Number of eigenfaces to use
A = read_all_faces()
average = mean_face(A)
F = face_space(A)

def grahm_schmidt(A):
    """Perform Gram-Schmidt orthonormalization on the columns of A."""
    Q, R = np.linalg.qr(A)
    return Q

proj = Projection(average, grahm_schmidt(F), original=x, image_shape=(92, 92))
proj.display(x, k)
