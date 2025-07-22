import numpy as np
from image_loading import read_pgm, read_all_faces
from faces_places import face_space, mean_face
def project_image(M, k, mean_face, eigenfaces):
    """
    Projects a grayscale image M onto the first k eigenfaces.
    Args:
        M (np.ndarray): Grayscale image array (flattened to shape (n,))
        k (int): Number of eigenfaces to project onto (k < n)
        mean_face (np.ndarray): Mean face array (shape (n,))
        eigenfaces (np.ndarray): Matrix of eigenfaces (shape (n, m))
    Returns:
        np.ndarray: Projection coefficients (shape (k,))
    """
    # Center the image by subtracting the mean face
    print(M.shape)
    M_centered = M.flatten() - mean_face
    # Select the first k eigenfaces
    Phi = eigenfaces[:, :k]
    # Project the centered image onto the eigenfaces
    projection = Phi.T @ M_centered
    return projection

x = read_pgm('face_data/s1/1.pgm')
k = 10  # Number of eigenfaces to use
A = read_all_faces()
mf = mean_face(A)
F = face_space(A)[1]

projection = project_image(x, k, mf, F)

import matplotlib.pyplot as plt

def show_projection(projection, mean_face, eigenfaces, image_shape=(92, 92)):
    """
    Reconstructs and displays the image from its projection coefficients.
    """
    k = projection.shape[0]
    Phi = eigenfaces[:, :k]
    reconstructed = mean_face + Phi @ projection
    img = reconstructed.reshape(image_shape)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# After computing projection
show_projection(projection, mf, F, image_shape=(92, 92))