import numpy as np
import os
import matplotlib.pyplot as plt
import time

data = []
for subject in os.listdir('face_data'):
    first_face = os.path.join('face_data', subject, '1.pgm')
    second_face = os.path.join('face_data', subject, '2.pgm')
    third_face = os.path.join('face_data', subject, '3.pgm')
    faces = [first_face, second_face, third_face]
    for face in faces:
        with open(face, 'rb') as f:
            f.readline()         # magic number
            f.readline()         # width height
            f.readline()         # maxval
            img = np.frombuffer(f.read(), dtype=np.uint8).reshape((112, 92))
            data.append(img)


image_matrix = np.zeros((len(data), 92**2))
for i, img in enumerate(data):
    img = img[10: img.shape[0]-10, :]
    image_matrix[i, :] = img.flatten()

# Subtract the mean face (as we all should in life)
mean_face = np.zeros_like(image_matrix[0])
for face in image_matrix:
    mean_face += face
mean_face /= len(data)
A = image_matrix - mean_face # Broadcast across columns, Shape (m, n)


def face_space(A):
    """
    Returns the eigenvectors of C and the eigenfaces as a tuple (eigvecs_C, eigenfaces).
    Args:
        A (np.ndarray): Mean-centered image matrix (shape: m x n)
    Returns:
        tuple: (eigvecs_C, eigenfaces)
            eigvecs_C: Eigenvectors of C (shape: m x m)
            eigenfaces: Eigenfaces matrix (shape: n x m)
    """
    C = A @ A.T # Eigen-decomposition of C

    if A.shape[0] > A.shape[1]:
        raise ValueError("Matrix A has more rows (m) than columns (n). Please ensure m <= n.")
    eigvals_C, eigvecs_C = np.linalg.eig(C)
    eigvecs_C = np.real(eigvecs_C)
    # Sort by descending eigenvalue
    idx = np.argsort(np.real(eigvals_C))[::-1]
    eigvecs_C = eigvecs_C[:, idx]
    # Compute eigenfaces
    eigenfaces = A.T @ eigvecs_C
    eigenfaces /= np.linalg.norm(eigenfaces, axis=0, keepdims=True)
    return eigvecs_C, eigenfaces


def mean_face(image_matrix):
    """
    Computes the mean face by averaging all rows in the image matrix.
    Args:
        image_matrix (np.ndarray): Matrix of flattened images (shape: m x n)
    Returns:
        np.ndarray: Mean face (shape: n,)
    """
    return np.mean(image_matrix, axis=0)

