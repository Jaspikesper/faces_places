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


C = A @ A.T   # Shape: (m, m)
# Eigen-decomposition of C
eigvals_C, eigvecs_C = np.linalg.eig(C)
eigvals_C = np.real(eigvals_C)
eigvecs_C = np.real(eigvecs_C)

# Sort eigenvalues and vectors in descending order
idx = np.argsort(eigvals_C)[::-1]
eigvals_C = eigvals_C[idx]
eigvecs_C = eigvecs_C[:, idx]

# Following formula (5) from Turk & Pentland, 1991
eigenfaces = A.T @ eigvecs_C    # Shape: (n, m)
eigenfaces /= np.linalg.norm(eigenfaces, axis=0, keepdims=True)

# Plot the first eigenface for visualization
first_eigenface = eigenfaces[:, 0].reshape((92, 92))
plt.imshow(first_eigenface, cmap='gray')
plt.show()
