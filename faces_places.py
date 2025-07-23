import numpy as np
import os
import matplotlib.pyplot as plt
import time
import numpy.linalg as la


def mean_face(image_matrix):
    """
    Computes the mean face by averaging all rows in the image matrix.
    Args:
        image_matrix (np.ndarray): Matrix of flattened images (shape: m x n)
    Returns:
        np.ndarray: Mean face (shape: n,)
    """
    return np.mean(image_matrix, axis=0)

def gram_schmidt(A):
    """Compute an orthogonal matrix using the Gram-Schmidt process on the rows of A"""

    Q = np.zeros_like(A.shape)
    Q[0] = A[0]
    for i in range(1, A.shape[0]):
        u = A[i]
        for v in Q:
            projuv = v * np.dot(u, v) / np.dot(v, v)
            print('u is: ', u)
            print('v is: ', v)
            print('projuv is: ', projuv)
            u -= projuv
        Q[i] = u
    return Q

def face_space(A):
    """
    Returns eigenfaces from the data matrix A
    O(M^2) due to eigenvalue decomposition of the covariance matrix C = A @ A.T / m and grahm-schmidt orthonormalization.
    Args:
        A (np.ndarray): Mean-centered image matrix (shape: m x n)
    Returns:
       orthonormalized np array (eigenfaces) Eigenvectors of C (shape: m x m)
    """
    C = A @ A.T / A.shape[0]

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
    return eigenfaces


class Projection:
    """An object for reconstructing images from the eigenfaces
    Requires the mean face and eigenfaces to be computed beforehand, plus optionally the original image to reconstruct."""
    def __init__(self, mean_face, eigenfaces, original=None, image_shape=(92, 92)):
        self.mean_face = mean_face
        self.eigenfaces = eigenfaces
        self.original = original
        self.image_shape = image_shape

    def project_onto_me(self, x, k):
        """ Projects the image x onto the eigenfaces and reconstructs it using k eigenfaces."""
        if k > self.eigenfaces.shape[1]:
            raise ValueError(f"k={k} exceeds the number of eigenfaces {self.eigenfaces.shape[1]}.")
        if k == self.eigenfaces.shape[1]:
            print('Warning: Using all eigenfaces for reconstruction, which may not be optimal.')

        Phi = self.eigenfaces[:, :k]
        x_centered = x.flatten() - self.mean_face
        projection = Phi.T @ x_centered
        reconstructed = self.mean_face + Phi @ projection
        return reconstructed.reshape(self.image_shape)

    def display(self, x, k):
        import matplotlib.pyplot as plt
        img_reconstructed = self.project_onto_me(x, k)
        if self.original is not None:
            img_original = self.original.reshape(self.image_shape)
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img_original, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(img_reconstructed, cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')
        else:
            plt.figure(figsize=(4, 4))
            plt.imshow(img_reconstructed, cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')
        plt.show()

def test_is_diag():
    import numpy as np

    # Diagonal matrix
    mat1 = np.diag([1, 2, 3])
    assert is_diag(mat1)

    # Non-diagonal matrix
    mat2 = np.array([[1, 2], [0, 1]])
    assert not is_diag(mat2)

    # Nearly diagonal (off-diagonal below tolerance)
    mat3 = np.array([[1, 1e-5], [0, 2]])
    assert is_diag(mat3)

    # Nearly diagonal (off-diagonal above tolerance)
    mat4 = np.array([[1, 1e-3], [0, 2]])
    assert not is_diag(mat4)

    print("All tests passed.")


def is_diag(mat, tol=1e-4):
    """
    Returns boolean answer to whether or not mat is a diagonal matrix
    :param mat: The matrix
    :param tol: The tolerance for element difference from different
    :return: Boolean check
    """
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i != j and mat[i, j] > tol:
                return False

    return True


def is_orthonormal(mat, tol=1e-4):
    """
    Prints 'yes', 'no', or 'trivially no' for row and column orthogonality and normality of mat.
    """
    n, m = mat.shape
    print(f"Matrix shape: {n} rows, {m} columns")

    # Row orthogonality
    if n > m:
        print("Row orthogonality: trivially no")
    else:
        G_row = mat @ mat.T
        print("Row orthogonality:", "yes" if is_diag(G_row) else "no")

    # Row normality
    row_norms = np.linalg.norm(mat, axis=1)
    print("Row normality:", "yes" if np.allclose(row_norms, 1, atol=tol) else "no")

    # Column orthogonality
    if m > n:
        print("Column orthogonality: trivially no")
    else:
        G_col = mat.T @ mat
        print("Column orthogonality:", "yes" if is_diag(G_col) else "no")

    # Column normality
    col_norms = np.linalg.norm(mat, axis=0)
    print("Column normality:", "yes" if np.allclose(col_norms, 1, atol=tol) else "no")