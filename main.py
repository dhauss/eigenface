# Used for matrix operations
import numpy as np

# Used for loading images
import cv2 as cv

# Used to navigate directories
import os

# Training image set path
TRAINING_FACE_SET = "Face dataset/Training/"
# Training image width
IMAGE_WIDTH = 195
# training image height
IMAGE_HEIGHT = 231
# Number of Training Images
N_TRAINING_IMAGES = 8


def main():
    """
    Implement the Eigenface method for face recognition.
    """

    # Create training face matrix
    A = calculuate_training_face_matrix(TRAINING_FACE_SET)

    V = get_eigenvector_matrix(A)

    U = get_eigenface_matrix(A, V)

    C = get_covariance_matrix(A)

    # Debugging
    print(C.shape)


def calculuate_training_face_matrix(directory):
    """
    Return a 2D numpy matrix made up of normalized column vector representations of the training images
    : param directory: The path of the directory containing the training images
    """

    # Store column vectors
    column_vectors = []
    # Iterate through training images
    for filename in os.listdir(directory):
        # Create image object
        img = cv.imread(directory + filename, 0)
        # store image shape
        rows, cols = img.shape

        # Store image pixel values
        pixel_values = []
        for i in range(rows):
            for j in range(cols):
                pixel = img[i, j]
                pixel_values.append(pixel)

        # Add image column vector to unprocessed column vector matrix
        column_vectors.append(pixel_values)

    # Create mean face
    m = []
    # Initlaize mean face
    for i in range(IMAGE_WIDTH * IMAGE_HEIGHT):
        m.append(0)

    # Sum column vectors to get meanface value before averaging
    for i in range(N_TRAINING_IMAGES):
        for j in range(IMAGE_WIDTH * IMAGE_HEIGHT):
            m[j] += column_vectors[i][j]

    # Average the values of the meanface vector to get the final meanface
    for i in range(len(m)):
        m[i] = int(m[i] / N_TRAINING_IMAGES)

    # Subtract the meanface column vector from each training face
    for i in range(N_TRAINING_IMAGES):
        for j in range(IMAGE_WIDTH * IMAGE_HEIGHT):
            column_vectors[i][j] -= m[j]

    A = np.matrix(column_vectors)
    
    return A.T

def get_eigenvector_matrix(A):
    L = A.T * A 
    w, V = np.linalg.eig(L)

    return V

def get_eigenface_matrix(A, V):
    U = A * V

    return U


def get_covariance_matrix(A):
    C = A * A.T 

    return C




if __name__ == "__main__":
    main()
