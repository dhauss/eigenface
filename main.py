# Used for matrix operations
import numpy as np

# Used for loading images
import cv2 as cv

# Used to navigate directories
import os

# Training image set path
TRAINING_FACE_SET = "Face dataset/Training/"
TESTING_FACE_SET = "Face dataset/Testing/"
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

    # Create training face matrix, return mean face list 
    A, m = calculuate_training_face_matrix(TRAINING_FACE_SET)

    V = get_eigenvector_matrix(A)

    U = get_eigenface_matrix(A, V)

    #get matrix of eigen coefficients for every training face
    O = get_eigen_coefficient_matrix(A, U)

    I_coefficients = get_input_coefficients(TESTING_FACE_SET + "subject01.normal.jpg", U, m)

    match_coefficients = classify_input_face(I_coefficients, O)

    reconstructed_match = reconstruct_face(match_coefficients, U)

    # Debugging
    #print(reconstructed_match.shape)


def calculuate_training_face_matrix(directory):
    """
    Return a 2D numpy matrix made up of normalized column vector representations of the training images and mean face to normalize input face later
    : param directory: The path of the directory containing the training images
    """

    # Store column vectors
    column_vectors = []
    # Iterate through training images
    for filename in os.listdir(directory):
        # Create image object
        img = cv.imread(directory + filename, 0)
        print(filename)
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
    # Initialize mean face
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
    
    return A.T, m

def get_eigenvector_matrix(A):
    L = A.T * A 
    w, V = np.linalg.eig(L)

    return V

def get_eigenface_matrix(A, V):
    U = A * V

    return U

def get_eigen_coefficient_matrix(A, U):
    omegas = np.zeros((N_TRAINING_IMAGES, N_TRAINING_IMAGES))
    for i in range(0, N_TRAINING_IMAGES):
        omega_i = U.T * A[:, i]
        for j in range(0, N_TRAINING_IMAGES):
            omegas[i][j] = omega_i[j][0]
      
    return omegas

def get_input_coefficients(input_face, U, m):
        img = cv.imread(input_face, 0)
        # store image shape
        rows, cols = img.shape

        # Store image pixel values
        R_i = []
        for i in range(rows):
            for j in range(cols):
                pixel = img[i, j]
                R_i.append(pixel)
        
        #normalize R_i
        for i in range(len(R_i)):
            R_i[i] -= m[i]

        R_i = np.array(R_i)
        omega_R_i = U.T @ R_i

        return omega_R_i.T

def classify_input_face(test_coefficients, training_coefficients_matrix):
    """
    returns column vector of training face eigen coefficients
    : param test_coefficients: eigen coefficients of input face
    : param training_coefficients_matrix: matrix of eigen coefficients of training faces
    """

    #MDC, max value is minimum distance, full equation: X^TX - (2R^TX - R^TR) where X is test_coefficients, R is candidate class
    d_max = float('-inf')
    match = np.zeros((8, 1))
    for i in range(0, N_TRAINING_IMAGES):
        #candidate = training_coefficients_matrix[:, i]
        candidate = np.zeros((8, 1))
        for j in range(0, N_TRAINING_IMAGES):
            candidate[j][0] = training_coefficients_matrix[j][i]
        d = 2 * candidate.T @ test_coefficients - candidate.T @ candidate
        if d > d_max:
            d_max = d
            print(i)
            for j in range(0, N_TRAINING_IMAGES):
                match[j][0] = training_coefficients_matrix[j][i]


    return match
    


def reconstruct_face(omega, U):
    """
    returns reconstructed matrix of pixel values
    : param omega: vector of eigen coefficients
    : param U: eigenface matrix from training images
    """
    #reconstruct (height by width) x 1 face vector
    face_column = U * omega

    reconstructed_face = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    k = 0
    for i in range(0, IMAGE_HEIGHT):
        for j in range(0, IMAGE_WIDTH):
            reconstructed_face[i][j] = face_column[k, 0]
            k += 1

    return reconstructed_face


def save_face(face):
    """
    normalize values and save an output image of reconstructed training face
    """
    ...
    return None


if __name__ == "__main__":
    main()
