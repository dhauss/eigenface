#  NumPy is used for the matrix operations it provides
import numpy as np

# OpenCV is used for processing images
import cv2 as cv

# The OS module is used to iteratively process image files
import os

# Set the path to the training and testing image sets
TRAINING_FACE_SET = "Face dataset/Training/"
TESTING_FACE_SET = "Face dataset/Testing/"
# Set the dimensions of the image widths and heights
IMAGE_WIDTH = 195
IMAGE_HEIGHT = 231
# Set the number of training images
N_TRAINING_IMAGES = 8


def main():
    """
    Conduct Eigenface training and Eigenface recognition.

    Eigenface training:
        Step 1: Convert M training images into column vectors
        Step 2: Compute the mean face
        Step 3: Subtract the mean face from each training face
        Step 4: Collect the training faces into one matrix
        Step 5: Find Eigenvalues by taking the the training face matrix, transposing it, and then multiplying it by itself
        Step 6: Collect the Eigenvectors of the Eigenvalues into a matrix
        Step 7: Collect the M largest Eigenvectors of the coverance matrix
        Step 8: Project each training face onto the face space to obtain its eigenface coefficients

    Eigenface Recogntion:
        Step 1: Subtract the mean face m from the input face
        Step 2: Compute the projection of the input face onto the face space to obtrain its eigenface coefficients
        Step 3: Reconstruct the input face image from eigenfaces
        Step 4: Compute the distance between the input face image and its reconstruction
        Step 5: Compute the distance between the input face and the training images in the face space (distance between eigenface coefficients)
        Step 6: The input face corresponds with the training face whose distance to the input face in face space is minimal
    """

    # Eigenface Training

    # Compute the meanface and Create the matrix of training face vectors
    A, m, fn = calculuate_training_face_matrix(TRAINING_FACE_SET)
    # Create a matrix that contains the M largest Eigenvalues
    V = get_eigenvector_matrix(A)
    # Compute the M largest Eigenvectors of the covariance matrix
    U = get_eigenface_matrix(A, V)
    # Compute the Eigenface coeffcients of the training images by projecting the training image column vectors onto face space
    O = get_eigen_coefficient_matrix(A, U)

    # Eigenface Recognition
    # Collect the filename of the input image from the user
    input_image_filename = input_image()
    # Calculate eigenface coefficients of an input face
    I_coefficients = get_input_coefficients(
        TESTING_FACE_SET + input_image_filename, U, m
    )

    # Perform the 1-NN based matching
    match_coefficients = classify_input_face(I_coefficients, O, fn)

    #save eigenface images
    save_eigen_faces(U, fn)
    #save mean face image
    save_mean_face(m)



def calculuate_training_face_matrix(directory):
    """
    Return a 2D numpy matrix made up of normalized column vector representations of the training images and the mean face
    : param directory: The path of the directory containing the training images
    """
    # Store filenames
    fn = []
    # Store column vectors
    column_vectors = []
    # Iterate through training images
    for filename in os.listdir(directory):
        # Create image object in grayscale mode
        fn.append(filename)

    for filename in fn:
        img = cv.imread(directory + filename, 0)
        # store image shape
        rows, cols = img.shape

        # Create a list of pixel values
        pixel_values = []
        for j in range(cols):
            for i in range(rows):
                pixel = img[i, j]
                # Append each pixel value to the list of pixel values
                pixel_values.append(pixel)

        # Add the list of pixel values which is functionally an unprocessed vector representation of the image to the column vector matrix
        column_vectors.append(pixel_values)

    # Create a list to represent the mean face
    m = []
    # Initialize the pixel values of the mean face
    for i in range(IMAGE_WIDTH * IMAGE_HEIGHT):
        m.append(0)

    # Sum the pixel value lists into the mean face list
    for i in range(N_TRAINING_IMAGES):
        for j in range(IMAGE_WIDTH * IMAGE_HEIGHT):
            m[j] += column_vectors[i][j]

    # Average the values of the meanface vector to get the final pixel value representation of the meanface
    for i in range(len(m)):
        m[i] = int(m[i] / N_TRAINING_IMAGES)

    # Subtract the meanface pixel value vector from each training face
    for i in range(N_TRAINING_IMAGES):
        for j in range(IMAGE_WIDTH * IMAGE_HEIGHT):
            column_vectors[i][j] -= m[j]

    # Create the matrix of training faces
    A = np.array(column_vectors)

    # Return the vector of training faces and the mean face
    return A.T, m, fn


def get_eigenvector_matrix(A):
    """
    Return the eigenvectors of the training faces
    :param A: The matrix of training faces
    """
    # Find the M largest eigenvalues
    L = A.T @ A
    # Store the M largest eigen values into one matrix
    w, V = np.linalg.eig(L)

    # Return the matrix that contains the M largest eigenvalues
    return V


def get_eigenface_matrix(A, V):
    """
    Return the M largest Eigen vectors of the coveriance matrix.
    : param A: The matrix of training faces
    : param V: The matrix of the M largest Eigenvalues
    """

    return A @ V


def get_eigen_coefficient_matrix(A, U):
    """
    Project training faces onto face space to obtain their eigenface coefficients.
    : param A: The matrix of training faces
    : param U: The matrix of the M largest Eigenvectors
    """

    #Find eigen coefficients of all training images
    omegas = U.T @ A

    # Return a matrix of training Eigenface coefficients
    return omegas


def get_input_coefficients(input_face, U, m):
    """
    Compute the Eigenface coefficients of an input face.
    : param input_face: The filename of the input face.
    : param U: The M larges Eigenvactors of the covariance matrix.
    : param m" The mean face.
    """

    # Load the input face file into an image object
    img = cv.imread(input_face, 0)
    # store image shape
    rows, cols = img.shape

    # Store image pixel values
    R_i = []
    for j in range(cols):
        for i in range(rows):
            pixel = img[i, j]
            R_i.append(pixel)

    # normalize the input face column vector
    for i in range(len(R_i)):
        R_i[i] -= m[i]

    # Compute the input face Eigen coefficients
    R_i = np.array(R_i)
    omega_R_i = U.T @ R_i

    # Return the inputface Eigen coeffcients
    return omega_R_i.T


def classify_input_face(test_coefficients, training_coefficients_matrix, filenames):
    """
    Returns the eigenface coefficients of the training face that is matched to the input face
    : param test_coefficients: eigen coefficients of the input face
    : param training_coefficients_matrix: matrix of eigen coefficients of the training faces
    """

    # MDC, max value is minimum distance, full equation: X^TX - (2R^TX - R^TR) where X is test_coefficients, R is candidate class
    # Initialize max distance
    d_max = float("-inf")
    # Initialize the eigenface coefficients of the matched training face
    match = np.zeros((8, 1))
    index = -1;
    # Iterate through all of the trainingface coefficients
    for i in range(len(filenames)):
        # Initialize a candidate set of Eigenface coefficients
        candidate = np.zeros((8, 1))
        # Iterate through the Eigen coefficients
        for j in range(0, N_TRAINING_IMAGES):
            candidate[j][0] = training_coefficients_matrix[j][i]
        # Calculate the dsitance beween the input face iand its training face
        d = 2 * candidate.T @ test_coefficients - candidate.T @ candidate
        # Identify the input face's corresponding training face
        if d > d_max:
            d_max = d
            index = i
            for j in range(0, N_TRAINING_IMAGES):
                match[j][0] = training_coefficients_matrix[j][i]

    # Return the eigenface coefficients of the training face
    print(filenames[index])
    return match


def reconstruct_face(omega, U):
    """
    Returns reconstructed matrix of pixel values
    : param omega: vector of eigen coefficients
    : param U: eigenface matrix from training images
    """
    # Reconstruct (height by width) x 1 face vector
    face_column = U @ omega

    # Initialize pixel value matrix
    reconstructed_face = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    k = 0
    # Populate pixel value matrix
    for i in range(0, IMAGE_HEIGHT):
        for j in range(0, IMAGE_WIDTH):
            reconstructed_face[i][j] = face_column[k, 0]
            k += 1

    # Return the matrix containing the pixel values
    return reconstructed_face


def save_eigen_faces(in_faces, fn):
    """
    Normalize values and save an output image of reconstructed training face
    : param face: (IMAGE_WIDTH * IMAGE_HEIGHT) * x matrix of stacked faces
    : param fn: dict of filenames indexed to training faces
    """

    #cycle through each face column
    for n in range(N_TRAINING_IMAGES):
        #find min, max for normalization and convert to IMAGE_HEIGHT x IMAGE_WIDTH sized np array
        k = 0
        max_pixel = float('-inf')
        min_pixel = float('inf')
        out_face = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        for j in range(IMAGE_WIDTH):
            for i in range(IMAGE_HEIGHT):
                out_face[i][j] = in_faces[k, n]
                k += 1
                if out_face[i][j] < min_pixel:
                    min_pixel = out_face[i][j]
                if out_face[i][j] > max_pixel:
                    max_pixel = out_face[i][j]
        
        #normalize values from 0-255
        for i in range(IMAGE_HEIGHT):
            for j in range(IMAGE_WIDTH):
                out_face[i][j] = int((out_face[i][j] - min_pixel) * 255/(max_pixel - min_pixel))
        
        out_file = "Eigenfaces/" + fn[n][:-4] + ".eigen-out" + ".jpg"
        cv.imwrite(out_file, out_face)


def save_mean_face(m):
    """
    normalize and save mean face
    : param m : list of mean values
    """
    k = 0
    max_pixel = float('-inf')
    min_pixel = float('inf')
    out_face = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for j in range(IMAGE_WIDTH):
        for i in range(IMAGE_HEIGHT):
            out_face[i][j] = m[k]
            k += 1
            if out_face[i][j] < min_pixel:
                min_pixel = out_face[i][j]
            if out_face[i][j] > max_pixel:
                 max_pixel = out_face[i][j]

            #normalize values from 0-255
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            out_face[i][j] = int((out_face[i][j] - min_pixel) * 255/(max_pixel - min_pixel))

    cv.imwrite("meanface-out.jpg", out_face)

def input_image():
    """
    Collect the name of the file that contains the input face from the user.
    """
    return input("Enter the input face filename: ")


if __name__ == "__main__":
    main()
