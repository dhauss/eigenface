# Implement Eigneface
# Please see project description below. Due date: Tues 12/20 @ 11:00 pm. Deadline for late Project 2 (with penalty): Fri 12/30 @ 11:00 pm. The Face dataset for training and testing your program is attached below. Face dataset: The Face Dataset consists of the Training Set and the Testing Set. The images are in .jpg format and are of dimension 195 x 231 (width x height) pixels. Each pixel uses 8 bits for grayscale. The Training Set consists of eight images from eight different persons. Use all eight training images in the Training Set to train for the Eigenfaces and put all eight Eigenfaces in the matrix U. The Testing Set consists of six images from four different persons. Recognize the faces in the Testing Set using the Eigenface method.  Note: In the lecture slides, the training and test images are of size N X N (same horizontal and vertical dimensions.) The face images in the above dataset have different horizontal and vertical dimensions, but the Eigenface method works the same way with no changes to the formulas. 

# Load in training images
# Apply Priciple Componenet Analysis to training face images
# Project face image onto the face space and represent using PCA coefficients
# Using PCA coefficients as weights and reconstruct the origianl face image using a linear combination of eigenfaces (eigenvectors)
# Match testing facs by computing the distance between the PCA coefficents of the testing face with the training faces


# More technically:
# For each training image i of size X x Y pixels, create rowstacked column vectors (Ri) of dimension X x Y for M training images
# Compute mean face (m) by taking the average of the M training face images
# Subtract the mean face m from each training face, ->Ri = Ri - m
# Put all training faces into a single matrix A (dimension N^2 x M): [R1, r2, r3, ..., Rm]
# ...