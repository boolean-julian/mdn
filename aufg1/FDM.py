import numpy as np
from numba import jit
from PIL import Image
import os


# Parameters
size = 100		# sqrt of samples
frames = 3000 	# number of video frames
alpha = 1 		# intensity [0,1]
k = 6 			# iterations per frame



# Video related stuff
colors = np.array([
	[0,		0,		127],
	[0,		255,	0],
	[255,	255,	0],
	[255,	128,	0],
	[255,	0,		0]], dtype=np.float64)

@jit
def _get_color(alpha):
	n = len(colors)
	alpha = alpha*(n-1)
	for i in range(1,n):
		if alpha < i:
			alpha = alpha%1
			return np.uint8((1-alpha) * colors[i-1] + alpha * colors[i])
	return np.uint8(colors[-1])
dcolors = [_get_color(i) for i in np.linspace(0,1,256)]

@jit
def _project_color_space(U):
	A = np.array([[[0]*3]*len(U[0])]*len(U), dtype = np.uint8) 
	for i in range(len(A)):
		for j in range(len(A[0])):
			A[i,j] = dcolors[int(U[i,j]*255)]
	return A

@jit
def make_movie(U, N, alpha, k, iterate):
	for i in range(N):
		print("{:3.2f}%".format(i/N*100))
		number = str(i).zfill(4)

		filename = "video/frame{}.png".format(number)

		a = Image.fromarray(_project_color_space(U))
		a.save(filename)

		for _ in range(k):
			U = iterate(U, alpha)

	os.system("ffmpeg -f image2 -r 30 -i video/frame%04d.png -vcodec libx264 -crf 15 -y heatmap.mp4")



# Implementation method #1 for FDM, element-wise
@jit
def get_neighbors(i,j):
    return [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]

@jit
def iterate_for(A,alpha):
    U = np.zeros(np.shape(A))
    for i in range(len(A)):
        for j in range(len(A[0])):
            for (k,l) in get_neighbors(i,j):
                try:
                    U[i,j] += A[k,l]
                except:
                    pass
            U[i,j] = 0.25*alpha*U[i,j] + (1-alpha)*A[i,j]
    return U



# Implementation method #2 for FDM, matrix-vector-multiplication
@jit
def tridiag(a,b,c,N):
    return np.diag([a]*(N-1), -1) + np.diag([b]*N, 0) + np.diag([c]*(N-1), 1)

@jit
def heat_matrix_vector(N,alpha):
    A = tridiag(1,-4,1, N**2)
    B = np.diag([1]*(N**2-N), N)
    return 0.25*alpha*(A+B+B.T)+np.eye(N*N)

HMV = heat_matrix_vector(size, alpha)
@jit
def iterate_matrix_vector(A, alpha):
    B = (HMV @ A.reshape(len(A)**2, 1)).reshape(len(A),len(A))
    return B



# Implementation method #3 for FDM, two matrix multiplications
@jit
def heat_matrix(N,alpha):
	return 0.5*alpha*tridiag(1,-2,1, N) + np.eye(N)

HMM = heat_matrix(size,alpha)
@jit
def iterate_matrix(A,alpha):
	return 0.5*(HMM@A + A@HMM)



# some random image to start off with
A = np.zeros((size,size))

c = np.pi/(size-1)
for i in range(len(A)):
	for j in range(len(A[0])):
		A[i,j] = np.sin(c*i)*np.sin(c*j)

make_movie(A, frames, alpha, k, iterate_matrix)