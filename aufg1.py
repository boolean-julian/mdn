import numpy as np
from numba import jit
from PIL import Image
import os
import sys


ALGO = "GAUSS"
np.set_printoptions(suppress=True, precision=2)

# Parameters
size = 30		# sqrt of samples
frames = 150 	# number of video frames
alpha = 0.2	# intensity [0,1)
k = 6			# iterations per frame


"""
GAUSS RELATED STUFF DO NOT TOUCH
"""

"""
# Checks whether matrix is quadratic
@jit
def check_mat(A):
	n = np.shape(A)[0]
	if n != np.shape(A)[1]:
		print("Error: Input matrix must have shape (n,n)")
		sys.exit()
	return n

# Checks if dimension of b is n
@jit
def check_b(b, n):
	return True if len(b) == n else False
"""

# Triangular segmentation of a matrix
@jit
def lu_zerlegung(A):
#	n = check_mat(A)
	n = A.shape[0]

	U = np.array(A, dtype="float64")
	L = np.eye(n, dtype="float64")

	for i in range(n-1):
		for k in range(i+1, n):
			L[k,i] = U[k,i] / U[i,i]
			for j in range(i, n):
				U[k,j] = U[k,j] - L[k,i] * U[i,j]
	return L, U

# It does what it says it does
@jit
def vorwaertselimination(A,b):
#	n = check_mat(A)
	n = A.shape[0]

	V = np.array(np.c_[A,np.array(b)], dtype="float64")
	for i in range(n):
		V[i] = V[i] * 1/V[i,i]
		for j in range(i+1,n):
			if V[i,i] != 0:
				V[j] = V[j] - V[i] * V[j,i]/V[i,i]
	return V

# It does what it says it does
@jit
def rueckwaertselimination(A,b):
	# check dimensions
#	n = check_mat(A)
	n = A.shape[0]

	B = np.array(np.c_[A, np.array(b)], dtype="float64")
	for i in reversed(range(n)):
		B[i] = B[i] * 1/B[i,i]
		for j in reversed(range(i)):
			if B[i,i] != 0:
				B[j] = B[j] - B[i] * B[j,i]/B[i,i]
	return B

@jit
def solve(L,U,b):
#	L, U = lu_zerlegung(A)
	Lx = vorwaertselimination(L,b)
	x = rueckwaertselimination(U,Lx[:,-1])
	return x[:,-1]

#A = np.array([[1,2,3],[1,1,1],[3,3,1]])
#b = np.array([2,2,0])
#L,U = lu_zerlegung(A)
#print(solve(L,U,b), "------")

"""
GAUSS OVER
"""

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


#K = [(0,a) for a in range(size)] + [(a, size-1) for a in range(size)] + [(size-1,a) for a in range(size-1,-1,-1)] + [(a,0) for a in range(size-1,-1,-1)]
@jit
def make_movie(U, N, alpha, k, iterate):
	for i in range(N):
		print("{:3.2f}%".format(i/N*100))
		number = str(i).zfill(4)

		filename = "video/frame{}.png".format(number)

		a = Image.fromarray(_project_color_space(U))
		a.save(filename)

		for j in range(k):
			
			U = iterate(U, alpha)
			#U[K[i%len(K)]] = 1
			"""
			if i < N/2:
				U[0,:]	= 0
				U[-1,:]	= 0
				U[:,-1]	= 0
				U[:,0]	= 1
			else:
				U[0,:]	= 0
				U[-1,:]	= 0
				U[:,-1]	= 1
				U[:,0]	= 0
			"""
	os.system("ffmpeg -f image2 -r 30 -i video/frame%04d.png -vcodec libx264 -crf 15 -y heatmap.mp4")



# Implementation method #1 for FDM, element-wise

def get_neighbors(i,j):
    return [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]


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

def tridiag(a,b,c,N):
    return np.diag([a]*(N-1), -1) + np.diag([b]*N, 0) + np.diag([c]*(N-1), 1)

def heat_matrix_vector(N,alpha):
    A = tridiag(1,-4,1, N**2)
    B = np.diag([1]*(N**2-N), N)
    return 0.25*alpha*(A+B+B.T)+np.eye(N*N)

if ALGO == "FDM":
	HMV = heat_matrix_vector(size, alpha)

def iterate_matrix_vector(A,bloop):
    B = (HMV @ A.reshape(len(A)**2, 1)).reshape(len(A),len(A))
    return B


# Implementation method #3 for FDM, two matrix multiplications

def heat_matrix(N, alpha):
	return 0.5*alpha*tridiag(1,-2,1, N) + np.eye(N)

if ALGO == "FDMkurz":
	HMM = heat_matrix(size,alpha)

@jit
def iterate_matrix(A, bloop):
	return 0.5*(HMM@A + A@HMM)


def matrix_cg_explicit(N,alpha):
    A = tridiag(1,-4,1, N**2)
    B = np.diag([1]*(N**2-N), N)
    return np.linalg.inv(0.25*alpha*(A+B+B.T)+np.eye(N*N))

def matrix_cg_implicit(N,alpha):
	M = N*N

	_L = tridiag(1,-4,1, M)
	_I = np.diag([1]*(M-N), N)

	A = 0.25*(_L + _I + _I.T)
	
	I = np.eye(M)

	return I - alpha*A @ np.linalg.inv(alpha*A + I)

if ALGO == "CG":
	HMV_cg = matrix_cg_explicit(size, alpha)

# Implementation of conjugate gradients
@jit
def iterate_cg(B, bloop):
	b = B.reshape(size**2,1)
	x = b
	r = b - HMV_cg@x
	d = r
	
	while np.linalg.norm(r) > 0.0001:
		temp1 = HMV_cg@d
		temp2 = r.T@r

		alpha = temp2 / (d.T@temp1)
		x 		= x + alpha*d
		r 		= r - alpha * temp1
		beta 	= (r.T@r)/temp2
		d 		= r+beta*d

	return x.reshape(size,size)

# Gauss Elimination
if ALGO == "GAUSS":
	HMV_gauss = matrix_cg_explicit(size, alpha)
	L_gauss, U_gauss = lu_zerlegung(HMV_gauss)

def iterate_gauss(B, bloop):
	b = B.reshape(size**2,1)
	return solve(L_gauss,U_gauss,b).reshape(size,size)


# some random image to start off with
A = np.zeros((size,size))

c = np.pi/(size-1)
for i in range(len(A)):
	for j in range(len(A[0])):
		A[i,j] = (np.sin(c*i)*np.sin(c*j))**2



make_movie(A, frames, alpha, k, iterate_gauss)


#P = iterate_cg(A,alpha)
#L = iterate_gauss(A,alpha)

#print(A)
#for _ in range(50):
#	A = iterate_gauss(A,alpha)
#print(A)
#print(iterate_gauss(A,alpha))

#Q = np.linalg.solve(HMV_cg, A.reshape(size**2,1)).reshape(size,size)
#R = iterate_for(iterate_for(A,alpha),alpha)

#print(np.allclose(P,Q) and np.allclose(P,R))
#print(np.allclose(P,R))
#print(P)
#print(R)
#print(HMV_cg)
"""
B = np.linalg.solve(HMV_cg, A.reshape(size**2,1))
B = np.linalg.solve(HMV_cg, B).reshape(size,size)

print(np.allclose(P,B))

print(P)
print()
print(B)
"""
