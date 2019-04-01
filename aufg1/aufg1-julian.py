import numpy as np
from PIL import Image
import os
from numba import jit

np.set_printoptions(precision=3)

size = 500
frames = 10000

@jit
def L(N):
	A = np.diag([-2]*N, 0)
	B = np.diag([1]*(N-1), -1)
	C = np.diag([1]*(N-1), 1)
	return 0.25*np.array(A+B+C, dtype=np.float64)

@jit
def create_image(size, boundary = 0):
	#img = np.random.rand(size, size)
	#big spot in center
	
	img = np.zeros((size,size))
	"""
	b = 10
	for i in range(size//2-b, size//2+b):
		for j in range(size//2-b, size//2+b):
			img[i,j] = 1
	"""
	
	c = 2*np.pi/(size-1)
	for i in range(size):
		for j in range(size):
			img[i,j] = np.sin(c*i)**2 * np.sin(c*j)**2

	return img

lap = L(size)
I = np.eye(size)
res = (I + lap)
def iterate(A):
	return (res@(res@A).T).T


# just some fancy stuff
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
def make_movie(U, N, heat=False):
	for i in range(N):
		print("{}%".format(i/N*100))
		number = str(i).zfill(4)

		filename = "video3/frame{}.png".format(number)
			
		a = Image.fromarray(_project_color_space(U))
		a.save(filename)

		U = iterate(U)

	os.system("ffmpeg -f image2 -r 30 -i video3/frame%04d.png -vcodec libx264 -crf 15 -y video3.mp4")

W = create_image(size)
make_movie(W, frames)
