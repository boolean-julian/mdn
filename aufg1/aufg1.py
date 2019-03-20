from PIL import Image
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import os

def tridiag(a,b,c,size):
	return np.diag([a]*(size-1),-1) + np.diag([b]*(size),0) + np.diag([c]*(size-1),1)

def set_margin(img, margin):
	img[0] 		= margin
	img[-1] 	= margin
	img[:,0] 	= margin
	img[:,-1] 	= margin

	return img

def create_image(size, margin = 0):
	img = np.random.rand(size, size)
	#big spot in center
	"""
	img = np.zeros((size,size))
	
	b = 5
	for i in range(size//2-b, size//2+b):
		for j in range(size//2-b, size//2+b):
			img[i,j] = 1
	"""
	img = set_margin(img, margin)
	
	return img

alpha = 1
delta_t = 1
hsquared = 1
coef = (alpha*delta_t)/hsquared

def _neighborhood(i,j):
	return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

@jit
def iter(A, margin = 0):
	img = (1-coef)*A.copy()
	for i in range(1, len(A)-1):
		for j in range(1, len(A[0])-1):
			s = 0
			for (k,l) in _neighborhood(i,j):
				s += A[k,l]
			img[i,j] += s/4 * coef

	img = set_margin(img, margin)
	return img

def make_movie(A, N, heat=False):
	for i in range(N):
		print("{}%".format(int(i/N*100)))
		number = str(i).zfill(3)

		filename = "video/frame{}.png".format(number)

		if heat:
			fig = plt.figure()
			plt.imshow(A, cmap='hot', interpolation='nearest')
			plt.savefig(filename)
			plt.close(fig)

		else:
			a = Image.fromarray(np.uint8(A*255))
			a.save(filename)

		A = iter(A)

	os.system("ffmpeg -f image2 -r 30 -i video/frame%03d.png -vcodec mpeg4 -y video.mp4")


A = create_image(200)
make_movie(A, 300, heat = True)