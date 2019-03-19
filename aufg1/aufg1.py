from PIL import Image
import numpy as np

def tridiag(a,b,c,size):
	return np.diag([a]*(size-1),-1) + np.diag([b]*(size),0) + np.diag([c]*(size-1),1)

def set_margin(img, margin):
	img[0] 		= margin
	img[-1] 	= margin
	img[:,0] 	= margin
	img[:,-1] 	= margin

def create_image(size, margin = 0):
	img = np.random.rand(size, size)
	""" big spot in center
	img = np.zeros((size,size))
	b = 20
	for i in range(size//2-b, size//2+b):
		for j in range(size//2-b, size//2+b):
			img[i,j] = 1
	"""
	return img

def iter(A):
	# Bla bla
	return A

def make_movie(A):
	for i in range(300):
		a = Image.fromarray(np.uint8(image*255))
		number = str(i).zfill(3)
		a.save("movie/frame{}.png".format(number))
		A = iter(A)

L = tridiag(1,-4,1,10)
print(L)