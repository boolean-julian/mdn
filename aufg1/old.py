import numpy as np
from PIL import Image

np.set_printoptions(precision=3)

margin = 1

def create_image(height, width):
	#img = np.random.rand(height, width)
	img = np.zeros((height,width))

	img[0] 		= margin
	img[-1] 	= margin
	img[:,0] 	= margin
	img[:,-1] 	= margin
	
	#img[height//2, width//2] = 1
	#img[height//2+1, width//2+1] = 1

	size = 20
	for i in range(height//2-size, height//2+size):
		for j in range(width//2-size, width//2+size):
			img[i,j] = 1

	return img

def _neighbors(i,j):
	return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

def _step(image, dt = 0.5):
	(height, width) = image.shape
	r = np.zeros(image.shape)

	for i in range(1,height-1):
		for j in range(1,width-1):
			for (k,l) in _neighbors(i,j):
				r[i,j] += image[k,l]
			r[i,j] += image[i,j]

	r = 0.2*r

	r[0] = margin
	r[-1] = margin
	r[:,0] = margin
	r[:,-1] = margin

	return r

image = create_image(100,100)

for i in range(1000):
	if i % 50 == 0:
	image = _step(image)
