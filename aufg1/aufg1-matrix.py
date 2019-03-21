import numpy as np
from numba import jit
from PIL import Image
import os


@jit
def L(N, h = 0.25):
	A = np.diag([-4]*N**2, 0)
	B = np.diag([1]*(N**2-1), -1)
	C = np.diag([1]*(N**2-N), N)

	return h * (A+B+B.T+C+C.T)*0.25 + np.eye(size*size)

def create_image(size, boundary = 0):
	#img = np.random.rand(size, size)
	#big spot in center
	
	img = np.zeros((size,size))

	b = 5
	for i in range(size//2-b, size//2+b):
		for j in range(size//2-b, size//2+b):
			img[i,j] = 1
		
	return img.reshape(size*size)

def make_movie(U, A, N, size, heat=False):
	for i in range(N):
		print("{}%".format(int(i/N*100)))
		number = str(i).zfill(3)

		filename = "video2/frame{}.png".format(number)

		Ureshaped = U.reshape((size,size))

		if heat:
			fig = plt.figure()
			plt.imshow(Ureshaped, cmap='hot', interpolation='nearest')
			plt.savefig(filename)
			plt.close(fig)

		else:
			a = Image.fromarray(np.uint8(Ureshaped*255))
			a.save(filename)

		U = A@U 

	os.system("ffmpeg -f image2 -r 30 -i video2/frame%03d.png -vcodec mpeg4 -y video2.mp4")


size = 100
frames = 300
A = L(size)
U = create_image(size)
make_movie(U, A, frames, size)