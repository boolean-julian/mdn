import numpy as np

def f(x):
	return x[0]**3 + (x[1])**2

def gradf(x):
	return np.array([3*x[0]**2,2*x[1]])

def hessf(x):
	return np.array([[6*x[0],0],[0,2]])

def newtons_naive(curr):
	complete = False
	x0 = curr
	while not complete:
		Inv = np.linalg.inv(hessf(curr))
		curr = curr - np.matmul(Inv, gradf(curr))

		if np.linalg.norm(gradf(curr)) < 10**-10:
			print(curr, " -> ", f(curr))
			complete = True
	return curr



x0 = [-100,100]
newtons_naive(x0)