import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# show significant figures
np.set_printoptions(precision=11)

# input functions
def g(x):
	r = np.array([0]*8, dtype=np.float64)
	r[0] = 105	-4*x[0]				-5*x[1]			+3*x[6]**2
	r[1] = 		-10*x[0]			+8*x[2]			+x[6]
	r[2] = 		8*x[0]				-2*x[1]			+12*x[4]		-5
	r[3] = 		-3*(x[0]-2)**2 		-4*(x[1]-3)**2 	-2*x[2]**2 		+7*x[3] 	+100
	r[4] = 		-5*x[0]**2- 		8*x[1] 			-(x[2]-6)**2 	+2*x[3] 	+40
	r[5] = 		-0.5*(x[0]-8)**2	-2*(x[1]-4)**2 	-3*x[4]**2 		+x[5] 		+30
	r[6] = 		-x[0]**2- 			2*(x[1]-2)**2 	+2*x[0]*x[1] 	-14*x[4] 	+6*x[5]
	r[7] = 		3*x[0] 				-6*x[1] 		-12*x[4]*x[5]

	return r

def Jg(x):
	return np.array([
		[-4,				-5,						0,				0,		0,			0,			6*x[6]],
		[-10,				0,						8,				0,		0,			0,			1],
		[8,					-2,						0,				0,		12,			0,			0],
		[-6*(x[0]-2),		-8*(x[1]-3),			-4*x[2],		7,		0,			0,			0],
		[-10*x[0],			-8,						-2*(x[2]-6),	2,		0,			0,			0],
		[-(x[0]-8),			-4*(x[1]-4),			0,				0,		-6*x[4],	1,			0],
		[-2*x[0]+2*x[1],	-4*(x[1]-2)+2*x[0],		0,				0,		-14,		6,			0],
		[3,					-6,						0,				0,		-12*x[5],	-12*x[4],	0]
	], dtype=np.float64)

def Hg(x):
	h1 = np.diag([0,		0,		0,		0,		0,		0,		6])
	h2 = np.zeros((7,7))
	h3 = np.zeros((7,7))
	h4 = np.diag([-6,		-8,		-4,		0,		0,		0,		0])
	h5 = np.diag([-10,		0,		-2,		0,		0,		0,		0])
	h6 = np.diag([-1,		-4,		0,		0,		-6,		0,		0])
	
	h7 = np.diag([-2,		-4,		0,		0,		0,		0,		0])
	h7[0,1] = 2
	h7[1,0] = 2

	h8 = np.zeros((7,7))
	h8[4,5] = -12
	h8[5,4] = -12

	return np.array([h1,h2,h3,h4,h5,h6,h7,h8], dtype=np.float64)

def func(x):
	d = g(x)
	return np.float64(0.5*np.absolute(np.dot(d,d)))

def gradf(x):
	return np.array(np.matmul(Jg(x).T, g(x)), dtype=np.float64)

def hessf(x):
	h = Hg(x)
	z = g(x)

	s = np.array(np.matmul(Jg(x).T, Jg(x)),dtype=np.float64)
	for i in range(len(z)):
		s = s + z[i]*h[i]

	return s


# helper functions
def _save_iteration_graph(xs, filename):
	plt.figure(filename)
	plt.title(filename)
	ls = np.arange(len(xs))
	plt.plot(ls,xs)
	plt.savefig("{}.png".format(filename))

def _save_step_size_graph(xs, filename):
	plt.figure(filename)
	plt.title(filename)
	ls = np.arange(len(xs)-1)
	ps = []
	for i in range(1, len(xs)):
		ps.append( np.linalg.norm(xs[i-1] - xs[i]) )
	plt.plot(ls, ps)
	plt.savefig("{}.png".format(filename))

# newton related things
def newton_with_hessian(curr, atol = 10**(-11)):
	xs = [curr]
	ys = [np.linalg.norm(gradf(xs[-1]))]

	while ys[-1] > atol:
		np.array(np.linalg.inv(hessf(xs[-1]))					,dtype=np.float64)
		hessf_inv = np.array(np.linalg.inv(hessf(xs[-1]))		,dtype=np.float64)
		xs.append( xs[-1] - np.matmul(gradf(xs[-1]),hessf_inv) )
		ys.append( np.linalg.norm(gradf(xs[-1])) )
	return xs, ys

def quasi_newton_bfgs(curr, atol = 10**(-11)):
	xs = [curr]
	ys = [np.linalg.norm(gradf(xs[-1]))]

	B = np.array(np.eye(len(curr)), dtype=np.float64)

	alphas = [2]
	for i in range(10):
		alphas.append(alphas[-1]/2)

	while ys[-1] > atol:
		p = solve(B, (-1)*gradf(xs[-1]))

		fargmin = []		
		for a in alphas:
			fargmin.append(func(xs[-1] + a*p))
		alpha = alphas[np.argmin(fargmin)]
		
		s = alpha * p
		xs.append(xs[-1] + s)
		ys.append(np.linalg.norm(gradf(xs[-1])))

		y = gradf(xs[-1]) - gradf(xs[-2])

		A = 1/np.dot(y,s) * np.outer(y,y)
		C = 1/np.dot(s, np.matmul(B,s))*np.matmul(B, np.matmul(np.outer(s,s), B.T))

		B = B + A - C

	return xs, ys, B


x = [0,0,0,0,0,0,0]

xs, ys = newton_with_hessian(x)
_save_iteration_graph(ys, "iteration-hessian")
_save_step_size_graph(xs, "step-size-hessian")

xs, ys, H_bfgs = quasi_newton_bfgs(x)
_save_iteration_graph(ys, "iteration-bfgs")
_save_step_size_graph(xs, "step-size-bfgs")