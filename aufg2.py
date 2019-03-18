import numpy as np
import sys

# Gegebene Funktionen
def g(x):
	r = np.array([0]*8)
	r[0] = 105	-4*x[0]	-5*x[1]	+3*x[6]**2
	r[1] = -10*x[0]	+8*x[2]+	x[6]
	r[2] = 8*x[0]	-2*x[1]	+12*x[4]	-5
	r[3] = -3*(x[0]-2)**2 -4*(x[1]-3)**2-2*x[2]**2+7*x[3]+100
	r[4] = -5*x[0]**2-8*x[1]-(x[2]-6)**2+2*x[3]+40
	r[5] = -0.5*(x[0]-8)**2-2*(x[1]-4)**2-3*x[4]**2+x[5]+30
	r[6] = -x[0]**2-2*(x[1]-2)**2+2*x[0]*x[1]-14*x[4]+6*x[5]
	r[7] = 3*x[0]-6*x[1]-12*x[4]*x[5]

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
	])

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

	return np.array([h1,h2,h3,h4,h5,h6,h7,h8])

def func(x):
	d = g(x)
	return 0.5*np.absolute(np.dot(d,d))

def gradf(x):
	return np.matmul(Jg(x).T, g(x))

def hessf(x):
	h = Hg(x)
	z = g(x)

	s = np.matmul(Jg(x).T, Jg(x))
	for i in range(len(z)):
		s = s + z[i]*h[i]

	return s

# Aufgabe 2
def newtons_naive(curr):
	complete = False
	x0 = curr
	while not complete:
		Inv = np.linalg.inv(hessf(curr))
		curr = curr - np.matmul(Inv, gradf(curr))

		if np.linalg.norm(gradf(curr)) < 1:
			print(curr)
			complete = True
	return curr

"""tbc
def _bfgs(Mold, xold, xnew):
	s = xnew - xold
	y = gradf(xnew) - gradf(xold)

	A = 1/np.dot(y,s) * np.outer(y,y)
	B = 1/np.dot(s, np.matmul(Mold, s)) * np.matmul(Mold, np.matmul(np.outer(s,s), Mold))

	return np.linalg.inv(Mold + A + B)
"""

x = [0,0,0,0,0,0,0]

"""debug
print("\ng:\n", g(x))
print("\nJg:\n", Jg(x))
print("\nHg:\n", Hg(x))

print("\nf:\n", func(x))
print("\ngradf:\n", gradf(x))
print("\nhessf:\n", hessf(x))
"""

newtons_naive(x)