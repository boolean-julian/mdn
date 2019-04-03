"""
n=8000 & m=100000
1.2742241528116438 berechnet nach Mises - untere Grenze
1.2742241528249134 berechnet nach EV - obere Grenze
Norm von A: 		1.2742241528		(11 Stellen)

n=8000 & m=16000
1.2742241528116438 berechnet nach Mises	- untere Grenze
1.2742241528221396 berechnet nach EV	- obere Grenze
Norm von A: 		1.2742241528		(11 Stellen)

n=12000 & m=24000
1.2742241528182108 berechnet nach Mises	- untere Grenze
1.2742241528211977 berechnet nach EV	- obere Grenze
Norm von A: 		1.2742241528		(11 Stellen)

Der Eigenwert zum betragsgrößten Eigenvektor von A_n hat immer eine spezielle Form.
Seine Einträge sind immer positiv und als Folge aufgefasst monoton fallend.
B ist symmetrisch -> diagonalisierbar -> Norm(Bv) ist größtes seiner Sorte -> Norm(Bv) = Norm(B) * Norm(v) nach Def.

"""

import numpy as np
from numpy import array, zeros, argmax, ones
from scipy.linalg import norm
from numpy.linalg import eigh
from math import pi, sqrt, log10
from numba import jit

# gerade 11 Stellen: n = 6110
# gerade 10 Stellen: n = 1794

N = 1800
TOL = 10**-12
EPS = 10**-9
M = N*2
GRE = M/5


@jit
def create(n=N): #--------------------------------------------------------------
	"""
	Erstellt die Matrix A_n
	Das ist die links obere nxn Teil-Matrix der unendlichen Matrix A
	Füllt zuerst die Nenner in eine leere Matrix und bildet danach
	für jeden Eintrag den Kehrwert
	"""
	B = zeros((n,n))
	B[0,0] = 1
	for j in range(1,n):
		B[0,j] = B[0,j-1] + j
	
	for i in range(1,n):
		B[i,0] = B[i-1,0] + i + 1
		for j in range(1,n):
			B[i,j] = B[i,j-1] + i + j
	B = 1/B
	return B

@jit
def mises(A, v, TOL=10**-12): #-------------------------------------------------
	"""
	Verfahren der Vektoriteration nach von-Mises:
	Abschätzung an den betragsgrößten Eigenwert,
	gibt außerdem auch den zugehörigen Eigenvektor zurück.
	Verfahren bricht bei erreichter Toleranz TOL ab
	"""
	y = A@v
	v = y / norm(y,2)
#	print(".")
	while abs(norm(A@v,2) - norm(y,2) * norm(v,2)) > TOL:
		y = A@v
		v = y / norm(y,2)
#		print(".")
	y = A@v
	return norm(y,2), v
	




@jit
def func(v): #------------------------------------------------------------------
	"""
	Berechnent B@v in den Schritten x = A @ v und y = A.T @ x in m Dimensionen. 
	Die Zahlen für die ersten n Dim. von v sind vorhanden, danach wird der
	letzte Eintrag, also min(v), für alle fehlenden Zahlen angenommen.
	Zurückgegeben wird die Wurzel aus der Norm von y.
	"""

#	print("Berechne A@v")	
	# A @ v
	z = 0
	x = zeros(M)
	for i in range(M):							
		z += i + 1		
		k = z
		a = v[0] * (1/k)
		for j in range(1,M):
			k += j + i
			s = 1/k
			if (j < N):
				a += s * v[j]
			else:
				a += s * v[-1]
		x[i] = a
#		if i % GRE == 0:
#			print(".")

	# A.T @ x
#	print("Berechne A.T@x")
	z = 1
	y = zeros(M)
	for j in range(M):							
		z += j
		k = z
		a = x[0] * (1/k)
		for i in range(1,M):
			k += i + j + 1
			s = 1/k
			if (i < N):
				a += s * x[i]
			else:
				a += s * x[-1]
		y[j] = a
	#	if j % GRE == 0:
#			print(".")

	return sqrt(norm(y,2))

		
	




print("n={} & m={}".format(N,M))
#print("Zielbereich: \t\t{}\t(Pi/sqrt(6))".format(pi/sqrt(6)))
#print("Erzeuge A_n")
A = create()
#print("Berechne A.T@A")
B = A.T@A					# langsam!
x = zeros(N)
x[0] = 1					# Einheitsvektor
#print("Mises")
e,v = mises(B,x,TOL)
a = sqrt(e)
b = func(v)
print(a,"berechnet nach Mises\t- untere Grenze")
print(b,"berechnet nach EV\t- obere Grenze")
for i in range(20):
	if (int(a * 10**i) != int(b * 10**i)):
		break
c = int(a * 10**(i-1)) / 10**(i-1)
print("Norm von A: \t\t{}\t\t({} Stellen)\n".format(c,i))


# Relikte ------------
#et,vt = mises(A@A.T,x,TOL)
#print(sqrt(et))
#print(sqrt(vektornorm(e*v)))



