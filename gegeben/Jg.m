function m=Jg(x)
% this one does the Jacobian of the
% sample function g
m=[
-4				-5					0 			0 0 		0 			6*x(7);
-10				0					8 			0 0 		0 			1;
8				-2 					0 			0 12 		0 			0;
-6*(x(1)-2)		-8*(x(2)-3) 		-4*x(3) 	7 0 		0 			0;
-10*x(1)		-8 					-2*(x(3)-6) 2 0 		0 			0;
-(x(1)-8)		-4*(x(2)-4) 		0 			0 -6*x(5) 	1 			0;
-2*x(1)+2*x(2)	-4*(x(2)-2)+2*x(1) 	0 			0 -14 		6 			0;
3				-6 					0 			0 -12*x(6) 	-12*x(5) 	0];

