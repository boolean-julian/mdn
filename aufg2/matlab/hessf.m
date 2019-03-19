function y=hessf(x)
% hessf is the Hessian of the function func(x)=(1/2)*norm(g(x))^2
[h1,h2,h3,h4,h5,h6,h7,h8]=Hg(x);
z=g(x);
y=Jg(x)'*Jg(x)+z(1)*h1+z(2)*h2+z(3)*h3+z(4)*h4+z(5)*h5+z(6)*h6;
y=y+z(7)*h7+z(8)*h8;

