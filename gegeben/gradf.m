function y=gradf(x)
% gradf is the gradient of func(x)=1/2*norm(g(x))^2
y=Jg(x)'*g(x);

