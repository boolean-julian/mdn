x = [0,0,0,0,0,0,0]'
while norm(gradf(x)) > 10^-11
    hessf_inv = inv(hessf(x))
    x = x - (hessf_inv * gradf(x));
end
x