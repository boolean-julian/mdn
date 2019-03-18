%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  sample_dvr.m
%                   Russell Luke
%  Department of MathSci, University of Delaware  
%                        and 
%  Inst. f. Num. u. Angew. Math., Universitaet Goettingen
%                    JUNE. 2, 2010  
%
%  This work was supported by NSF-DMS grant 0712796. 
%
%
% DESCRIPTION: Matlab function-file for the Rosenbrock Function
% 
% USAGE: 
%         [f,gf] = Rosenbrock1000(x)
%

function [f,gf] = Rosenbrock1000(x)

dim = length(x);
f  = (sum(100*(x(dim/2+1:dim)-x(1:dim/2).^2).^2 + 1*(1 - x(1:dim/2)).^2));
if(nargout==2)
    gf = [-400*(x(dim/2+1:dim)-x(1:dim/2).^2).*x(1:dim/2) - 2*(1 - x(1:dim/2));...
    200*(x(dim/2+1:dim)-x(1:dim/2).^2)];
end




