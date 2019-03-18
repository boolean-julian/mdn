    function [h1,h2,h3,h4,h5,h6,h7,h8]=Hg(x)
    % this function file gives the Hessians of the
    % component functions of g
    
    h1=[zeros(6,7);zeros(1,6),6];
    h2=  zeros(7,7);
    h3=  h2;
    h4=  [[-6,zeros(1,6);
    0,-8,zeros(1,5);
    0,0,-4,0,0,0,0];
    zeros(4,7)];
    h5=  [[-10,zeros(1,6);
    zeros(1,7);
    0,0,-2,0,0,0,0];
    zeros(4,7)];
    h6=  [-1,zeros(1,6);
    0,-4,0,0,0,0,0;
    zeros(2,7);
    0,0,0,0,-6,0,0;
    zeros(2,7)];
    h7=  [-2,2,0,0,0,0,0;
    2,-4,0,0,0,0,0;
    zeros(5,7)];
    h8=  [zeros(4,7);
    zeros(1,5),-12,0;
    zeros(1,4),-12,0,0;
    zeros(1,7)];


