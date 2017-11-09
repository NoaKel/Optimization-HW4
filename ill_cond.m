function [ f,g,H ] = ill_cond( x )
%my example for ill conditioned function

n= length(x);
A= diag(linspace(1,n*100,n));  
b= linspace(1/10,n*10,n);
c=1;

f=0.5*x'*A*x+b*x+c;

if nargout >1
    g=A*x+b';
    if nargout >2
        H=A;
    end
end

end

