function [ f,g,H ] = well_cond( x )
%my example for well conditioned function

n= length(x);
A= diag(linspace(1,n,n));  
b= linspace(1,n,n);
c=1;

f=0.5*x'*A*x+b*x+c;

if nargout >1
    g=A*x+b';
    if nargout >2
        H=A;
    end
end

end 

