function [s] = armijo_step(s,x,func,f0,g,d,sigma,beta)
% Armijo step size, inexact line search
% sigma - by how much should the function decrease (defines what is a large
% engouth desecnt)
% beta- by how much will we decrease the step if conditions are
% not satisfied
    c=sigma*g'*d;
    while (func(x+s*d)>f0+s*c)
        s=beta*s; 
    end
end

