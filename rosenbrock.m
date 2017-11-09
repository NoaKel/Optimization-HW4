function [ f,g,H ] = rosenbrock( x )
% rosenbrock function

n=length(x); 
f= sum((ones(n-1,1)-x(1:n-1)).^2 + 100*(x(2:n)-x(1:n-1).^2).^2);
 
if nargout>1 % anaytical computition of gradient
    g1(1:n-1)=-2*(ones(n-1,1)-x(1:n-1))-400*x(1:n-1).*(x(2:n)-x(1:n-1).^2);
    g1(n)=0;
    g2(2:n)=200*(x(2:n)-x(1:n-1).^2);
    g2(1)=0;
    g=g1+g2; g=g(:);
    
    if nargout>2 % anaytical computition of hessian
        H=-400*diag(x(1:n-1),1)-400*diag(x(1:n-1),-1)+...
            diag([0 200*ones(1,n-1)])+...
            diag([2*ones(1,n-1)-400*(x(2:n)'-3*x(1:n-1).^2') 0]);
    end
end
    
end 
