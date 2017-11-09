function [ xconvergence, fconvergence ] = BFGS_WLS( func, x )
%Quasi Newton BFGS method with Amrijo inexact line search

% Initializing
n=length(x);
sigma=0.25;
epsilon=10^-5;
a0=1;
B=eye(n); % initial guess
k=1;
k_max=10^5;
[f,g]=func(x);
fconvergence=zeros(k_max,1);
xconvergence=zeros(k_max,n);

% Quasi Newton BFGS
while(norm(g)>=epsilon && k<k_max)
    fconvergence(k,1)=f;
    xconvergence(k,:)=x;
    x_prev=x;
    g_prev=g;
    % step 1: obtain d
    d=-B*g;
    % step 2: line search
    gtd=g'*(d/norm(d));
    alpha=WolfeLineSearch(x,a0,d,f,g,gtd,sigma,0.9,2,false,25,0,false,false,false,func);
    % step 3: calculate new x,f,g
    x=x+alpha*d;
    [f,g]=func(x);
    p=x-x_prev;
    q=g-g_prev;
    % step 4: updating B
    d_norm=d/norm(d);
    if ((g_prev'*d_norm)<(g'*d_norm) || k==1 )
        s=B*q;
        tau=s'*q;
        meu=p'*q;
        v=(p/meu)-(s/tau);
        B=B+((p*p')/meu) -((s*s')/tau)+(tau*(v*v'));
    end
    k=k+1;
end

fconvergence=fconvergence(1:k-1,:);
xconvergence=xconvergence(1:k-1,:);

end

