close all; clear all; clc; 
n=10;
x_start=zeros(n,1);

% analytical p* computation
p_rb=0;
p_wc=-0.5*sum(linspace(1,n,n))+1; 
p_ic=sum(linspace(1,n*100,n))/200-sum(linspace(1/10,n*10,n))/10+1;

%% Quasi Newton BFGS with Amrijo Rule
% remark: in order to calculate error according to best numerical result
% instead of anlytic take f_rb(end), f_wc(end), f_ic(end) instead of p_rb,p_wc,p_ic  

[x_rb,f_rb]=BFGS(@rosenbrock,x_start);
bfgs_rosenbrock_error=f_rb-p_rb;

[x_wc,f_wc]=BFGS(@well_cond,x_start);
bfgs_well_cond_error=f_wc-p_wc;

[x_ic,f_ic]=BFGS(@ill_cond,x_start);
bfgs_ill_cond_error=f_ic-p_ic;

% comparison to fminunc.m
options = optimoptions(@fminunc,'Algorithm','quasi-newton');
[xval_rb,fval_rb,~,output_rb]=fminunc(@rosenbrock, x_start, options );
[xval_wc,fval_wc,~,output_wc]=fminunc(@well_cond, x_start, options );
[xval_ic,fval_ic,~,output_ic]=fminunc(@ill_cond, x_start, options );

comparex_rosenbrock=norm(x_rb(end)-xval_rb);
comparef_rosenbrock=abs(f_rb(end)-fval_rb);
comparex_well_cond=norm(x_wc(end)-xval_wc);
comparef_well_cond=abs(f_wc(end)-fval_wc);
comparex_ill_cond=norm(x_ic(end)-xval_ic);
comparef_ill_cond=abs(f_ic(end)-fval_ic);



%% plots
figure;
semilogy(1:length(bfgs_rosenbrock_error),bfgs_rosenbrock_error);
xlabel('Iteration Number');
ylabel('Error on f(x)');
title('Qusai Newton BFGS - Rosenbrock');

figure;
semilogy(1:1:length(bfgs_well_cond_error),bfgs_well_cond_error);
xlabel('Iteration Number');
ylabel('Error on f(x)');
title('Qusai Newton BFGS - Well Conditioned Function');

figure;
semilogy(1:1:length(bfgs_ill_cond_error),bfgs_ill_cond_error);
xlabel('Iteration Number');
ylabel('Error on f(x)');
title('Qusai Newton BFGS - Ill Conditioned Function');

f = figure('Position',[400 200 500 100]);
d = [comparef_rosenbrock, comparex_rosenbrock  ; comparef_well_cond, comparex_well_cond ; comparef_ill_cond, comparex_ill_cond ];
cnames = {'abs(fBFGS-fFminunc)','norm(xBFGS-xFminunc)'};
rnames = {'Rosenbrock','Well Conditioned','Ill Conditioned'};
t = uitable(f,'Data',d,'ColumnName',cnames,'RowName',rnames,'ColumnWidth',{90});
set(t,'Position',[0 0 1500 100]);

%% Bonus: Quasi Newton BFGS with Wolfe Line Search

[x_min_rb,f_rb_wls]=BFGS_WLS(@rosenbrock,x_start);
bfgs_rosenbrock_error=f_rb_wls-p_rb;

figure;
semilogy(1:length(bfgs_rosenbrock_error),bfgs_rosenbrock_error);
xlabel('Iteration Number');
ylabel('Error on f(x)');
title('Bonus: Qusai Newton BFGS with Wolfe Line Search - Rosenbrock');

%% Bonus: 2D Rosenbrock BFGS Convergence

banana = @(x,y) (1-x).^2 + 100*(y-x.^2).^2;

x = linspace(-1.5,1.5);
y = linspace(-1,1.5);
[xx,yy] = meshgrid(x,y); 
ff = banana(xx,yy);

levels =[1:1:10,10:10:100];
figure;
contour(x,y,ff,levels);
colorbar;
axis([-1.5 1.5 -1 1.5]); axis square; hold on;
title('Rosenbrock function 2D')

[x_bfgs,~]=BFGS(@rosenbrock,[-0.5;1.5]);

plot(x_bfgs(:,1),x_bfgs(:,2),'k','LineWidth',2);

legend('Rosenbrock Function','BFGS Quasi Newton');