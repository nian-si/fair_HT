function [var] = compute_empirical_variance(X,A,Y,theta,tau,phi_func,phi_derivative)
w = -log(1/tau - 1);
d = abs(X * theta - w) / norm(theta);
N = length(A);

C = (1./(1+exp(-X * theta)) >= tau);
phi = phi_func((A==1 & Y==1),(A==0 & Y==1));
phi_deri = phi_derivative((A==1 & Y==1),(A==0 & Y==1));
exp_first = mean(phi_deri .*C)';
RV =  [(A==1 & Y==1),(A==0 & Y==1)] * exp_first + phi .* C;
var = cov(RV);
h = N^(-0.2) ;
Kernel = @(x,h) 1/sqrt(2 * pi) * exp(-(x./h).^2/2) ./ h;
C2 = Kernel( d .* (2 * C - 1) , h) .* phi;
C2 = C2' * phi / N;
var = 1/2/C2 * var;