function [val] = discontinuous_RWPI_cvx(X,A,Y,theta,N,tau,phi_func)
w = -log(1/tau - 1);
d = abs(X * theta - w) / norm(theta);
C = (1./(1+exp(-X * theta)) >= tau);
phi = phi_func((A==1 & Y==1),(A==0 & Y==1));
s = - sum(C.*phi );

cvx_begin
    variable p(N)
    minimize( p'*d )
    subject to
        sum((1 - 2 * C).* phi.*p) == s
        0 <= p <= 1
cvx_end
