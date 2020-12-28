function [val] = discontinuous_RWPI(X,A,Y,theta,N,tau,phi_func)
w = -log(1/tau - 1);
d = abs(X * theta - w) / norm(theta);
C = (1./(1+exp(-X * theta)) >= tau);
phi = phi_func((A==1 & Y==1),(A==0 & Y==1));
s = - sum(C.*phi );
t = [sign(s) ./d .* (1 - 2* C) .* phi,d];
t = sortrows(t,'descend');
val = 0;
s = abs(s);
for i = 1:N
    if t(i,1)*t(i,2) <= s
        s = s - t(i,1)*t(i,2);
        val = val + t(i,2);
    else
        val = val + s/t(i,1);
        break;
    end
end

        
