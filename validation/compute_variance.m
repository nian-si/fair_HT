function [var] = compute_variance(p11 , p01 , p10, p00)

p_win = 0.5;
var = ( (1/p01/p11))^2 * (p_win * (p01^2*p11 + p11^2 * p01 ) + p01 * (p_win * p11)^2 +  p11 * (p_win * p01)^2 -2* p_win^2 *p11^2 *p01 -  2*p_win^2 *p11 *p01^2)
theta = [0,1]';

Sigma01 = [5 0;0 5];
Sigma11 = [3.5 0;0 5];
mu01 = [-2;0];
mu11 = [6;0];
w = 0;
x01 = 1/sqrt(theta' * Sigma01 * theta) * (-theta'*mu01);
x11 = 1/sqrt(theta' * Sigma11 * theta) * (-theta'*mu11);
C =   1/sqrt(theta' * Sigma01 * theta) * normpdf(x01)/p01 + 1/sqrt(theta' * Sigma11 * theta) * normpdf(x11)/p11
var = 1/2/C * var;