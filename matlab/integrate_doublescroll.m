function [t_eval, states] = integrate_doublescroll(maxtime, dt, initial_cond)
%INTEGRATE_DOUBLESCROLL Integrate double-scroll circuit equations.
if nargin < 3 || isempty(initial_cond)
    initial_cond = [0.37926545; 0.058339; -0.08167691];
end
r1 = 1.2;
r2 = 3.44;
r4 = 0.193;
alpha = 11.6;
ir = 2 * 2.25e-5; % factor 2 for sinh symmetry

scroll_rhs = @(t, y) doublescroll_rhs(y, r1, r2, r4, alpha, ir);
t_eval = 0:dt:maxtime;
opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-9*ones(1, numel(initial_cond)));
[~, y] = ode45(scroll_rhs, t_eval, initial_cond(:), opts);
states = y.';
end

function dydt = doublescroll_rhs(y, r1, r2, r4, alpha, ir)
%DOUBLESCROLL_RHS Right-hand side for the double-scroll oscillator.
dV = y(1) - y(2);
g = (dV / r2) + ir * sinh(alpha * dV);
dy0 = (y(1) / r1) - g;
dy1 = g - y(3);
dy2 = y(2) - r4 * y(3);
dydt = [dy0; dy1; dy2];
end
