function [t_eval, states] = integrate_lorenz(maxtime, dt, initial_cond, params)
%INTEGRATE_LORENZ Integrate the Lorenz '63 system with fixed step outputs.
%   [t_eval, states] = INTEGRATE_LORENZ(maxtime, dt, initial_cond, params)
%   uses ode45 to integrate from t=0 to t=maxtime with samples spaced by
%   dt. "states" is returned as a d x N matrix where d=3.

if nargin < 4
    params.sigma = 10;
    params.beta = 8/3;
    params.rho = 28;
end

if nargin < 3 || isempty(initial_cond)
    initial_cond = [17.67715816276679; 12.931379185960404; 43.91404334248268];
end

t_eval = 0:dt:maxtime;
lorenz_ode = @(t, y) lorenz_rhs(t, y, params);
opts = odeset('RelTol', 1e-9, 'AbsTol', 1e-9*ones(1, numel(initial_cond)));
[~, y] = ode45(lorenz_ode, t_eval, initial_cond(:), opts);
states = y.';
end

function dydt = lorenz_rhs(~, y, params)
%LORENZ_RHS Right-hand side for Lorenz '63.
dydt = [params.sigma * (y(2) - y(1));
        y(1) * (params.rho - y(3)) - y(2);
        y(1) * y(2) - params.beta * y(3)];
end
