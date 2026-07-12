%% Wheel-Legged Parallel Five-Bar Cart LQR Demo
% This script simulates an equivalent wheel-legged inverted-pendulum model
% and visualizes a parallel five-bar leg mechanism attached to the body.
%
% Run from MATLAB:
%   cd Matlab_LQR
%   run_wheel_leg_lqr_demo

clear; clc; close all;

cfg = wheel_leg_lqr_config();

[A, B] = linearized_wheel_leg_model(cfg);
Q = diag([18, 2.5, 260, 18]);
R = 0.35;
K = lqr(A, B, Q, R);

fprintf('LQR gain K = [%.4f %.4f %.4f %.4f]\n', K);

x0 = [-0.45; 0.00; deg2rad(9.0); 0.00];
t_span = 0:cfg.sim.dt:cfg.sim.t_final;

ode = @(t, x) wheel_leg_dynamics(t, x, K, cfg);
opts = odeset('RelTol', 1e-7, 'AbsTol', 1e-8);
[t, x] = ode45(ode, t_span, x0, opts);

u = zeros(size(t));
for i = 1:numel(t)
    u(i) = saturate(-K * x(i, :).', cfg.ctrl.u_min, cfg.ctrl.u_max);
end

animate_wheel_leg_lqr(t, x, u, K, cfg);
plot_lqr_response(t, x, u, cfg);

