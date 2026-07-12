function dx = wheel_leg_dynamics(~, x, K, cfg)
%WHEEL_LEG_DYNAMICS Nonlinear cart-pendulum dynamics with LQR feedback.

M = cfg.robot.m_body + 2 * cfg.robot.m_wheel;
m = cfg.robot.m_body;
l = cfg.robot.com_height;
g = cfg.robot.gravity;

pos = x(1);
vel = x(2);
theta = x(3);
omega = x(4);

u = saturate(-K * x, cfg.ctrl.u_min, cfg.ctrl.u_max);

s = sin(theta);
c = cos(theta);
den = M + m * s^2;

pos_dot = vel;
vel_dot = (u - m * s * (l * omega^2 + g * c)) / den;
theta_dot = omega;
omega_dot = ((M + m) * g * s - c * (u + m * l * omega^2 * s)) / (l * den);

% Weak viscous damping avoids unrealistic long high-frequency oscillation in
% the nonlinear visualization while preserving the LQR response shape.
vel_dot = vel_dot - 0.04 * vel;
omega_dot = omega_dot - 0.03 * omega;

dx = [pos_dot; vel_dot; theta_dot; omega_dot];

% Keep position variable referenced for clarity in state documentation.
if ~isfinite(pos)
    error('State contains a non-finite cart position.');
end
end

