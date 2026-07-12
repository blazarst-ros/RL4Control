function cfg = wheel_leg_lqr_config()
%WHEEL_LEG_LQR_CONFIG Parameters for the wheel-legged LQR demo.

cfg.robot.m_wheel = 0.45;       % kg
cfg.robot.m_body = 3.20;        % kg
cfg.robot.body_width = 0.78;    % m
cfg.robot.body_height = 0.22;   % m
cfg.robot.wheel_radius = 0.115; % m
cfg.robot.com_height = 0.48;    % m, equivalent inverted-pendulum length
cfg.robot.gravity = 9.81;       % m/s^2

% Five-bar visual geometry. These values are used for display only.
cfg.leg.hip_spacing = 0.46;     % m
cfg.leg.hip_drop = 0.035;       % m
cfg.leg.upper = 0.30;           % m
cfg.leg.lower = 0.30;           % m
cfg.leg.knee_bend = 0.075;      % m

% A first-order force actuator is approximated by saturation in simulation.
cfg.ctrl.u_min = -28.0;         % N
cfg.ctrl.u_max = 28.0;          % N

cfg.sim.dt = 0.02;              % s
cfg.sim.t_final = 8.0;          % s
cfg.sim.playback_stride = 2;

cfg.viz.x_window = 2.2;         % m
cfg.viz.ground_y = 0.0;
cfg.viz.save_animation = false;
cfg.viz.gif_name = 'wheel_leg_lqr_demo.gif';
end

