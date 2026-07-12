function [A, B] = linearized_wheel_leg_model(cfg)
%LINEARIZED_WHEEL_LEG_MODEL Equivalent linear model around upright posture.
%
% State:
%   x = [cart_position; cart_velocity; body_pitch; body_pitch_rate]
% Input:
%   u = horizontal wheel-ground force.
%
% This is the standard cart-pendulum small-angle model, used here as a
% course-level equivalent model for a wheel-legged vehicle with a regulated
% leg configuration.

M = cfg.robot.m_body + 2 * cfg.robot.m_wheel;
m = cfg.robot.m_body;
l = cfg.robot.com_height;
g = cfg.robot.gravity;

den = M;

A = [0, 1,        0, 0;
     0, 0,   -(m*g)/den, 0;
     0, 0,        0, 1;
     0, 0, g*(M + m)/(l*den), 0];

B = [0;
     1/den;
     0;
     -1/(l*den)];
end

