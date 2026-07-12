function pts = five_bar_leg_points(body_center, theta, side_sign, cfg)
%FIVE_BAR_LEG_POINTS Return display points for one parallel five-bar leg.
%
% The mechanism is drawn as two symmetric two-link chains from two hip
% pivots to one wheel axle. It is a kinematic visualization coupled to the
% body pose, not a constraint solver for leg dynamics.

R = [cos(theta), -sin(theta); sin(theta), cos(theta)];

half_hip = cfg.leg.hip_spacing / 2;
hip_y = -cfg.robot.body_height / 2 - cfg.leg.hip_drop;

hip_a_local = [-half_hip; hip_y];
hip_b_local = [ half_hip; hip_y];

wheel_x_local = 0.0;
wheel_y_local = -cfg.robot.com_height + cfg.robot.wheel_radius;
wheel_local = [wheel_x_local; wheel_y_local];

hip_a = body_center(:) + R * hip_a_local;
hip_b = body_center(:) + R * hip_b_local;
wheel = body_center(:) + R * wheel_local;

mid_a = 0.5 * (hip_a + wheel) + side_sign * cfg.leg.knee_bend * [cos(theta); sin(theta)];
mid_b = 0.5 * (hip_b + wheel) - side_sign * cfg.leg.knee_bend * [cos(theta); sin(theta)];

pts.hip_a = hip_a;
pts.hip_b = hip_b;
pts.knee_a = mid_a;
pts.knee_b = mid_b;
pts.wheel = wheel;
end

