function animate_wheel_leg_lqr(t, x, u, K, cfg)
%ANIMATE_WHEEL_LEG_LQR Animate the wheel-legged robot response.

fig = figure('Name', 'Parallel Five-Bar Wheel-Legged Cart LQR', ...
    'Color', 'w', ...
    'Position', [80, 80, 1100, 620]);

axis equal;
grid on;
hold on;
xlabel('x / m');
ylabel('y / m');
title('LQR stabilization of an equivalent wheel-legged inverted pendulum');

ground_y = cfg.viz.ground_y;
body_y = ground_y + cfg.robot.wheel_radius + cfg.robot.com_height;

for i = 1:cfg.sim.playback_stride:numel(t)
    cla;

    pos = x(i, 1);
    theta = x(i, 3);
    body_center = [pos; body_y];

    xlim([pos - cfg.viz.x_window/2, pos + cfg.viz.x_window/2]);
    ylim([-0.08, body_y + 0.55]);

    plot([pos - 10, pos + 10], [ground_y, ground_y], 'k-', 'LineWidth', 2);
    draw_rotated_rectangle(body_center, cfg.robot.body_width, ...
        cfg.robot.body_height, theta, [0.80, 0.90, 1.00], [0.10, 0.25, 0.45]);

    left_leg = five_bar_leg_points(body_center, theta, 1, cfg);
    right_leg = five_bar_leg_points(body_center, theta, -1, cfg);
    draw_leg(left_leg, [0.10, 0.35, 0.80]);
    draw_leg(right_leg, [0.75, 0.22, 0.18]);

    wheel_center = [pos; ground_y + cfg.robot.wheel_radius];
    draw_circle(wheel_center, cfg.robot.wheel_radius, [0.08, 0.08, 0.08]);
    plot(wheel_center(1), wheel_center(2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);

    com = body_center + [cfg.robot.com_height * sin(theta); ...
                         cfg.robot.com_height * cos(theta)] * 0.24;
    plot([body_center(1), com(1)], [body_center(2), com(2)], ...
        '-', 'Color', [0.20, 0.20, 0.20], 'LineWidth', 2);
    plot(com(1), com(2), 'o', 'MarkerFaceColor', [0.95, 0.55, 0.10], ...
        'MarkerEdgeColor', 'none', 'MarkerSize', 8);

    info = sprintf(['t = %.2f s    x = %.3f m    pitch = %.2f deg\n', ...
                    'u = %.2f N    K = [%.2f %.2f %.2f %.2f]'], ...
                    t(i), pos, rad2deg(theta), u(i), K);
    text(pos - cfg.viz.x_window/2 + 0.05, body_y + 0.43, info, ...
        'FontName', 'Consolas', 'FontSize', 10, ...
        'BackgroundColor', [1, 1, 1], 'Margin', 6);

    drawnow;

    if cfg.viz.save_animation
        frame = getframe(fig);
        [img, map] = rgb2ind(frame2im(frame), 256);
        if i == 1
            imwrite(img, map, cfg.viz.gif_name, 'gif', 'LoopCount', inf, 'DelayTime', cfg.sim.dt);
        else
            imwrite(img, map, cfg.viz.gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', cfg.sim.dt);
        end
    end
end
end

function draw_circle(center, radius, color)
ang = linspace(0, 2*pi, 80);
plot(center(1) + radius * cos(ang), center(2) + radius * sin(ang), ...
     '-', 'Color', color, 'LineWidth', 1.5);
end

function draw_leg(leg, color)
plot([leg.hip_a(1), leg.knee_a(1), leg.wheel(1)], ...
     [leg.hip_a(2), leg.knee_a(2), leg.wheel(2)], ...
     '-', 'Color', color, 'LineWidth', 3);
plot([leg.hip_b(1), leg.knee_b(1), leg.wheel(1)], ...
     [leg.hip_b(2), leg.knee_b(2), leg.wheel(2)], ...
     '-', 'Color', color * 0.75, 'LineWidth', 3);
plot([leg.hip_a(1), leg.hip_b(1)], [leg.hip_a(2), leg.hip_b(2)], ...
     '-', 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.2);
plot([leg.hip_a(1), leg.hip_b(1), leg.knee_a(1), leg.knee_b(1), leg.wheel(1)], ...
     [leg.hip_a(2), leg.hip_b(2), leg.knee_a(2), leg.knee_b(2), leg.wheel(2)], ...
     'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', color, 'MarkerSize', 5);
end
