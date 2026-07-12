function plot_lqr_response(t, x, u, cfg)
%PLOT_LQR_RESPONSE Plot state and control histories.

figure('Name', 'LQR State Response', 'Color', 'w', 'Position', [120, 120, 980, 680]);

subplot(3, 1, 1);
plot(t, x(:, 1), 'LineWidth', 1.8);
grid on;
ylabel('x / m');
title('Position response');

subplot(3, 1, 2);
plot(t, rad2deg(x(:, 3)), 'LineWidth', 1.8);
grid on;
ylabel('\theta / deg');
title('Body pitch response');

subplot(3, 1, 3);
plot(t, u, 'LineWidth', 1.8);
hold on;
yline(cfg.ctrl.u_max, 'r--', 'LineWidth', 1.0);
yline(cfg.ctrl.u_min, 'r--', 'LineWidth', 1.0);
grid on;
xlabel('time / s');
ylabel('u / N');
title('Wheel-ground force command');
end

