function h = draw_rotated_rectangle(center, width, height, angle, face_color, edge_color)
%DRAW_ROTATED_RECTANGLE Draw a filled rectangle with arbitrary rotation.

local = 0.5 * [-width, -height;
                width, -height;
                width,  height;
               -width,  height].';
R = [cos(angle), -sin(angle); sin(angle), cos(angle)];
xy = center(:) + R * local;

h = patch(xy(1, :), xy(2, :), face_color, ...
    'EdgeColor', edge_color, ...
    'LineWidth', 1.5);
end

