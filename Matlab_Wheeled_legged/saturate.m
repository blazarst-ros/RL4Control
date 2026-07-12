function y = saturate(x, xmin, xmax)
%SATURATE Clamp a scalar value between lower and upper bounds.
y = min(max(x, xmin), xmax);
end

