function [x_test, out_test] = rollout_nvar(W_out, x_initial, steps, d, include_constant)
%ROLLOUT_NVAR Predict future states using learned W_out.
if nargin < 4
    d = 3;
end

if nargin < 5
    include_constant = true;
end

dlin = numel(x_initial);
x_test = zeros(dlin, steps);
out_test = zeros(1 + dlin + dlin * (dlin + 1) / 2, 1);
x_test(:, 1) = x_initial;
for j = 1:(steps - 1)

    out_test(1) = include_constant;

    out_test(2:(dlin + 1)) = x_test(:, j);
    row = dlin + 2;
    for r = 1:dlin
        for c = r:dlin
            out_test(row) = x_test(r, j) * x_test(c, j);
            row = row + 1;
        end
    end
    x_test((d + 1):dlin, j + 1) = x_test(1:(dlin - d), j);
    x_test(1:d, j + 1) = x_test(1:d, j) + W_out * out_test;
end
end
