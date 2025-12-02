function out = quadratic_features(x)
%QUADRATIC_FEATURES Generate constant+linear+quadratic feature vector.
%   x is (dlin x T). Returns dtot x T where dtot = 1 + dlin + dlin*(dlin+1)/2.
[dlin, t] = size(x);
dnonlin = dlin * (dlin + 1) / 2;
out = ones(1 + dlin + dnonlin, t);
out(2:(dlin + 1), :) = x;
row = dlin + 2;
for i = 1:dlin
    for j = i:dlin
        out(row, :) = x(i, :) .* x(j, :);
        row = row + 1;
    end
end
end
