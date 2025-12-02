function out = build_polynomial_features(x, degree, include_constant)
%BUILD_POLYNOMIAL_FEATURES Create polynomial features up to given degree.
%   x: dlin x T. degree: 1,2,3 supported. include_constant default true.
if nargin < 3
    include_constant = true;
end
[dlin, t] = size(x);
if degree == 1
    out = [ones(1, t) * include_constant; x];
    return;
end
dnonlin = 0;
if degree == 2
    dnonlin = dlin * (dlin + 1) / 2;
elseif degree == 3
    dnonlin = dlin * (dlin + 1) * (dlin + 2) / 6;
else
    error('Only degree 1-3 supported');
end
const_block = ones(1, t) * include_constant;
lin_block = x;
out = [const_block; lin_block; zeros(dnonlin, t)];
row = 1 + dlin + 1; % start after constant and linear
if degree == 2
    row = 2 + dlin;
    for i = 1:dlin
        for j = i:dlin
            out(row, :) = x(i, :) .* x(j, :);
            row = row + 1;
        end
    end
else
    for i = 1:dlin
        for j = i:dlin
            for k = j:dlin
                out(row, :) = x(i, :) .* x(j, :) .* x(k, :);
                row = row + 1;
            end
        end
    end
end
end
