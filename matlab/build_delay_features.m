function x = build_delay_features(states, k)
%BUILD_DELAY_FEATURES Construct delayed linear feature taps.
%   states should be d x N. Returns x with size (k*d) x N.
[d, n] = size(states);
x = zeros(k * d, n);
for delay = 0:(k - 1)
    idx = (delay * d + 1):((delay + 1) * d);
    src = states(:, 1:(n - delay));
    x(idx, (delay + 1):n) = src;
    if delay > 0
        x(idx, 1:delay) = repmat(states(:, 1), 1, delay);
    end
end
end
