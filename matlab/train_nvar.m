function [W_out, x_predict] = train_nvar(x, warmup_pts, traintime_pts, ridge_param, d)
%TRAIN_NVAR Train the output weights using ridge regression for NG-RC.
%   x is dlin x N containing delayed taps. d defaults to 3.
if nargin < 5
    d = 3;
end
train_inputs = quadratic_features(x(:, warmup_pts:(warmup_pts + traintime_pts - 1)));
target = x(1:d, (warmup_pts + 1):(warmup_pts + traintime_pts));
prev = x(1:d, warmup_pts:(warmup_pts + traintime_pts - 1));
reg = ridge_param * eye(size(train_inputs, 1));
W_out = (target - prev) * train_inputs.' / (train_inputs * train_inputs.' + reg);
x_predict = prev + W_out * train_inputs;
end
