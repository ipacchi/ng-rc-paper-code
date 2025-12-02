function [W_out, predictions, train_inputs] = train_cross_nvar(source_delay, target_signals, ridge_param, degree, include_constant)
%TRAIN_CROSS_NVAR Train zero-lag (cross) NG-RC mapping from source to target.
%   source_delay: k*d_source x N delay-embedded source signals
%   target_signals: d_target x N target signals aligned to source (same time index)
%   ridge_param: scalar ridge penalty
%   degree: polynomial degree (2 or 3)
%   include_constant: logical flag to prepend constant feature

if nargin < 3
    error('train_cross_nvar requires source_delay, target_signals, and ridge_param.');
end
if nargin < 4
    degree = 2;
end
if nargin < 5
    include_constant = true;
end

train_inputs = build_polynomial_features(source_delay, degree, include_constant);
reg = ridge_param * eye(size(train_inputs, 1));
W_out = target_signals * train_inputs.' / (train_inputs * train_inputs.' + reg);
predictions = W_out * train_inputs;
end
