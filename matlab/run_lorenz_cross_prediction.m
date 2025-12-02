function results = run_lorenz_cross_prediction(opts)
%RUN_LORENZ_CROSS_PREDICTION Zero-lag NG-RC mapping between Lorenz trajectories.
%   This routine trains a mapping from a "source" Lorenz trajectory to a
%   separate "target" trajectory using delay-embedded source signals and
%   polynomial features (cross prediction). No autoregressive rollout is
%   performed: test predictions use the measured source signals only.
arguments
    opts.dt (1,1) double = 0.025
    opts.warmup (1,1) double = 5
    opts.traintime (1,1) double = 10
    opts.testtime (1,1) double = 20
    opts.lyaptime (1,1) double = 1.104
    opts.ridge (1,1) double = 2.5e-6
    opts.k (1,1) double = 2
    opts.degree (1,1) double = 2
    opts.include_constant (1,1) logical = true
    opts.d_source (1,1) double = 3
    opts.d_target (1,1) double = 3
    opts.source_initial (3,1) double = [17.67715816276679; 12.931379185960404; 43.91404334248268]
    opts.target_initial (3,1) double = [16.0; 12.5; 45.0]
    opts.noise_std (1,1) double = 0
end

warmup_pts = round(opts.warmup / opts.dt);
traintime_pts = round(opts.traintime / opts.dt);
warmtrain_pts = warmup_pts + traintime_pts;
testtime_pts = round(opts.testtime / opts.dt);
lyaptime_pts = round(opts.lyaptime / opts.dt);
maxtime = opts.warmup + opts.traintime + opts.testtime;

[t_eval, source_states] = integrate_lorenz(maxtime, opts.dt, opts.source_initial);
[~, target_states] = integrate_lorenz(maxtime, opts.dt, opts.target_initial);

if opts.noise_std > 0
    source_states = source_states + randn(size(source_states)) * opts.noise_std;
    target_states = target_states + randn(size(target_states)) * opts.noise_std;
end

source_delay = build_delay_features(source_states(1:opts.d_source, :), opts.k);
target_delay = build_delay_features(target_states(1:opts.d_target, :), opts.k);

train_features = source_delay(:, warmup_pts:(warmtrain_pts - 1));
train_targets = target_delay(1:opts.d_target, warmup_pts:(warmtrain_pts - 1));
[W_out, train_pred, train_inputs] = train_cross_nvar(train_features, train_targets, opts.ridge, opts.degree, opts.include_constant);

% Test using measured source signals (zero-lag cross prediction)
test_features = source_delay(:, warmtrain_pts:(warmtrain_pts + testtime_pts - 1));
test_inputs = build_polynomial_features(test_features, opts.degree, opts.include_constant);
test_pred = W_out * test_inputs;

total_var = sum(var(target_delay(1:opts.d_target, :), 0, 2));
train_nrmse = sqrt(mean((train_targets - train_pred).^2, 'all') / total_var);
lyap_end = warmtrain_pts + lyaptime_pts - 1;
lyap_pred = W_out * build_polynomial_features(source_delay(:, warmtrain_pts:lyap_end), opts.degree, opts.include_constant);
test_nrmse = sqrt(mean((target_delay(1:opts.d_target, warmtrain_pts:lyap_end) - lyap_pred).^2, 'all') / total_var);

results.W_out = W_out;
results.train_pred = train_pred;
results.test_pred = test_pred;
results.source_delay = source_delay;
results.target_delay = target_delay;
results.t_eval = t_eval;
results.train_inputs = train_inputs;
results.test_inputs = test_inputs;
results.train_nrmse = train_nrmse;
results.test_nrmse = test_nrmse;
results.warmup_pts = warmup_pts;
results.traintime_pts = traintime_pts;
results.warmtrain_pts = warmtrain_pts;
results.lyaptime_pts = lyaptime_pts;
results.testtime_pts = testtime_pts;
results.total_var = total_var;
end
