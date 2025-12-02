function results = run_lorenz_experiment(opts)
%RUN_LORENZ_EXPERIMENT Generic NG-RC pipeline for Lorenz system.
arguments
    opts.dt (1,1) double = 0.025
    opts.warmup (1,1) double = 5
    opts.traintime (1,1) double = 10
    opts.testtime (1,1) double = 20
    opts.plottime (1,1) double = 20
    opts.lyaptime (1,1) double = 1.104
    opts.ridge (1,1) double = 2.5e-6
    opts.k (1,1) double = 2
    opts.d (1,1) double = 3
    opts.degree (1,1) double = 2
    opts.include_constant (1,1) logical = true
    opts.initial (3,1) double = [17.67715816276679; 12.931379185960404; 43.91404334248268]
    opts.noise_std (1,1) double = 0
end

warmup_pts = round(opts.warmup / opts.dt);
traintime_pts = round(opts.traintime / opts.dt);
warmtrain_pts = warmup_pts + traintime_pts;
testtime_pts = round(opts.testtime / opts.dt);
plottime_pts = round(opts.plottime / opts.dt);
lyaptime_pts = round(opts.lyaptime / opts.dt);
maxtime = opts.warmup + opts.traintime + opts.testtime;

[t_eval, states] = integrate_lorenz(maxtime, opts.dt, opts.initial);
if opts.noise_std > 0
    states = states + randn(size(states)) * opts.noise_std;
end

x_lin = build_delay_features(states, opts.k);

% Training
train_inputs = build_polynomial_features(x_lin(:, warmup_pts:(warmtrain_pts - 1)), opts.degree, opts.include_constant);
prev = x_lin(1:opts.d, warmup_pts:(warmtrain_pts - 1));
target = x_lin(1:opts.d, (warmup_pts + 1):warmtrain_pts);
reg = opts.ridge * eye(size(train_inputs, 1));
W_out = (target - prev) * train_inputs.' / (train_inputs * train_inputs.' + reg);
x_predict = prev + W_out * train_inputs;

% Prediction rollout
x_test = zeros(size(x_lin, 1), testtime_pts);
x_test(:, 1) = x_lin(:, warmtrain_pts);
out_vec = zeros(size(train_inputs, 1), 1);
for j = 1:(testtime_pts - 1)
    out_vec(1) = opts.include_constant;
    out_vec(2:(opts.k * opts.d + 1)) = x_test(:, j);
    row = opts.k * opts.d + 2;
    for r = 1:(opts.k * opts.d)
        for c = r:(opts.k * opts.d)
            if opts.degree == 2
                out_vec(row) = x_test(r, j) * x_test(c, j);
                row = row + 1;
            else
                for s = c:(opts.k * opts.d)
                    out_vec(row) = x_test(r, j) * x_test(c, j) * x_test(s, j);
                    row = row + 1;
                end
            end
        end
    end
    x_test((opts.d + 1):(opts.k * opts.d), j + 1) = x_test(1:((opts.k - 1) * opts.d), j);
    x_test(1:opts.d, j + 1) = x_test(1:opts.d, j) + W_out * out_vec;
end

% Metrics
total_var = var(states(1, :)) + var(states(2, :)) + var(states(3, :));
train_nrmse = sqrt(mean((x_lin(1:opts.d, warmup_pts:(warmtrain_pts - 1)) - x_predict).^2, 'all') / total_var);
test_nrmse = sqrt(mean((x_lin(1:opts.d, warmtrain_pts:(warmtrain_pts + lyaptime_pts - 1)) - x_test(1:opts.d, 1:lyaptime_pts)).^2, 'all') / total_var);

results.W_out = W_out;
results.x_predict = x_predict;
results.x_lin = x_lin;
results.x_test = x_test;
results.t_eval = t_eval;
results.train_nrmse = train_nrmse;
results.test_nrmse = test_nrmse;
results.warmup_pts = warmup_pts;
results.traintime_pts = traintime_pts;
results.warmtrain_pts = warmtrain_pts;
results.plottime_pts = plottime_pts;
results.lyaptime_pts = lyaptime_pts;
results.total_var = total_var;
end
