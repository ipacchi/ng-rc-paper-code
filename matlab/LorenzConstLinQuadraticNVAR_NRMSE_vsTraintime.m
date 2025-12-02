function LorenzConstLinQuadraticNVAR_NRMSE_vsTraintime()
% Sweep training duration and report NRMSE statistics.
traintimes = linspace(4, 24, 21);
warmup = 5.0;
dt = 0.025;
means = zeros(size(traintimes));
for i = 1:numel(traintimes)
    opts = struct('warmup', warmup, 'traintime', traintimes(i), 'testtime', 1.104);
    res = run_lorenz_experiment(opts);
    means(i) = res.test_nrmse;
    fprintf('traintime %.3f -> test nrmse %.5f\n', traintimes(i), res.test_nrmse);
end
figure('Name','NRMSE vs train time');
plot(traintimes / dt, means, 'o-');
xlabel('training data set size'); ylabel('NRMSE');
end
