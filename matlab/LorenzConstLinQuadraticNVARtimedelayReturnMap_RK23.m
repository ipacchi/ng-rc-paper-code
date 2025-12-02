function LorenzConstLinQuadraticNVARtimedelayReturnMap_RK23()
% Plot return map for Lorenz prediction.
opts = struct('testtime', 60);
res = run_lorenz_experiment(opts);
fprintf('training nrmse: %g\n', res.train_nrmse);
fprintf('test nrmse: %g\n', res.test_nrmse);
figure('Name','Return map');
truth = res.x_lin(3, res.warmtrain_pts:(res.warmtrain_pts + res.plottime_pts - 1));
pred = res.x_test(3, 1:res.plottime_pts);
plot(truth(1:end-1), truth(2:end), 'k.', 'DisplayName','truth'); hold on;
plot(pred(1:end-1), pred(2:end), 'r.', 'DisplayName','prediction');
xlabel('z_t'); ylabel('z_{t+1}'); legend show;
end
