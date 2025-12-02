function LorenzCrossPrediction_RK23()
% Demonstrate zero-lag cross prediction between two Lorenz trajectories.
opts = struct('testtime', 60);
res = run_lorenz_cross_prediction(opts);
fprintf('cross prediction NRMSE: %g\n', res.test_nrmse);

figure('Name','Lorenz cross prediction');
plot(res.t_eval(res.warmtrain_pts:(res.warmtrain_pts + res.testtime_pts - 1)) - res.t_eval(res.warmtrain_pts), ...
     res.target_delay(1, res.warmtrain_pts:(res.warmtrain_pts + res.testtime_pts - 1)), 'LineWidth', 1.1);
hold on;
plot(res.t_eval(res.warmtrain_pts:(res.warmtrain_pts + res.testtime_pts - 1)) - res.t_eval(res.warmtrain_pts), ...
     res.test_pred(1, :), 'r', 'LineWidth', 1.1);
xlabel('time'); ylabel('target x'); legend('ground truth','cross prediction');
end
