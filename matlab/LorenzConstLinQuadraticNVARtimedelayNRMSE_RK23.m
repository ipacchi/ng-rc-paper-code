function LorenzConstLinQuadraticNVARtimedelayNRMSE_RK23()
% Compute NRMSE for Lorenz NG-RC prediction over one Lyapunov time.
opts = struct('testtime', 120, 'plottime', 120);
res = run_lorenz_experiment(opts);
fprintf('training nrmse: %g\n', res.train_nrmse);
fprintf('test nrmse: %g\n', res.test_nrmse);
figure('Name','Lorenz NRMSE');
plot(res.t_eval(res.warmtrain_pts:(res.warmtrain_pts + res.plottime_pts -1)) - res.t_eval(res.warmtrain_pts), res.x_lin(1,res.warmtrain_pts:(res.warmtrain_pts+res.plottime_pts-1)), 'LineWidth',1.1);
hold on;
plot(res.t_eval(res.warmtrain_pts:(res.warmtrain_pts + res.plottime_pts -1)) - res.t_eval(res.warmtrain_pts), res.x_test(1,1:res.plottime_pts), 'r', 'LineWidth',1.1);
xlabel('time'); ylabel('x'); legend('ground truth','prediction');
end
