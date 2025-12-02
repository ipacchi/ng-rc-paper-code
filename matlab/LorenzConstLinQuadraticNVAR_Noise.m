function LorenzConstLinQuadraticNVAR_Noise()
%LORENZCONSTLINQUADRATICNVAR_NOISE MATLAB translation including additive noise.
opts = struct('noise_std', 0.05);
res = run_lorenz_experiment(opts);
fprintf('training nrmse: %g\n', res.train_nrmse);
fprintf('test nrmse: %g\n', res.test_nrmse);
figure('Name', 'Lorenz with noise');
plot(res.t_eval(1:res.warmtrain_pts), res.x_lin(1,1:res.warmtrain_pts)); hold on;
plot(res.t_eval(res.warmtrain_pts:(res.warmtrain_pts + res.plottime_pts -1)) - res.t_eval(res.warmtrain_pts), res.x_test(1,1:res.plottime_pts),'r');
xlabel('time'); ylabel('x'); legend('train','prediction');
end
