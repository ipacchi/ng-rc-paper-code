function LorenzPredictZQuadraticNVARtimedelay_RK23()
% Predict Lorenz z-component using quadratic NG-RC.
opts = struct('testtime', 60);
res = run_lorenz_experiment(opts);
fprintf('z prediction NRMSE: %g\n', res.test_nrmse);
figure('Name','Lorenz z prediction');
plot(res.t_eval(res.warmtrain_pts:(res.warmtrain_pts + res.plottime_pts -1)) - res.t_eval(res.warmtrain_pts), res.x_lin(3,res.warmtrain_pts:(res.warmtrain_pts+res.plottime_pts-1)),'LineWidth',1.1);
hold on;
plot(res.t_eval(res.warmtrain_pts:(res.warmtrain_pts + res.plottime_pts -1)) - res.t_eval(res.warmtrain_pts), res.x_test(3,1:res.plottime_pts),'r','LineWidth',1.1);
xlabel('time'); ylabel('z'); legend('ground truth','prediction');
end
