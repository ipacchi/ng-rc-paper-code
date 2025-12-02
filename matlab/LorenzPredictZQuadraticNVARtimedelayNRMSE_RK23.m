function LorenzPredictZQuadraticNVARtimedelayNRMSE_RK23()
% Report NRMSE for z-component forecast.
opts = struct('testtime', 120, 'plottime', 120);
res = run_lorenz_experiment(opts);
fprintf('overall test nrmse: %g\n', res.test_nrmse);
end
