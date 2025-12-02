function LorenzPredictZQuadraticNVARtimedelayWout_RK23()
% Train NG-RC and export output weights.
res = run_lorenz_experiment();
fprintf('training nrmse: %g\n', res.train_nrmse);
fprintf('First few W_out coefficients (row-major):\n');
disp(res.W_out(:,1:min(5, size(res.W_out,2))));
end
