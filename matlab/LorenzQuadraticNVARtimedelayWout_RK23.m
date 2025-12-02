function LorenzQuadraticNVARtimedelayWout_RK23()
% Train NG-RC without constant bias term and show W_out.
opts = struct('include_constant', false);
res = run_lorenz_experiment(opts);
fprintf('training nrmse: %g\n', res.train_nrmse);
fprintf('First few W_out coefficients (row-major):\n');
disp(res.W_out(:,1:min(5, size(res.W_out,2))));
end
