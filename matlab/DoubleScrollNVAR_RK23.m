function DoubleScrollNVAR_RK23()
%DOUBLESCROLLNVAR_RK23 MATLAB translation of DoubleScrollNVAR-RK23.py.
res = run_doublescroll_experiment();
fprintf('training nrmse: %g\n', res.train_nrmse);
fprintf('test nrmse: %g\n', res.test_nrmse);

figure('Name', 'Double-scroll NG-RC');
subplot(1,2,1);
plot(res.x_lin(1, res.warmtrain_pts:end), res.x_lin(2, res.warmtrain_pts:end), 'LineWidth', 0.4);
title('ground truth'); xlabel('V1'); ylabel('V2');
subplot(1,2,2);
plot(res.x_test(1, :), res.x_test(2, :), 'r', 'LineWidth', 0.4);
title('prediction'); xlabel('V1'); ylabel('V2');
end
