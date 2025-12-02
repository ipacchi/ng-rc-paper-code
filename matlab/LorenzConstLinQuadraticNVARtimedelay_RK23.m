function LorenzConstLinQuadraticNVARtimedelay_RK23()
%LORENZCONSTLINQUADRATICNVAR... MATLAB translation of Python script.
% Forecast Lorenz system with NG-RC using constant, linear, and quadratic
% features with two delay taps.

%% Parameters
dt = 0.025;
warmup = 5.0;
traintime = 10.0;
testtime = 120.0;
plottime = 20.0;
lyaptime = 1.104;
ridge_param = 2.5e-6;

warmup_pts = round(warmup / dt);
traintime_pts = round(traintime / dt);
warmtrain_pts = warmup_pts + traintime_pts;
testtime_pts = round(testtime / dt);
maxtime_pts = round((warmup + traintime + testtime) / dt);
plottime_pts = round(plottime / dt);
lyaptime_pts = round(lyaptime / dt);

k = 2; d = 3; dlin = k * d;

%% Lorenz simulation
maxtime = warmup + traintime + testtime;
[t_eval, states] = integrate_lorenz(maxtime, dt);

% Feature construction
delayed = build_delay_features(states, k);

%% Training
[W_out, x_predict] = train_nvar(delayed, warmup_pts, traintime_pts, ridge_param, d);

total_var = var(states(1, :)) + var(states(2, :)) + var(states(3, :));
train_err = sqrt(mean((delayed(1:d, warmup_pts:warmtrain_pts - 1) - x_predict).^2, 'all') / total_var);
fprintf('training nrmse: %g\n', train_err);

%% Prediction
[x_test, ~] = rollout_nvar(W_out, delayed(:, warmtrain_pts), testtime_pts, d);

test_err = sqrt(mean((delayed(1:d, warmtrain_pts:(warmtrain_pts + lyaptime_pts - 1)) - x_test(1:d, 1:lyaptime_pts)).^2, 'all') / total_var);
fprintf('test nrmse: %g\n', test_err);

%% Plot
figure('Name', 'Lorenz NG-RC forecast', 'Position', [100 100 1200 800]);
subplot(2,2,1);
plot(delayed(1, warmtrain_pts:end), delayed(3, warmtrain_pts:end), 'LineWidth', 0.3);
xlabel('x'); ylabel('z'); title('ground truth'); axis([-21 21 2 48]);

subplot(2,2,2);
plot(t_eval(warmup_pts:warmtrain_pts-1) - warmup, delayed(1, warmup_pts:warmtrain_pts-1), 'LineWidth', 1.1);
hold on;
plot(t_eval(warmup_pts:warmtrain_pts-1) - warmup, x_predict(1, :), 'r', 'LineWidth', 1.1);
hold off; ylabel('x'); title('training phase');

subplot(2,2,3);
plot(t_eval(warmtrain_pts:(warmtrain_pts + plottime_pts - 1)) - warmup, delayed(1, warmtrain_pts:(warmtrain_pts + plottime_pts - 1)), 'LineWidth', 1.1);
hold on;
plot(t_eval(warmtrain_pts:(warmtrain_pts + plottime_pts - 1)) - warmup, x_test(1, 1:plottime_pts), 'r', 'LineWidth', 1.1);
hold off; xlabel('time'); ylabel('x'); title('testing phase');

subplot(2,2,4);
plot(x_test(1, :), x_test(3, :), 'r', 'LineWidth', 0.3);
xlabel('x'); ylabel('z'); title('NG-RC prediction'); axis([-21 21 2 48]);

end
