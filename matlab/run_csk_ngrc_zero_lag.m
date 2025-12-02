function run_csk_ngrc_zero_lag()
%RUN_CSK_NGRC_ZERO_LAG BER vs Eb/N0 for ±Chebyshev CSK using zero-lag NG-RC.
%   This script mirrors the single-reservoir ESN setup but replaces the RC
%   with a nonlinear vector autoregressive (NG-RC) readout trained on pooled
%   polynomial delay features. No autoregressive rollout is performed; the
%   classifier maps pooled per-symbol features directly to logits.

%% ------ 基本参数（单帧） ------
SIM.EbN0_dB   = 6:2:20;        % SNR 扫描（单点）
SIM.frameBits = 10000;         % 每个 SNR 仿真 1 帧
SIM.seed      = 1;             % 基础随机种子
train_ratio   = 0.50;          % 训练/测试划分比例（按符号数）

% 片长与初值
CFG.L   = 50;                  % 每符号码片数
CFG.x0  = 0.123456;            % Chebyshev 初值

% NG-RC 特征设置（零时滞）
nvar_degree    = 2;            % 多项式阶数（含常数/线性/二次）
include_const  = true;
m             = 1; tau = 1;   % 嵌入维度/步长

% 分类 readout 正则项
ridge_cls = 1e-3;

% 判决窗口跳过（每符号前缀）
Wskip = floor(CFG.L/5);

numSNR  = numel(SIM.EbN0_dB);
ber_cls = zeros(1,numSNR);     % NG-RC：readout 相似度 logit

rng(SIM.seed, 'twister');
tic;

parfor iSNR = 1:numSNR
    EbN0dB = SIM.EbN0_dB(iSNR);
    rng(SIM.seed + iSNR, 'twister');

    % ===== 1) 生成 ±Chebyshev 长序列，并映射成片段 =====
    Nd     = SIM.frameBits;
    Tchips = CFG.L * Nd;

    x   = chebyshev_series(Tchips, CFG.x0); % [-1,1]
    xm  = -x;

    % bit=0 发送 x；bit=1 发送 -x
    bits     = randi([0,1],[Nd,1]);
    tx_mat_x = reshape(x,  CFG.L, Nd).';
    tx_mat_m = reshape(xm, CFG.L, Nd).';
    tx_mat   = tx_mat_x;
    tx_mat(bits==1,:) = tx_mat_m(bits==1,:);
    target   = tx_mat.'; target = target(:);

    % ===== 2) 含噪观测 =====
    Eavg_bit = mean(target.^2) * CFG.L;
    y_cplx   = add_awgn_complex(target.', EbN0dB, Eavg_bit);
    obs      = real(y_cplx(:));

    % 嵌入（整帧一次性完成）
    U_mix  = delay_embed_signal(obs, m, tau);
    trim   = (m-1)*tau;

    % ===== 3) NG-RC 多项式特征并符号池化 =====
    feature_stream = build_polynomial_features(U_mix(:, trim+1:end), nvar_degree, include_const);
    [Phi_all, valid_mask] = pool_symbol_features(feature_stream, Nd, CFG.L, trim, Wskip);
    valid_idx  = find(valid_mask);
    if numel(valid_idx) < 2
        warning('SNR=%d: valid symbol 数不足，跳过该 SNR', EbN0dB);
        ber_cls(iSNR) = NaN;
        continue;
    end

    n_valid = numel(valid_idx);
    n_train_sym = min(max(floor(train_ratio * n_valid), 1), n_valid-1);
    train_ids = valid_idx(1:n_train_sym);
    test_ids  = valid_idx(n_train_sym+1:end);

    train_mask = false(Nd,1); train_mask(train_ids) = true;
    test_mask  = false(Nd,1); test_mask(test_ids) = true;

    Phi_train = Phi_all(train_mask,:);
    y_train   = 1 - 2*bits(train_mask);       % bit=0 -> +1, bit=1 -> -1
    Icls = eye(size(Phi_train,2));
    W_cls = (Phi_train' * Phi_train + ridge_cls * Icls) \ (Phi_train' * y_train);

    logits = Phi_all * W_cls;
    b_cls  = (logits < 0);
    err_cls = sum(b_cls(test_mask) ~= bits(test_mask));
    ber_cls(iSNR) = err_cls / sum(test_mask);
end

toc;

%% ====== 绘图并保存 ======
h = figure('Color','w'); hold on; grid on; set(gca,'YScale','log');
semilogy(SIM.EbN0_dB, ber_cls, '^--','LineWidth',1.5);
xlabel('E_b/N_0 (dB)'); ylabel('BER');
legend({'Single NG-RC pooled logit (zero-lag)'}, ...
    'Location','southwest');
title(sprintf('BER | \x00b1Cheb CSK | NG-RC, L=%d, m=%d, pooled logit only', ...
    CFG.L, m));

% 保存到 ./fig 子目录
outdir = fullfile(pwd,'fig');
if ~exist(outdir,'dir'), mkdir(outdir); end
ts = datestr(datetime('now'),'yyyymmdd_HHMMSS');
fname = sprintf(['BER_CSK_NGRC_singleRC1L_', ...
    'L%d_m%d_Wskip%d_', ...
    'deg%d_const%d_', ...
    'ridgecls%.0e_', ...
    'seed%d_tr%.2f_%s.fig'], ...
    CFG.L, m, Wskip, ...
    nvar_degree, include_const, ...
    ridge_cls, ...
    SIM.seed, train_ratio, ts);
savefig(h, fullfile(outdir, fname));
end

function x = chebyshev_series(T, x0)
%CHEBYSHEV_SERIES Generate Chebyshev map sequence of order 2 in [-1,1].
%   x(n+1) = cos(2 * acos(x(n))).
x = zeros(1, T);
x(1) = x0;
for n = 1:(T - 1)
    x(n+1) = cos(2 * acos(x(n)));
end
end

function noisy = add_awgn_complex(sig, EbN0dB, Eavg_bit)
%ADD_AWGN_COMPLEX Add complex AWGN to achieve target Eb/N0.
EbN0 = 10.^(EbN0dB/10);
N0 = Eavg_bit ./ EbN0;
noiseVar = N0 / 2;
noise = sqrt(noiseVar/2) * (randn(size(sig)) + 1i * randn(size(sig)));
noisy = sig + noise;
end

function U = delay_embed_signal(signal, m, tau)
%DELAY_EMBED_SIGNAL Delay-embed 1D signal with step tau (cols = time).
signal = signal(:).';
T = numel(signal);
U = zeros(m, T);
for d = 0:(m-1)
    offset = d * tau;
    valid_len = T - offset;
    U(d+1, (offset+1):T) = signal(1:valid_len);
    if offset > 0
        U(d+1, 1:offset) = signal(1);
    end
end
end

function [Phi, valid_mask] = pool_symbol_features(features, Nd, L, trim, Wskip)
%POOL_SYMBOL_FEATURES Mean-pool per-symbol NG-RC features after Wskip.
%   features: F x Ttrimmed (already dropped trim samples)
F = size(features, 1);
Phi = zeros(Nd, F);
valid_mask = false(Nd,1);
Ttrimmed = size(features,2);
for s = 1:Nd
    chip_start = trim + (s-1)*L + 1;
    chip_end = chip_start + L - 1;
    if chip_end > (Ttrimmed + trim)
        continue;
    end
    pool_range = (chip_start + Wskip):chip_end;
    pool_range = pool_range(pool_range > trim); % ensure post-trim
    pool_range = pool_range - trim;
    pool_range = pool_range(pool_range >= 1 & pool_range <= Ttrimmed);
    if isempty(pool_range)
        continue;
    end
    Phi(s, :) = mean(features(:, pool_range).', 1);
    valid_mask(s) = true;
end
end
