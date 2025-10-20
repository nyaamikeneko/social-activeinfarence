%% 期待自由エネルギーの精度（beta/gamma）の更新シミュレーション例
% (神経プロセス理論におけるドーパミンと関連付けられる)

% 参考文献: A Step-by-Step Tutorial on Active Inference Modelling and its
% Application to Empirical Data
% 著者: Ryan Smith, Karl J. Friston, Christopher J. Whyte

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

% このスクリプトは図9のシミュレーション結果を再現します

% ここでは、方針の数や、方針の事前・事後精度に寄与する分布を設定できます

E = [1 1 1 1 1]';                     % 方針に関する固定の事前分布（習慣）を設定

G = [12.505 9.51 12.5034 12.505 12.505]'; % 方針に関する期待自由エネルギーの分布例を設定

F = [17.0207 1.7321 1.7321 17.0387 17.0387]'; % 新しい観測後の、方針に関する変分自由エネルギーの分布例を設定


gamma_0 = 1;          % 期待自由エネルギー精度の初期値
gamma = gamma_0;          % 更新される期待自由エネルギー精度の初期値
beta_prior = 1/gamma;     % 期待自由エネルギー精度の事前確率の初期値
beta_posterior = beta_prior; % 期待自由エネルギー精度の事後確率の初期値
psi = 2;              % ステップサイズパラメータ（安定した収束を促進）

for ni = 1:16 % 変分更新の回数 (16回)

    % 方針の事前確率と事後確率を計算（方程式の詳細は本文参照）

    pi_0 = exp(log(E) - gamma*G) / sum(exp(log(E) - gamma*G)); % 方針の事前確率

    pi_posterior = exp(log(E) - gamma*G - F) / sum(exp(log(E) - gamma*G - F)); % 方針の事後確率

    % 期待自由エネルギーの精度を計算

    G_error = (pi_posterior - pi_0)' * -G; % 期待自由エネルギーの予測誤差

    beta_update = beta_posterior - beta_prior + G_error; % betaの変化量: Fのgammaに関する勾配
                                                         % (gamma = 1/beta を想起)

    beta_posterior = beta_posterior - beta_update / psi; % 事後精度の推定値を更新
                                                         % (ステップサイズpsi=2で更新の大きさを抑え、
                                                         % 安定した収束を促進)

    gamma = 1 / beta_posterior; % 期待自由エネルギーの精度を更新

    % ドーパミン応答をシミュレート

    n = ni;

    gamma_dopamine(n,1) = gamma; % 各変分更新ステップでの精度の神経エンコーディングをシミュレート
                                 % (beta_posterior^-1)

    policies_neural(:,n) = pi_posterior; % 各変分更新ステップでの方針の事後確率の神経エンコーディング
end

%% 結果の表示

disp(' ');
disp('最終的な方針の事前確率:');
disp(pi_0);
disp(' ');
disp('最終的な方針の事後確率:');
disp(pi_posterior);
disp(' ');
disp('最終的な方針の差分ベクトル:');
disp(pi_posterior-pi_0);
disp(' ');
disp('負の期待自由エネルギー:');
disp(-G);
disp(' ');
disp('Gの事前精度 (事前のGamma):');
disp(gamma_0);
disp(' ');
disp('Gの事後精度 (Gamma):');
disp(gamma);
disp(' ');

gamma_dopamine_plot = [gamma_0; gamma_0; gamma_0; gamma_dopamine]; % プロット用に初期値を含める

figure
plot(gamma_dopamine_plot);
ylim([min(gamma_dopamine_plot)-.05 max(gamma_dopamine_plot)+.05])
title('Expected Free Energy Precision (Tonic Dopamine)');
xlabel('Updates');
ylabel('\gamma');

figure
plot([gradient(gamma_dopamine_plot)], 'r');
ylim([min(gradient(gamma_dopamine_plot))-.01 max(gradient(gamma_dopamine_plot))+.01])
title('Rate of Change in Precision (Phasic Dopamine)');
xlabel('Updates');
ylabel('\gamma gradient');

% 各方針の信念をエンコードする発火率を表示/プロットしたい場合は、以下のコメントを解除してください
% (列 = 方針, 行 = 時間経過に伴う更新)

% plot(policies_neural);
% disp('方針に関する信念をエンコードする発火率:');
% disp(policies_neural');
% disp(' ');
