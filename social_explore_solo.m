%% 社会的探索を行う能動的推論モデル (エージェント1人版)
% =========================================================================
clear;
close all;
clc;

%% ヘルパー関数
function y = softmax(x)
    % 数値的安定性のための修正
    x_shifted = x - max(x);
    y = exp(x_shifted) / sum(exp(x_shifted));
end

%% ステップ 0: 初期化
% -------------------------------------------------------------------------
% シミュレーション設定
num_trials = 100; % 総試行回数
num_arms = 16;    % バンディットの腕の数
alpha = 0.1

% 環境設定
winning_arm = randi(num_arms); % 当たりの腕をランダムに決定
fprintf('環境設定: 当たりのレバーは %d 番です。\n\n', winning_arm);

% --- 生成モデルの定義 ---
% A: 尤度行列 (観測は正確)
A = eye(num_arms); % 観測は報酬(o1)のみなので、セル配列ではなく単一の行列に

% B: 遷移行列 (行動iは状態iへ決定論的に遷移)
B = cell(num_arms, 1);
for i = 1:num_arms
    B{i} = zeros(num_arms, num_arms);
    B{i}(i, :) = 1;
end

% --- エージェントの初期化 ---
agent_A.s = ones(num_arms, 1) / num_arms;
agent_A.pi = ones(num_arms, 1) / num_arms;
agent_A.o1_cum = zeros(num_arms, 1); % 報酬観測(o1)の累積

C_o1_A = softmax(agent_A.o1_cum);

% --- 履歴の保存用変数 --- (エージェントBとo2に関する部分を削除)
history.actions_A = zeros(1, num_trials);
history.beliefs_A = zeros(num_arms, num_trials);
history.g_A = zeros(num_arms, num_trials);       % softmax後の値を保存
history.c_A = zeros(num_arms, num_trials);
history.c1_A = zeros(num_arms, num_trials);      % C1の履歴
history.pi_A = zeros(num_arms, num_trials);
history.o1_A = zeros(num_arms, num_trials);

% 微小値
epsilon = 1e-9;

%% シミュレーションループ
% =========================================================================
fprintf('シミュレーションを開始します...\n');
for t = 1:num_trials
    fprintf('--- トライアル %d ---\n', t);

    if t == 1
        % 初回はランダムに行動を選択
        action_A = randi(num_arms);
        agent_A.pi = zeros(num_arms, 1); agent_A.pi(action_A) = 1;
    else
        % ステップ 1: 期待自由エネルギー (G) の計算
        G_A = zeros(num_arms, 1);
        for i = 1:num_arms
            s_predicted_A = B{i} * agent_A.s;
            qo1_A = A * s_predicted_A; % 期待される観測

            % KLダイバージェンスの計算 (観測が1種類なのでkronは不要)
            G_A(i) = qo1_A' * (log(qo1_A + epsilon) - log(C_o1_A + epsilon));
        end

        % Gをソフトマックス関数で変換して行動選択確率に
        transformed_G_A = softmax(-G_A);

        % Gに基づく行動選択
        action_A = find(rand < cumsum(transformed_G_A), 1);

        % 方策(pi)を更新
        agent_A.pi = zeros(num_arms, 1); agent_A.pi(action_A) = 1;
    end

    % ステップ 3: 観測の生成
    o1_A = zeros(num_arms, 1);
    if action_A == winning_arm
        o1_A(action_A) = 1;
    end
    o1_A = softmax(o1_A);

    % ステップ 4: 信念 (s) の更新
    s_prev_A = agent_A.s;

    % 事前信念の計算
    ln_s_prior_A = log(B{action_A}' + epsilon) * s_prev_A;

    % 尤度に基づく更新 (社会的観測o2の項を削除)
    X_A = log(A + epsilon)' * o1_A;
    log_s_new_A = ln_s_prior_A + X_A;
    log_s_posterior_A = ln_s_prior_A + X_A;

    % 学習率を適用して更新
    log_s_new_A = (1 - alpha) * log(s_prev_A + epsilon) + alpha * log_s_posterior_A;
    agent_A.s = softmax(log_s_new_A);

    % ステップ 5: 事前選好 (C) の更新
    agent_A.o1_cum = agent_A.o1_cum + o1_A;
    C_o1_A = softmax(agent_A.o1_cum);

    % エージェントの選好CはC1と同一に (社会的選好C2を削除)
    agent_A.C = C_o1_A;

    % ステップ 6: 全てのデータを履歴に記録
    history.actions_A(t) = action_A;
    history.beliefs_A(:, t) = agent_A.s;
    if t > 1
        history.g_A(:, t) = transformed_G_A;
    end
    history.c_A(:, t) = agent_A.C;
    history.c1_A(:, t) = C_o1_A;
    history.pi_A(:, t) = agent_A.pi;
    history.o1_A(:, t) = o1_A;
end
fprintf('シミュレーション完了。\n');

%% シミュレーション結果の可視化スクリプト (ウィンドウを分割)
% =========================================================================
fprintf('結果をプロットします...\n');

trials_vec = 1:num_trials;

%% --- エージェントAのグラフ (Figure 1) ---
figure('Name', 'Agent A: Simulation Dynamics', 'Position', [10, 300, 1200, 600]);

% 1. Action Choice
subplot(2, 4, 1);
scatter(trials_vec, history.actions_A, 36, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
title(['Agent A: Action Choice (Win Arm: ' num2str(winning_arm) ')']);
xlabel('Trial'); ylabel('Arm Number'); ylim([0.5, num_arms + 0.5]); yticks(1:num_arms); grid on;

% 2. Belief (s)
subplot(2, 4, 2);
imagesc(trials_vec, 1:num_arms, history.beliefs_A);
title('Agent A: Belief (s)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 3. Preference (C1)
subplot(2, 4, 3);
imagesc(trials_vec, 1:num_arms, history.c1_A);
title('Agent A: Preference (C1)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 4. Expected Free Energy (G)
subplot(2, 4, 4);
imagesc(trials_vec, 1:num_arms, history.g_A);
title('Agent A: Expected Free Energy (G)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar;

% 5. Observation o1 (Own Success)
subplot(2, 4, 5);
imagesc(trials_vec, 1:num_arms, history.o1_A);
title('Agent A: Obs. o1 (Own Success)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 7. Policy (pi)
subplot(2, 4, 7);
imagesc(trials_vec, 1:num_arms, history.pi_A);
title('Agent A: Policy (\pi)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);
