%% 社会的探索を行う能動的推論モデル (修正版)
% =========================================================================
clear;
close all;
clc;

%% ヘルパー関数
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function y = softmax(x)
    % 数値的安定性のための修正
    x_shifted = x - max(x);
    y = exp(x_shifted) / sum(exp(x_shifted));
end

%% ステップ 0: 初期化
% -------------------------------------------------------------------------
% シミュレーション設定
num_trials = 10; % 総試行回数
num_arms = 4;   % バンディットの腕の数
alpha = 0.1;   % 信念更新の学習率 (0 < alpha <= 1)

% 環境設定
winning_arm = randi(num_arms); % 当たりの腕をランダムに決定
fprintf('環境設定: 当たりのレバーは %d 番です。\n\n', winning_arm);

% --- 生成モデルの定義 ---
% A: 尤度行列 (観測は正確)
A = {eye(num_arms), eye(num_arms)};

% B: 遷移行列 (行動iは状態iへ決定論的に遷移)
B = cell(num_arms, 1);
for i = 1:num_arms
    B{i} = zeros(num_arms, num_arms);
    B{i}(i, :) = 1;
end

% --- エージェントの初期化 ---
agent_A.s = ones(num_arms, 1) / num_arms;
agent_A.pi = ones(num_arms, 1) / num_arms;
agent_A.o1_cum = zeros(num_arms, 1);
agent_A.o2_cum = zeros(num_arms, 1);
C_o1_A = sigmoid(agent_A.o1_cum);
C_o2_A = sigmoid(agent_A.o2_cum);
agent_A.C = sigmoid(C_o1_A .* C_o2_A);

agent_B = agent_A; % エージェントBをAと同じ設定で初期化

% --- 履歴の保存用変数 --- %%% 変更点: 全ての変数をここで初期化 %%%
history.actions_A = zeros(1, num_trials);
history.actions_B = zeros(1, num_trials);
history.beliefs_A = zeros(num_arms, num_trials);
history.beliefs_B = zeros(num_arms, num_trials);
history.g_A = zeros(num_arms, num_trials);
history.g_B = zeros(num_arms, num_trials); % G_Bの履歴を追加
history.c_A = zeros(num_arms, num_trials);
history.c_B = zeros(num_arms, num_trials); % c_Bの履歴を初期化
history.pi_A = zeros(num_arms, num_trials);
history.pi_B = zeros(num_arms, num_trials);
history.o1_A = zeros(num_arms, num_trials);
history.o2_A = zeros(num_arms, num_trials);
history.o1_B = zeros(num_arms, num_trials);
history.o2_B = zeros(num_arms, num_trials);


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
        action_B = randi(num_arms);
        agent_A.pi = zeros(num_arms, 1); agent_A.pi(action_A) = 1;
        agent_B.pi = zeros(num_arms, 1); agent_B.pi(action_B) = 1;
    else
        % ステップ 1: 期待自由エネルギー (G) の計算
        G_A = zeros(num_arms, 1);
        G_B = zeros(num_arms, 1);
        for i = 1:num_arms
            s_predicted_A = B{i}' * agent_A.s;
            s_predicted_B = B{i}' * agent_B.s;
            G_A(i) = s_predicted_A' * (log(s_predicted_A + epsilon) - log(agent_A.C + epsilon));
            G_B(i) = s_predicted_B' * (log(s_predicted_B + epsilon) - log(agent_B.C + epsilon));
        end

        % Gをシグモイド関数で変換して行動選択確率に
        transformed_G_A = softmax(-G_A);
        transformed_G_B = softmax(-G_B);

        % Gが最大の行動を選択 (同値があればランダムに選択)
        maxValue_A = max(transformed_G_A);
        tied_indices_A = find(transformed_G_A == maxValue_A);
        action_A = tied_indices_A(randi(length(tied_indices_A)));

        maxValue_B = max(transformed_G_B);
        tied_indices_B = find(transformed_G_B == maxValue_B);
        action_B = tied_indices_B(randi(length(tied_indices_B)));

        % 方策(pi)を更新
        agent_A.pi = zeros(num_arms, 1); agent_A.pi(action_A) = 1;
        agent_B.pi = zeros(num_arms, 1); agent_B.pi(action_B) = 1;
    end

    % ステップ 3: 観測の生成
    o1_A = zeros(num_arms, 1); if action_A == winning_arm; o1_A(action_A) = 1; end
    o1_B = zeros(num_arms, 1); if action_B == winning_arm; o1_B(action_B) = 1; end
    o2_A = zeros(num_arms, 1); o2_A(action_B) = 1;
    o2_B = zeros(num_arms, 1); o2_B(action_A) = 1;

    o1_A = softmax(o1_A)
    o1_B = softmax(o1_B)
    o2_A = softmax(o2_A)
    o2_B = softmax(o2_B)

    % ステップ 4: 信念 (s) の更新 (修正版)
    %----------------------------------------------------------------------
    s_prev_A = agent_A.s;
    s_prev_B = agent_B.s;

    % 1. o1, o2をsigmoid変換しアダマール積で統合観測oを生成
    o_combined_A = o1_A .* o2_A;
    o_combined_B = o1_B .* o2_B;

    % 2. ターゲットとなる信念を計算 (ln o + ln st-1)
    log_s_target_A = log(o_combined_A + epsilon) + log(s_prev_A + epsilon);
    log_s_target_B = log(o_combined_B + epsilon) + log(s_prev_B + epsilon);

    % 3. 学習率を用いた信念の更新 (この部分は変更なし)
    log_s_new_A = log(s_prev_A + epsilon) + alpha * (log_s_target_A - log(s_prev_A + epsilon));
    log_s_new_B = log(s_prev_B + epsilon) + alpha * (log_s_target_B - log(s_prev_B + epsilon));

    % 4. 正規化 (この部分は変更なし)
    agent_A.s = softmax(log_s_new_A);
    agent_B.s = softmax(log_s_new_B);

    % ステップ 5: 事前選好 (C) の更新
    agent_A.o1_cum = agent_A.o1_cum + o1_A;
    agent_A.o2_cum = agent_A.o2_cum + o2_A;
    agent_B.o1_cum = agent_B.o1_cum + o1_B;
    agent_B.o2_cum = agent_B.o2_cum + o2_B;

    w_reward = 0.7;
    w_social = 1.0 - w_reward;

    C_o1_A = softmax(agent_A.o1_cum);
    C_o2_A = softmax(agent_A.o2_cum);
    agent_A.C = softmax((w_reward * C_o1_A) .* (w_social * C_o2_A));

    C_o1_B = softmax(agent_B.o1_cum);
    C_o2_B = softmax(agent_B.o2_cum);
    agent_B.C = softmax((w_reward * C_o1_B) .* (w_social * C_o2_B));

    % ステップ 6: 全てのデータを履歴に記録 %%% 変更点: ここで全てのデータを記録 %%%
    history.actions_A(t) = action_A;
    history.actions_B(t) = action_B;
    history.beliefs_A(:, t) = agent_A.s;
    history.beliefs_B(:, t) = agent_B.s;
    if t > 1 % Gは2回目以降に計算される
        history.g_A(:, t) = G_A;
        history.g_B(:, t) = G_B;
    end
    history.c_A(:, t) = agent_A.C;
    history.c_B(:, t) = agent_B.C;
    history.pi_A(:, t) = agent_A.pi;
    history.pi_B(:, t) = agent_B.pi;
    history.o1_A(:, t) = o1_A;
    history.o2_A(:, t) = o2_A;
    history.o1_B(:, t) = o1_B;
    history.o2_B(:, t) = o2_B;
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

% 3. Preference (C)
subplot(2, 4, 3);
imagesc(trials_vec, 1:num_arms, history.c_A);
title('Agent A: Preference (C)');
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

% 6. Observation o2 (Other's Action)
subplot(2, 4, 6);
imagesc(trials_vec, 1:num_arms, history.o2_A);
title('Agent A: Obs. o2 (Other''s Action)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 7. Policy (pi)
subplot(2, 4, 7);
imagesc(trials_vec, 1:num_arms, history.pi_A);
title('Agent A: Policy (\pi)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);


%% --- エージェントBのグラフ (Figure 2) ---
figure('Name', 'Agent B: Simulation Dynamics', 'Position', [60, 200, 1200, 600]);

% 1. Action Choice
subplot(2, 4, 1);
scatter(trials_vec, history.actions_B, 36, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
title(['Agent B: Action Choice (Win Arm: ' num2str(winning_arm) ')']);
xlabel('Trial'); ylabel('Arm Number'); ylim([0.5, num_arms + 0.5]); yticks(1:num_arms); grid on;

% 2. Belief (s)
subplot(2, 4, 2);
imagesc(trials_vec, 1:num_arms, history.beliefs_B);
title('Agent B: Belief (s)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 3. Preference (C)
subplot(2, 4, 3);
imagesc(trials_vec, 1:num_arms, history.c_B);
title('Agent B: Preference (C)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 4. Expected Free Energy (G)
subplot(2, 4, 4);
imagesc(trials_vec, 1:num_arms, history.g_B);
title('Agent B: Expected Free Energy (G)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar;

% 5. Observation o1 (Own Success)
subplot(2, 4, 5);
imagesc(trials_vec, 1:num_arms, history.o1_B);
title('Agent B: Obs. o1 (Own Success)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 6. Observation o2 (Other's Action)
subplot(2, 4, 6);
imagesc(trials_vec, 1:num_arms, history.o2_B);
title('Agent B: Obs. o2 (Other''s Action)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

% 7. Policy (pi)
subplot(2, 4, 7);
imagesc(trials_vec, 1:num_arms, history.pi_B);
title('Agent B: Policy (\pi)');
xlabel('Trial'); ylabel('Arm Number'); yticks(1:num_arms); colorbar; caxis([0 1]);

fprintf('プロット完了。\n');


