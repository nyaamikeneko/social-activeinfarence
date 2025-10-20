%% 社会的探索を行う能動的推論モデル (エージェント3人版)
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
num_trials = 50; % 総試行回数
num_arms = 16;    % バンディットの腕の数

alpha = 0.1;

% 微小値
epsilon = 1e-9;

% 選好の重み
w_reward = 0.8;
w_social = 1.0 - w_reward;

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
% エージェントA
agent_A.s = ones(num_arms, 1) / num_arms;
agent_A.pi = ones(num_arms, 1) / num_arms;
agent_A.o1_cum = zeros(num_arms, 1);
agent_A.o2_cum = zeros(num_arms, 1);
C_o1_A = softmax(agent_A.o1_cum);
C_o2_A = softmax(agent_A.o2_cum);

% エージェントB
agent_B = agent_A;
C_o1_B = C_o1_A;
C_o2_B = C_o2_A;

% ★★★ エージェントCを追加 ★★★
agent_C = agent_A;
C_o1_C = C_o1_A;
C_o2_C = C_o2_A;


% --- 履歴の保存用変数 ---
history.actions_A = zeros(1, num_trials);
history.actions_B = zeros(1, num_trials);
history.beliefs_A = zeros(num_arms, num_trials);
history.beliefs_B = zeros(num_arms, num_trials);
history.g_A = zeros(num_arms, num_trials);
history.g_B = zeros(num_arms, num_trials);
history.c1_A = zeros(num_arms, num_trials);
history.c2_A = zeros(num_arms, num_trials);
history.c1_B = zeros(num_arms, num_trials);
history.c2_B = zeros(num_arms, num_trials);
history.pi_A = zeros(num_arms, num_trials);
history.pi_B = zeros(num_arms, num_trials);
history.o1_A = zeros(num_arms, num_trials);
history.o2_A = zeros(num_arms, num_trials);
history.o1_B = zeros(num_arms, num_trials);
history.o2_B = zeros(num_arms, num_trials);

% ★★★ エージェントCの履歴変数を追加 ★★★
history.actions_C = zeros(1, num_trials);
history.beliefs_C = zeros(num_arms, num_trials);
history.g_C = zeros(num_arms, num_trials);
history.c1_C = zeros(num_arms, num_trials);
history.c2_C = zeros(num_arms, num_trials);
history.pi_C = zeros(num_arms, num_trials);
history.o1_C = zeros(num_arms, num_trials);
history.o2_C = zeros(num_arms, num_trials);


%% シミュレーションループ
% =========================================================================
fprintf('シミュレーションを開始します...\n');
for t = 1:num_trials
    fprintf('--- トライアル %d ---\n', t);

    if t == 1
        % 初回はランダムに行動を選択
        action_A = randi(num_arms);
        action_B = randi(num_arms);
        action_C = randi(num_arms); % ★Cの行動を追加

        agent_A.pi = zeros(num_arms, 1); agent_A.pi(action_A) = 1;
        agent_B.pi = zeros(num_arms, 1); agent_B.pi(action_B) = 1;
        agent_C.pi = zeros(num_arms, 1); agent_C.pi(action_C) = 1; % ★Cの方策を更新
    else
        % ステップ 1: 期待自由エネルギー (G) の計算
        % --- エージェントA ---
        Qc_A = kron(C_o1_A, C_o2_A);
        G_A = zeros(num_arms, 1);
        for i = 1:num_arms
            s_predicted_A = B{i} * agent_A.s;
            qo1_A = A{1} * s_predicted_A;
            qo2_A = A{2} * s_predicted_A;
            Qs_A = kron(qo1_A, qo2_A);
            G_A(i) = Qs_A' * (log(Qs_A + epsilon) - log(Qc_A + epsilon));
        end

        % --- エージェントB ---
        Qc_B = kron(C_o1_B, C_o2_B);
        G_B = zeros(num_arms, 1);
        for i = 1:num_arms
            s_predicted_B = B{i} * agent_B.s;
            qo1_B = A{1} * s_predicted_B;
            qo2_B = A{2} * s_predicted_B;
            Qs_B = kron(qo1_B, qo2_B);
            G_B(i) = Qs_B' * (log(Qs_B + epsilon) - log(Qc_B + epsilon));
        end

        % ★★★ エージェントCのG計算を追加 ★★★
        % --- エージェントC ---
        Qc_C = kron(C_o1_C, C_o2_C);
        G_C = zeros(num_arms, 1);
        for i = 1:num_arms
            s_predicted_C = B{i} * agent_C.s;
            qo1_C = A{1} * s_predicted_C;
            qo2_C = A{2} * s_predicted_C;
            Qs_C = kron(qo1_C, qo2_C);
            G_C(i) = Qs_C' * (log(Qs_C + epsilon) - log(Qc_C + epsilon));
        end

        % Gをソフトマックス関数で変換して行動選択確率に
        transformed_G_A = softmax(-G_A);
        transformed_G_B = softmax(-G_B);
        transformed_G_C = softmax(-G_C); % ★CのGを変換

        % Gに基づく行動選択
        action_A = find(rand < cumsum(transformed_G_A), 1);
        action_B = find(rand < cumsum(transformed_G_B), 1);
        action_C = find(rand < cumsum(transformed_G_C), 1); % ★Cの行動を選択

        % 方策(pi)を更新
        agent_A.pi = zeros(num_arms, 1); agent_A.pi(action_A) = 1;
        agent_B.pi = zeros(num_arms, 1); agent_B.pi(action_B) = 1;
        agent_C.pi = zeros(num_arms, 1); agent_C.pi(action_C) = 1; % ★Cの方策を更新
    end

    % ステップ 3: 観測の生成
    o1_A = zeros(num_arms, 1); if action_A == winning_arm; o1_A(action_A) = 1; end
    o1_B = zeros(num_arms, 1); if action_B == winning_arm; o1_B(action_B) = 1; end
    o1_C = zeros(num_arms, 1); if action_C == winning_arm; o1_C(action_C) = 1; end % ★Cの個人的観測

    % ★★★ 社会的観測を他の2エージェントの行動から生成 ★★★
    o2_A_vec = zeros(num_arms, 1); o2_A_vec(action_B) = 1; o2_A_vec(action_C) = o2_A_vec(action_C) + 1;
    o2_B_vec = zeros(num_arms, 1); o2_B_vec(action_A) = 1; o2_B_vec(action_C) = o2_B_vec(action_C) + 1;
    o2_C_vec = zeros(num_arms, 1); o2_C_vec(action_A) = 1; o2_C_vec(action_B) = o2_C_vec(action_B) + 1;

    o1_A = softmax(o1_A);
    o1_B = softmax(o1_B);
    o1_C = softmax(o1_C); % ★
    o2_A = softmax(o2_A_vec);
    o2_B = softmax(o2_B_vec);
    o2_C = softmax(o2_C_vec); % ★

    % ステップ 4: 信念 (s) の更新
    s_prev_A = agent_A.s;
    s_prev_B = agent_B.s;
    s_prev_C = agent_C.s; % ★

    % --- エージェントA ---
    ln_s_prior_A = log(B{action_A}' + epsilon) * s_prev_A;
    X_A = w_reward * (log(A{1} + epsilon)' * o1_A) + w_social * (log(A{2} + epsilon)' * o2_A);
    log_s_posterior_A = ln_s_prior_A + X_A;
    log_s_new_A = (1 - alpha) * log(s_prev_A + epsilon) + alpha * log_s_posterior_A;
    agent_A.s = softmax(log_s_new_A);

    % --- エージェントB ---
    ln_s_prior_B = log(B{action_B}' + epsilon) * s_prev_B;
    X_B = w_reward * (log(A{1} + epsilon)' * o1_B) + w_social * (log(A{2} + epsilon)' * o2_B);
    log_s_posterior_B = ln_s_prior_B + X_B;
    log_s_new_B = (1 - alpha) * log(s_prev_B + epsilon) + alpha * log_s_posterior_B;
    agent_B.s = softmax(log_s_new_B);

    % ★★★ エージェントCの信念更新を追加 ★★★
    % --- エージェントC ---
    ln_s_prior_C = log(B{action_C}' + epsilon) * s_prev_C;
    X_C = w_reward * (log(A{1} + epsilon)' * o1_C) + w_social * (log(A{2} + epsilon)' * o2_C);
    log_s_posterior_C = ln_s_prior_C + X_C;
    log_s_new_C = (1 - alpha) * log(s_prev_C + epsilon) + alpha * log_s_posterior_C;
    agent_C.s = softmax(log_s_new_C);

    % ステップ 5: 事前選好 (C) の更新
    agent_A.o1_cum = agent_A.o1_cum + w_reward * o1_A;
    agent_A.o2_cum = agent_A.o2_cum + w_social * o2_A;
    agent_B.o1_cum = agent_B.o1_cum + w_reward * o1_B;
    agent_B.o2_cum = agent_B.o2_cum + w_social * o2_B;
    % ★★★ エージェントCの事前選好更新を追加 ★★★
    agent_C.o1_cum = agent_C.o1_cum + w_reward * o1_C;
    agent_C.o2_cum = agent_C.o2_cum + w_social * o2_C;

    C_o1_A = softmax(agent_A.o1_cum);
    C_o2_A = softmax(agent_A.o2_cum);
    C_o1_B = softmax(agent_B.o1_cum);
    C_o2_B = softmax(agent_B.o2_cum);
    % ★★★ エージェントCのCを更新 ★★★
    C_o1_C = softmax(agent_C.o1_cum);
    C_o2_C = softmax(agent_C.o2_cum);

    % ステップ 6: 全てのデータを履歴に記録
    history.actions_A(t) = action_A;
    history.actions_B(t) = action_B;
    history.actions_C(t) = action_C; % ★
    history.beliefs_A(:, t) = agent_A.s;
    history.beliefs_B(:, t) = agent_B.s;
    history.beliefs_C(:, t) = agent_C.s; % ★
    if t > 1
        history.g_A(:, t) = transformed_G_A;
        history.g_B(:, t) = transformed_G_B;
        history.g_C(:, t) = transformed_G_C; % ★
    end
    history.c1_A(:, t) = C_o1_A;
    history.c2_A(:, t) = C_o2_A;
    history.c1_B(:, t) = C_o1_B;
    history.c2_B(:, t) = C_o2_B;
    history.c1_C(:, t) = C_o1_C; % ★
    history.c2_C(:, t) = C_o2_C; % ★
    history.pi_A(:, t) = agent_A.pi;
    history.pi_B(:, t) = agent_B.pi;
    history.pi_C(:, t) = agent_C.pi; % ★
    history.o1_A(:, t) = o1_A;
    history.o2_A(:, t) = o2_A;
    history.o1_B(:, t) = o1_B;
    history.o2_B(:, t) = o2_B;
    history.o1_C(:, t) = o1_C; % ★
    history.o2_C(:, t) = o2_C; % ★
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

% 3. Preference (C1)
subplot(2, 4, 3);
imagesc(trials_vec, 1:num_arms, history.c1_B);
title('Agent B: Preference (C1)');
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

%% --- パフォーマンスの可視化 (得点率の推移) ---
fprintf('得点率をプロットします...\n');

% 試行回数のベクトルを定義
trials_vec = 1:num_trials;

% 1. 得点率の計算
% 各試行で正解したかどうかを判定 (正解なら1, 不正解なら0)
is_correct_A = (history.actions_A == winning_arm);
is_correct_B = (history.actions_B == winning_arm);
is_correct_C = (history.actions_C == winning_arm); % ★エージェントCの正解判定を追加

% 各試行までの正解数の累計を計算
cumulative_correct_A = cumsum(is_correct_A);
cumulative_correct_B = cumsum(is_correct_B);
cumulative_correct_C = cumsum(is_correct_C); % ★エージェントCの累計を追加

% 各試行までの得点率を計算
scoring_rate_A = cumulative_correct_A ./ trials_vec;
scoring_rate_B = cumulative_correct_B ./ trials_vec;
scoring_rate_C = cumulative_correct_C ./ trials_vec; % ★エージェントCの得点率を追加

% 2. 新しいFigureを作成してプロット
figure('Name', 'Performance Comparison: Scoring Rate');
hold on; % 複数のプロットを重ね書きする設定

% エージェントAの得点率をプロット
plot(trials_vec, scoring_rate_A, 'b-', 'LineWidth', 2, 'DisplayName', 'Agent A');
% エージェントBの得点率をプロット
plot(trials_vec, scoring_rate_B, 'r-', 'LineWidth', 2, 'DisplayName', 'Agent B');
% ★エージェントCの得点率をプロット
plot(trials_vec, scoring_rate_C, 'g-', 'LineWidth', 2, 'DisplayName', 'Agent C');

hold off; % 重ね書き設定を終了

% グラフの体裁を整える
title('Scoring Rate Transition');
xlabel('Trial');
ylabel('Scoring Rate (Cumulative)');
legend('show'); % 凡例を表示
grid on;
ylim([0 1.05]); % Y軸の範囲を0から1.05に設定

% チャンスレベル (ランダムに選択した場合の期待値) の線を追加
chance_level = 1 / num_arms;
yline(chance_level, 'k--', 'DisplayName', sprintf('Chance Level (%.2f)', chance_level));

fprintf('得点率プロット完了。\n');
