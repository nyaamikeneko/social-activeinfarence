%% 社会的探索を行う能動的推論モデル (エージェント4人版)
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
num_trials = 80; % 総試行回数
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

% 他のエージェントも同様に初期化
agent_B = agent_A;
C_o1_B = C_o1_A;
C_o2_B = C_o2_A;

agent_C = agent_A;
C_o1_C = C_o1_A;
C_o2_C = C_o2_A;

% ★★★ エージェントDを追加 ★★★
agent_D = agent_A;
C_o1_D = C_o1_A;
C_o2_D = C_o2_A;

% --- 履歴の保存用変数 ---
history = struct(); % 構造体を初期化

% ★★★ エージェントDの履歴変数を追加 ★★★
agents = {'A', 'B', 'C', 'D'};
for i = 1:length(agents)
    agent_name = agents{i};
    history.(['actions_', agent_name]) = zeros(1, num_trials);
    history.(['beliefs_', agent_name]) = zeros(num_arms, num_trials);
    history.(['g_', agent_name]) = zeros(num_arms, num_trials);
    history.(['c1_', agent_name]) = zeros(num_arms, num_trials);
    history.(['c2_', agent_name]) = zeros(num_arms, num_trials);
    history.(['pi_', agent_name]) = zeros(num_arms, num_trials);
    history.(['o1_', agent_name]) = zeros(num_arms, num_trials);
    history.(['o2_', agent_name]) = zeros(num_arms, num_trials);
end


%% シミュレーションループ
% =========================================================================
fprintf('シミュレーションを開始します...\n');
for t = 1:num_trials
    fprintf('--- トライアル %d ---\n', t);

    if t == 1
        % 初回はランダムに行動を選択
        action_A = randi(num_arms);
        action_B = randi(num_arms);
        action_C = randi(num_arms);
        action_D = randi(num_arms); % ★Dの行動を追加

        agent_A.pi(action_A) = 1;
        agent_B.pi(action_B) = 1;
        agent_C.pi(action_C) = 1;
        agent_D.pi(action_D) = 1; % ★Dの方策を更新
    else
        % ステップ 1: 期待自由エネルギー (G) の計算
        % --- エージェントA ---
        Qc_A = kron(C_o1_A, C_o2_A); G_A = zeros(num_arms, 1);
        for i = 1:num_arms
            Qs_A = kron(A{1}*(B{i}*agent_A.s), A{2}*(B{i}*agent_A.s));
            G_A(i) = Qs_A' * (log(Qs_A + epsilon) - log(Qc_A + epsilon));
        end

        % --- エージェントB ---
        Qc_B = kron(C_o1_B, C_o2_B); G_B = zeros(num_arms, 1);
        for i = 1:num_arms
            Qs_B = kron(A{1}*(B{i}*agent_B.s), A{2}*(B{i}*agent_B.s));
            G_B(i) = Qs_B' * (log(Qs_B + epsilon) - log(Qc_B + epsilon));
        end

        % --- エージェントC ---
        Qc_C = kron(C_o1_C, C_o2_C); G_C = zeros(num_arms, 1);
        for i = 1:num_arms
            Qs_C = kron(A{1}*(B{i}*agent_C.s), A{2}*(B{i}*agent_C.s));
            G_C(i) = Qs_C' * (log(Qs_C + epsilon) - log(Qc_C + epsilon));
        end

        % ★★★ エージェントDのG計算を追加 ★★★
        Qc_D = kron(C_o1_D, C_o2_D); G_D = zeros(num_arms, 1);
        for i = 1:num_arms
            Qs_D = kron(A{1}*(B{i}*agent_D.s), A{2}*(B{i}*agent_D.s));
            G_D(i) = Qs_D' * (log(Qs_D + epsilon) - log(Qc_D + epsilon));
        end

        % Gをソフトマックス関数で変換し行動選択
        transformed_G_A = softmax(-G_A); action_A = find(rand < cumsum(transformed_G_A), 1);
        transformed_G_B = softmax(-G_B); action_B = find(rand < cumsum(transformed_G_B), 1);
        transformed_G_C = softmax(-G_C); action_C = find(rand < cumsum(transformed_G_C), 1);
        transformed_G_D = softmax(-G_D); action_D = find(rand < cumsum(transformed_G_D), 1); % ★Dの行動を選択

        % 方策(pi)を更新
        agent_A.pi(:) = 0; agent_A.pi(action_A) = 1;
        agent_B.pi(:) = 0; agent_B.pi(action_B) = 1;
        agent_C.pi(:) = 0; agent_C.pi(action_C) = 1;
        agent_D.pi(:) = 0; agent_D.pi(action_D) = 1; % ★Dの方策を更新
    end

    % ステップ 3: 観測の生成
    o1_A = zeros(num_arms, 1); if action_A == winning_arm; o1_A(action_A) = 1; end
    o1_B = zeros(num_arms, 1); if action_B == winning_arm; o1_B(action_B) = 1; end
    o1_C = zeros(num_arms, 1); if action_C == winning_arm; o1_C(action_C) = 1; end
    o1_D = zeros(num_arms, 1); if action_D == winning_arm; o1_D(action_D) = 1; end % ★Dの個人的観測

    % ★★★ 社会的観測を他の3エージェントの行動から生成 ★★★
    o2_A_vec = zeros(num_arms, 1); o2_A_vec(action_B) = o2_A_vec(action_B)+1; o2_A_vec(action_C) = o2_A_vec(action_C)+1; o2_A_vec(action_D) = o2_A_vec(action_D)+1;
    o2_B_vec = zeros(num_arms, 1); o2_B_vec(action_A) = o2_B_vec(action_A)+1; o2_B_vec(action_C) = o2_B_vec(action_C)+1; o2_B_vec(action_D) = o2_B_vec(action_D)+1;
    o2_C_vec = zeros(num_arms, 1); o2_C_vec(action_A) = o2_C_vec(action_A)+1; o2_C_vec(action_B) = o2_C_vec(action_B)+1; o2_C_vec(action_D) = o2_C_vec(action_D)+1;
    o2_D_vec = zeros(num_arms, 1); o2_D_vec(action_A) = o2_D_vec(action_A)+1; o2_D_vec(action_B) = o2_D_vec(action_B)+1; o2_D_vec(action_C) = o2_D_vec(action_C)+1;

    o1_A = softmax(o1_A); o1_B = softmax(o1_B); o1_C = softmax(o1_C); o1_D = softmax(o1_D);
    o2_A = softmax(o2_A_vec); o2_B = softmax(o2_B_vec); o2_C = softmax(o2_C_vec); o2_D = softmax(o2_D_vec);

    % ステップ 4: 信念 (s) の更新
    s_prev_A = agent_A.s; s_prev_B = agent_B.s; s_prev_C = agent_C.s; s_prev_D = agent_D.s;

    % --- エージェントA ---
    ln_s_prior_A = log(B{action_A}'*s_prev_A + epsilon);
    X_A = w_reward * (log(A{1}'*o1_A + epsilon)) + w_social * (log(A{2}'*o2_A + epsilon));
    log_s_posterior_A = ln_s_prior_A + X_A;
    agent_A.s = softmax((1 - alpha) * log(s_prev_A + epsilon) + alpha * log_s_posterior_A);

    % --- エージェントB ---
    ln_s_prior_B = log(B{action_B}'*s_prev_B + epsilon);
    X_B = w_reward * (log(A{1}'*o1_B + epsilon)) + w_social * (log(A{2}'*o2_B + epsilon));
    log_s_posterior_B = ln_s_prior_B + X_B;
    agent_B.s = softmax((1 - alpha) * log(s_prev_B + epsilon) + alpha * log_s_posterior_B);

    % --- エージェントC ---
    ln_s_prior_C = log(B{action_C}'*s_prev_C + epsilon);
    X_C = w_reward * (log(A{1}'*o1_C + epsilon)) + w_social * (log(A{2}'*o2_C + epsilon));
    log_s_posterior_C = ln_s_prior_C + X_C;
    agent_C.s = softmax((1 - alpha) * log(s_prev_C + epsilon) + alpha * log_s_posterior_C);

    % ★★★ エージェントDの信念更新を追加 ★★★
    ln_s_prior_D = log(B{action_D}'*s_prev_D + epsilon);
    X_D = w_reward * (log(A{1}'*o1_D + epsilon)) + w_social * (log(A{2}'*o2_D + epsilon));
    log_s_posterior_D = ln_s_prior_D + X_D;
    agent_D.s = softmax((1 - alpha) * log(s_prev_D + epsilon) + alpha * log_s_posterior_D);

    % ステップ 5: 事前選好 (C) の更新
    agent_A.o1_cum = agent_A.o1_cum + w_reward*o1_A; agent_A.o2_cum = agent_A.o2_cum + w_social*o2_A;
    agent_B.o1_cum = agent_B.o1_cum + w_reward*o1_B; agent_B.o2_cum = agent_B.o2_cum + w_social*o2_B;
    agent_C.o1_cum = agent_C.o1_cum + w_reward*o1_C; agent_C.o2_cum = agent_C.o2_cum + w_social*o2_C;
    agent_D.o1_cum = agent_D.o1_cum + w_reward*o1_D; agent_D.o2_cum = agent_D.o2_cum + w_social*o2_D; % ★Dを追加

    C_o1_A = softmax(agent_A.o1_cum); C_o2_A = softmax(agent_A.o2_cum);
    C_o1_B = softmax(agent_B.o1_cum); C_o2_B = softmax(agent_B.o2_cum);
    C_o1_C = softmax(agent_C.o1_cum); C_o2_C = softmax(agent_C.o2_cum);
    C_o1_D = softmax(agent_D.o1_cum); C_o2_D = softmax(agent_D.o2_cum); % ★Dを追加

    % ステップ 6: 全てのデータを履歴に記録
    history.actions_A(t) = action_A; history.actions_B(t) = action_B; history.actions_C(t) = action_C; history.actions_D(t) = action_D;
    history.beliefs_A(:, t) = agent_A.s; history.beliefs_B(:, t) = agent_B.s; history.beliefs_C(:, t) = agent_C.s; history.beliefs_D(:, t) = agent_D.s;
    if t > 1
        history.g_A(:, t) = transformed_G_A; history.g_B(:, t) = transformed_G_B; history.g_C(:, t) = transformed_G_C; history.g_D(:, t) = transformed_G_D;
    end
    history.c1_A(:, t) = C_o1_A; history.c1_B(:, t) = C_o1_B; history.c1_C(:, t) = C_o1_C; history.c1_D(:, t) = C_o1_D;
    history.c2_A(:, t) = C_o2_A; history.c2_B(:, t) = C_o2_B; history.c2_C(:, t) = C_o2_C; history.c2_D(:, t) = C_o2_D;
    history.pi_A(:, t) = agent_A.pi; history.pi_B(:, t) = agent_B.pi; history.pi_C(:, t) = agent_C.pi; history.pi_D(:, t) = agent_D.pi;
    history.o1_A(:, t) = o1_A; history.o1_B(:, t) = o1_B; history.o1_C(:, t) = o1_C; history.o1_D(:, t) = o1_D;
    history.o2_A(:, t) = o2_A; history.o2_B(:, t) = o2_B; history.o2_C(:, t) = o2_C; history.o2_D(:, t) = o2_D;
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



%% --- パフォーマンスの可視化 (得点率の推移) ---
fprintf('得点率をプロットします...\n');

% 試行回数のベクトルを定義
trials_vec = 1:num_trials;

% 1. 得点率の計算
% 各試行で正解したかどうかを判定
is_correct_A = (history.actions_A == winning_arm);
is_correct_B = (history.actions_B == winning_arm);
is_correct_C = (history.actions_C == winning_arm);
is_correct_D = (history.actions_D == winning_arm); % ★エージェントDの正解判定を追加

% 各試行までの正解数の累計を計算
cumulative_correct_A = cumsum(is_correct_A);
cumulative_correct_B = cumsum(is_correct_B);
cumulative_correct_C = cumsum(is_correct_C);
cumulative_correct_D = cumsum(is_correct_D); % ★エージェントDの累計を追加

% 各試行までの得点率を計算
scoring_rate_A = cumulative_correct_A ./ trials_vec;
scoring_rate_B = cumulative_correct_B ./ trials_vec;
scoring_rate_C = cumulative_correct_C ./ trials_vec;
scoring_rate_D = cumulative_correct_D ./ trials_vec; % ★エージェントDの得点率を追加

% 2. 新しいFigureを作成してプロット
figure('Name', 'Performance Comparison: Scoring Rate');
hold on; % 複数のプロットを重ね書きする設定

% 各エージェントの得点率をプロット
plot(trials_vec, scoring_rate_A, 'b-', 'LineWidth', 2, 'DisplayName', 'Agent A');
plot(trials_vec, scoring_rate_B, 'r-', 'LineWidth', 2, 'DisplayName', 'Agent B');
plot(trials_vec, scoring_rate_C, 'g-', 'LineWidth', 2, 'DisplayName', 'Agent C');
plot(trials_vec, scoring_rate_D, 'm-', 'LineWidth', 2, 'DisplayName', 'Agent D'); % ★エージェントDのプロットを追加

hold off; % 重ね書き設定を終了

% グラフの体裁を整える
title('Scoring Rate Transition');
xlabel('Trial');
ylabel('Scoring Rate (Cumulative)');
legend('show'); % 凡例を表示
grid on;
ylim([0 1.05]);

% チャンスレベルの線を追加
chance_level = 1 / num_arms;
yline(chance_level, 'k--', 'DisplayName', sprintf('Chance Level (%.2f)', chance_level));

fprintf('得点率プロット完了。\n');
