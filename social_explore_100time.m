%% 社会的探索を行う能動的推論モデル (複数重み・複数回実行対応版)
% =========================================================================
clear;
close all;
clc;

%% ヘルパー関数
% =========================================================================
function y = local_softmax(x)
    % parfor内でグローバルな関数を定義できないため、ローカル関数として定義
    x_shifted = x - max(x);
    y = exp(x_shifted) / sum(exp(x_shifted));
end

%% シミュレーション全体の設定
% -------------------------------------------------------------------------
social_weights = [0.2]; % 試行するw_socialの値
%social_weights = [0, 0.001, 0.2, 0.4, 0.6, 0.8, 1]; % 試行するw_socialの値
num_repetitions = 100;  % 各重みでの繰り返し回数
num_trials = 80;        % 1シミュレーションあたりの総試行回数
num_arms = 9;           % バンディットの腕の数
alpha = 0.1;            % 学習率

% 結果保存用の変数
final_results = zeros(length(social_weights), num_trials);
legend_labels = cell(length(social_weights), 1);

%% メインループ: socialの重みを変えながら実行
% =========================================================================
for w_idx = 1:length(social_weights)
    w_social = social_weights(w_idx);
    w_reward = 1.0 - w_social;

    fprintf('w_social = %.3f のシミュレーションを開始します (%d/%d)...\n', w_social, w_idx, length(social_weights));

    % 現在の重み設定における全繰り返しの得点を保存する変数
    scores_for_current_weight = zeros(num_repetitions, num_trials);

    %並列処理の有効化
    parfor r_idx = 1:num_repetitions
        %disp(['  繰り返し: ', num2str(r_idx), '/', num2str(num_repetitions)]);

        % --- この繰り返し回でのエージェントと環境を初期化 ---

        % 微小値
        epsilon = 1e-9;

        % 環境設定
        winning_arm = randi(num_arms); % 当たりの腕をランダムに決定

        % --- 生成モデルの定義 ---
        A = {eye(num_arms), eye(num_arms)};
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

        C_o1_A = local_softmax(agent_A.o1_cum);
        C_o2_A = local_softmax(agent_A.o2_cum);

        agent_B = agent_A;
        C_o1_B = C_o1_A;
        C_o2_B = C_o2_A;

        % --- 履歴の保存用変数 ---
        history_actions_A = zeros(1, num_trials); % parfor用に変数を変更

        % --- シミュレーションループ (1回の繰り返し) ---
        for t = 1:num_trials
            if t == 1
                % 初回はランダムに行動を選択
                action_A = randi(num_arms);
                action_B = randi(num_arms);
            else
                % ステップ 1: 期待自由エネルギー (G) の計算
                Qc_A = kron(C_o1_A, C_o2_A);
                G_A = zeros(num_arms, 1);
                for i = 1:num_arms
                    s_predicted_A = B{i} * agent_A.s;
                    qo1_A = A{1} * s_predicted_A;
                    qo2_A = A{2} * s_predicted_A;
                    Qs_A = kron(qo1_A, qo2_A);
                    G_A(i) = Qs_A' * (log(Qs_A + epsilon) - log(Qc_A + epsilon));
                end
                transformed_G_A = local_softmax(-G_A);
                action_A = find(rand < cumsum(transformed_G_A), 1);

                Qc_B = kron(C_o1_B, C_o2_B);
                G_B = zeros(num_arms, 1);
                for i = 1:num_arms
                    s_predicted_B = B{i} * agent_B.s;
                    qo1_B = A{1} * s_predicted_B;
                    qo2_B = A{2} * s_predicted_B;
                    Qs_B = kron(qo1_B, qo2_B);
                    G_B(i) = Qs_B' * (log(Qs_B + epsilon) - log(Qc_B + epsilon));
                end
                transformed_G_B = local_softmax(-G_B);
                action_B = find(rand < cumsum(transformed_G_B), 1);
            end

            % ステップ 3: 観測の生成
            o1_A = zeros(num_arms, 1); if action_A == winning_arm; o1_A(action_A) = 1; end
            o1_B = zeros(num_arms, 1); if action_B == winning_arm; o1_B(action_B) = 1; end
            o2_A = zeros(num_arms, 1); o2_A(action_B) = 1;
            o2_B = zeros(num_arms, 1); o2_B(action_A) = 1;

            % ステップ 4: 信念 (s) の更新
            s_prev_A = agent_A.s;
            ln_s_prior_A = log(B{action_A}' + epsilon) * s_prev_A;
            X_A = w_reward * (log(A{1} + epsilon)' * o1_A) + w_social * (log(A{2} + epsilon)' * o2_A);
            log_s_posterior_A = ln_s_prior_A + X_A;
            log_s_new_A = (1 - alpha) * log(s_prev_A + epsilon) + alpha * log_s_posterior_A;
            agent_A.s = local_softmax(log_s_new_A);

            s_prev_B = agent_B.s;
            ln_s_prior_B = log(B{action_B}' + epsilon) * s_prev_B;
            X_B = w_reward * (log(A{1} + epsilon)' * o1_B) + w_social * (log(A{2} + epsilon)' * o2_B);
            log_s_posterior_B = ln_s_prior_B + X_B;
            log_s_new_B = (1 - alpha) * log(s_prev_B + epsilon) + alpha * log_s_posterior_B;
            agent_B.s = local_softmax(log_s_new_B);

            % ステップ 5: 事前選好 (C) の更新
            agent_A.o1_cum = agent_A.o1_cum + w_reward * o1_A;
            agent_A.o2_cum = agent_A.o2_cum + w_social * o2_A;
            agent_B.o1_cum = agent_B.o1_cum + w_reward * o1_B;
            agent_B.o2_cum = agent_B.o2_cum + w_social * o2_B;

            C_o1_A = local_softmax(agent_A.o1_cum);
            C_o2_A = local_softmax(agent_A.o2_cum);
            C_o1_B = local_softmax(agent_B.o1_cum);
            C_o2_B = local_softmax(agent_B.o2_cum);

            % ステップ 6: 行動を記録
            history_actions_A(t) = action_A;
        end % trial loop

        % この繰り返しの得点（当たり=1, ハズレ=0）を計算
        is_correct_A = (history_actions_A == winning_arm);
        scores_for_current_weight(r_idx, :) = is_correct_A;

    end % repetition loop

    % 現在の重み設定での平均得点率を計算
    final_results(w_idx, :) = mean(scores_for_current_weight, 1);
    legend_labels{w_idx} = sprintf('w_social = %.3f', w_social);

end % social weight loop
fprintf('全てのシミュレーションが完了しました。\n');

%% 結果のプロット
% =========================================================================
figure('Name', 'Score Rate Transition by Social Weight', 'NumberTitle', 'off');
hold on;

% 色のサイクルを取得
colors = get(gca, 'ColorOrder');

for w_idx = 1:length(social_weights)
    % plot(x軸, y軸, 'オプション')
    plot(1:num_trials, final_results(w_idx, :), 'LineWidth', 1.5, 'Color', colors(w_idx,:));
end
hold off;

title('Score Rate Transition by Social Weight', 'FontSize', 14);
xlabel('Trial', 'FontSize', 12);
ylabel('Average Score Rate', 'FontSize', 12);
legend(legend_labels, 'Location', 'southeast');
grid on;
ylim([0 1]); % Y軸の範囲を0から1に設定



