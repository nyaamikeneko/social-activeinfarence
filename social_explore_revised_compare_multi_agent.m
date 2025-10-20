%% 社会的探索を行う能動的推論モデル (複数エージェント・複数シミュレーション対応版)
% =========================================================================
clear;
close all;
clc;

%% ヘルパー関数
function y = softmax(x)
    % 数値的安定性のための修正
    x_shifted = x - max(x, [], 1);
    y = exp(x_shifted) ./ sum(exp(x_shifted), 1);
end

%% ステップ 0: シミュレーション全体の初期化
% -------------------------------------------------------------------------
% シミュレーション設定
num_simulations = 100; % シミュレーションの総回数
num_trials = 80;       % 1シミュレーションあたりの総試行回数
num_arms = 9;          % バンディットの腕の数
agent_counts = [1, 2, 3, 4]; % シミュレーションするエージェントの数のリスト

alpha = 0.1;           % 学習率
epsilon = 1e-9;        % 微小値

% 選好の重み
w_reward = 0.8;
w_social_base = 1.0 - w_reward;

% 結果を保存するためのセル配列
results_scoring_rate = cell(1, length(agent_counts));
for i = 1:length(agent_counts)
    results_scoring_rate{i} = zeros(num_simulations, num_trials);
end

fprintf('シミュレーションを開始します...\n');
% =========================================================================
%% メインループ (エージェント数 > シミュレーション回数 > 試行回数)
% =========================================================================
for agent_idx = 1:length(agent_counts)
    num_agents = agent_counts(agent_idx);
    fprintf('エージェント数: %d のシミュレーションを開始します。\n', num_agents);

    for sim = 1:num_simulations
        if mod(sim, 10) == 0
            fprintf('  シミュレーション: %d / %d\n', sim, num_simulations);
        end

        % --- 環境設定 ---
        winning_arm = randi(num_arms); % 当たりの腕を毎シミュレーションでランダムに決定

        % --- 生成モデルの定義 ---
        A = {eye(num_arms), eye(num_arms)};
        B = cell(num_arms, 1);
        for i = 1:num_arms
            B{i} = zeros(num_arms, num_arms);
            B{i}(i, :) = 1;
        end

        % --- エージェントの初期化 ---
        agent_template.s = ones(num_arms, 1) / num_arms;
        agent_template.pi = ones(num_arms, 1) / num_arms;
        agent_template.o1_cum = zeros(num_arms, 1);
        agent_template.o2_cum = zeros(num_arms, 1);

        agents = repmat(agent_template, 1, num_agents);

        C_o1 = repmat(softmax(agents(1).o1_cum), 1, num_agents);
        C_o2 = repmat(softmax(agents(1).o2_cum), 1, num_agents);

        % --- トライアルループ ---
        for t = 1:num_trials

            actions = zeros(num_agents, 1);

            % エージェントが1体の場合は社会的選好を無効にする
            if num_agents == 1
                w_social = 0;
            else
                w_social = w_social_base;
            end

            if t == 1
                % 初回はランダムに行動を選択
                for i = 1:num_agents
                    actions(i) = randi(num_arms);
                    agents(i).pi = zeros(num_arms, 1);
                    agents(i).pi(actions(i)) = 1;
                end
            else
                % ステップ 1 & 2: 期待自由エネルギー (G) の計算と行動選択
                for i = 1:num_agents
                    Qc = kron(C_o1(:,i), C_o2(:,i));
                    G = zeros(num_arms, 1);
                    for pol_idx = 1:num_arms
                        s_predicted = B{pol_idx} * agents(i).s;
                        qo1 = A{1} * s_predicted;
                        qo2 = A{2} * s_predicted;
                        Qs = kron(qo1, qo2);
                        G(pol_idx) = Qs' * (log(Qs + epsilon) - log(Qc + epsilon));
                    end

                    transformed_G = softmax(-G);
                    actions(i) = find(rand < cumsum(transformed_G), 1);
                    agents(i).pi = zeros(num_arms, 1);
                    agents(i).pi(actions(i)) = 1;
                end
            end

            % ステップ 3: 観測の生成
            o1 = zeros(num_arms, num_agents);
            o2 = zeros(num_arms, num_agents);

            for i = 1:num_agents
                % 報酬観測 (o1): 自分の行動が当たりか
                if actions(i) == winning_arm
                    o1(actions(i), i) = 1;
                end

                % 社会的観測 (o2): 他者の行動
                if num_agents > 1
                    other_actions = actions([1:i-1, i+1:end]);
                    for other_act = other_actions'
                        o2(other_act, i) = o2(other_act, i) + 1;
                    end
                end
            end

            % 観測を確率分布に変換
            o1 = softmax(o1);
            o2 = softmax(o2);

            % ステップ 4 & 5: 信念 (s) と事前選好 (C) の更新
            for i = 1:num_agents
                s_prev = agents(i).s;

                ln_s_prior = log(B{actions(i)}' + epsilon) * s_prev;
                X = w_reward * (log(A{1} + epsilon)' * o1(:,i)) + w_social * (log(A{2} + epsilon)' * o2(:,i));
                log_s_posterior = ln_s_prior + X;

                log_s_new = (1 - alpha) * log(s_prev + epsilon) + alpha * log_s_posterior;
                agents(i).s = softmax(log_s_new);

                agents(i).o1_cum = agents(i).o1_cum + w_reward * o1(:,i);
                agents(i).o2_cum = agents(i).o2_cum + w_social * o2(:,i);

                C_o1(:,i) = softmax(agents(i).o1_cum);
                C_o2(:,i) = softmax(agents(i).o2_cum);
            end

            % ステップ 6: 得点率を記録
            correct_actions = sum(actions == winning_arm);
            scoring_rate = correct_actions / num_agents;
            results_scoring_rate{agent_idx}(sim, t) = scoring_rate;
        end
    end
end
fprintf('全てのシミュレーションが完了しました。\n');

%% 結果のプロット (修正版)
% =========================================================================
figure;
hold on;
grid on;

colors = {'b', 'r', 'g', 'm'};
plot_handles = [];
legend_labels = {};

% フォントサイズの設定
axis_font_size = 14;
title_font_size = 16;

for agent_idx = 1:length(agent_counts)
    % 100回のシミュレーションの平均得点率を計算
    mean_scoring_rate = mean(results_scoring_rate{agent_idx}, 1);

    % プロット
    h = plot(1:num_trials, mean_scoring_rate, 'LineWidth', 2, 'Color', colors{agent_idx});
    plot_handles(end+1) = h;

    % 凡例用のラベルを作成
    if agent_counts(agent_idx) == 1
        legend_labels{end+1} = '1 Agent';
    else
        legend_labels{end+1} = [num2str(agent_counts(agent_idx)), ' Agents'];
    end
end

% グラフの装飾
title('Scoring Rate Transition by Number of Agents', 'FontSize', title_font_size);
xlabel('Trials', 'FontSize', axis_font_size);
ylabel('Average Scoring Rate', 'FontSize', axis_font_size);

% 凡例を表示
legend(plot_handles, legend_labels, 'Location', 'SouthEast', 'FontSize', 12);

ylim([0 1]);
set(gca, 'FontSize', 12); % 軸の目盛りのフォントサイズも調整
hold off;
