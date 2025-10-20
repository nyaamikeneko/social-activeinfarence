%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- 筆記演習問題のコードと解答 --%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 参考文献: A Step-by-Step Tutorial on Active Inference Modelling and its
% Application to Empirical Data
% 著者: Ryan Smith, Karl J. Friston, Christopher J. Whyte

% 注意: 各セクションは個別（セルごと）に実行してください。


%% 静的知覚 (Static perception)

clear
close all
rng('default')

% 事前確率 (D)
D = [0.75 0.25]';

% 尤度マッピング (A行列)
A = [0.8 0.2;
     0.2 0.8];

% 観測 (o)
o = [1 0]';

% 生成モデルを更新式で表現
% lns = log(D) + log(A' * o) と同じ意味
lns = nat_log(D) + nat_log(A' * o);

% ソフトマックス関数で正規化し、事後確率を求める
s = exp(lns) / sum(exp(lns));

disp('状態の事後確率 q(s):');
disp(' ');
disp(s);

% メモ: log(0) は未定義のため、nat_log関数はゼロを非常に小さい値に置き換えます。
% このため、この関数で生成された答えは、教科書に示されている厳密解とは
% わずかに異なる場合があります。

return % このセクションの終わりで処理を停止

%% 動的知覚 (Dynamic perception)

clear
close all
rng('default')

% 事前確率 (D)
D = [0.5 0.5]';

% 尤度マッピング (A行列)
A = [0.9 0.1;
     0.1 0.9];

% 状態遷移 (B行列)
B = [1 0;
     0 1];

% 観測 (o)
% o{試行, 時刻} = [観測値]
o{1,1} = [1 0]';
o{1,2} = [0 0]'; % この観測は実際には使われない
o{2,1} = [1 0]'; % 1回目の観測
o{2,2} = [1 0]'; % 2回目の観測

% 時間ステップの数
T = 2;

% 事後確率を初期化
Qs = zeros(2, T);
for t = 1:T
    Qs(:,t) = [0.5 0.5]';
end

% tは観測の更新回数（この演習では未使用、常に2回目の観測セットo{2,:}を使う）
t = 2;
for i = 1:T %  belief updating iterations
    for tau = 1:T % time points
        % 各時刻tauに対するメッセージを計算
        if tau == 1 % 最初の時刻
            lnD_past = nat_log(D); % 過去からのメッセージ (事前確率)
            lnB_future = nat_log(B' * Qs(:,tau+1)); % 未来からのメッセージ

            % 尤度
            lnAo = nat_log(A' * o{t,tau});

            % 推定の更新
            lns = lnD_past + lnB_future + lnAo;

        elseif tau == T % 最後の時刻
            lnB_past = nat_log(B * Qs(:,tau-1)); % 過去からのメッセージ

            % 尤度
            lnAo = nat_log(A' * o{t,tau});

            % 推定の更新
            lns = lnB_past + lnAo;
        end

        % ソフトマックス関数で正規化し、事後確率を更新
        Qs(:,tau) = exp(lns) / sum(exp(lns));
    end
end

disp('状態の事後確率 q(s):');
disp(' ');
disp(Qs);
