%% 変分自由エネルギーの計算例

% 参考文献: A Step-by-Step Tutorial on Active Inference Modelling and its
% Application to Empirical Data
% 著者: Ryan Smith, Karl J. Friston, Christopher J. Whyte

clear all

True_observation = [1 0]'; % 観測を設定;
                           % ここでは観測を追加することも可能。例えば、
                           % [0 0 1]' と設定すれば3番目の観測を提示できる。
                           % その場合、尤度行列に対応する3行目を追加して、
                           % 各状態における3番目の観測の確率を指定する必要がある。
                           % 同様に、事前確率に3番目の要素を追加し、
                           % 尤度行列に対応する3列目を追加すれば、
                           % 状態を3つにすることもできる。

%% 生成モデル

% 事前確率と尤度を指定

Prior = [0.5 0.5]'; % 事前分布 p(s)

Likelihood = [0.8 0.2;
              0.2 0.8]; % 尤度分布 p(o|s): 列=状態, 行=観測

Likelihood_of_observation = Likelihood' * True_observation;

Joint_probability = Prior .* Likelihood_of_observation; % 同時確率分布 p(o,s)

Marginal_probability = sum(Joint_probability, 1); % 周辺尤度（観測の確率） p(o)

%% ベイズの定理：厳密な事後確率

% これが、変分推論を用いて近似したい分布です。
% 多くの実用的な応用では、これを直接解くことはできません。

Posterior = Joint_probability / Marginal_probability; % 観測が与えられたときの事後確率 p(s|o)

disp(' ');
disp('厳密な事後確率:');
disp(Posterior);
disp(' ');

%% 変分自由エネルギー

% 注: q(s) = 近似事後信念。
% 新しい観測を得た後、これを真の事後確率 p(s|o) にできるだけ近づけたい。

% 自由エネルギー(F)の様々な分解

% 1. F = E_q(s)[ln(q(s)/p(o,s))]

% 2. F = E_q(s)[ln(q(s)/p(s))] - E_q(s)[ln(p(o|s))] % 複雑さ(Complexity)と正確さ(Accuracy)による分解

% 第1項は「複雑さ」の項と解釈できる（事前信念 p(s) と近似事後信念 q(s) の間のカルバック・ライブラー・ダイバージェンス）。
% 言い換えれば、新しい観測の後に信念がどれだけ変化したかを表す。

% 第2項（負号を除く）は「正確さ」の項、または（負号を含めると）観測のエントロピー（期待サプライザル）と呼ばれる。
% このように書くと、自由エネルギー最小化は統計的なオッカムの剃刀と等価になる。
% つまり、エージェントは信念の変化を最小限に抑えつつ、最も正確な事後信念を見つけようとする。

% 3. F = E_q(s)[ln(q(s)) - ln(p(s|o)p(o))]

% 4. F = E_q(s)[ln(q(s)/p(s|o))] - ln(p(o))

% これら2つの分解も同様に、Fを q(s) と真の事後確率 p(s|o) との差で示している。ここでは4番目に注目する。

% 第1項は、近似事後確率 q(s) と未知の厳密な事後確率 p(s|o) との間のKLダイバージェンス（相対エントロピーとも呼ばれる）。

% 第2項（負号を除く）は対数エビデンス、または（負号を含めると）観測のサプライザル。
% ln(p(o)) は q(s) に依存しないため、q(s) の下での期待値は単に ln(p(o)) となる。

% この項は q(s) に依存しないため、自由エネルギーを最小化することは、q(s) が未知の目標である p(s|o) に近似することを意味する。

% 5. F = E_q(s)[ln(q(s))-ln(p(o|s)p(s))]

% 以下の変分推論では、計算の便宜上この分解を用いる。
% この分解が F=E_q(s)(ln(q(s)/p(o,s)) という表現と等価であることに注意。
% なぜなら、ln(x)-ln(y) = ln(x/y) であり、p(o|s)p(s)=p(o,s) だからである。

%% 変分推論

Initial_approximate_posterior = Prior; % 近似事後分布の初期値。
                                       % 生成モデルの事前確率と一致させる。

% Fを計算
Initial_F = Initial_approximate_posterior(1) * (log(Initial_approximate_posterior(1)) ...
    - log(Joint_probability(1))) + Initial_approximate_posterior(2) ...
    * (log(Initial_approximate_posterior(2)) - log(Joint_probability(2)));

Optimized_approximate_posterior = Posterior; % 近似分布を真の事後確率に設定

% Fを計算
Minimized_F = Optimized_approximate_posterior(1) * (log(Optimized_approximate_posterior(1)) ...
    - log(Joint_probability(1))) + Optimized_approximate_posterior(2) ...
    * (log(Optimized_approximate_posterior(2)) - log(Joint_probability(2)));

% 近似事後確率 q(s) が真の分布 p(s|o) に近いほど、Fが小さくなることがわかる

disp(' ');
disp('初期の近似事後確率:');
disp(Initial_approximate_posterior);
disp(' ');

disp(' ');
disp('初期の変分自由エネルギー:');
disp(Initial_F);
disp(' ');

disp(' ');
disp('最適化された近似事後確率:');
disp(Optimized_approximate_posterior);
disp(' ');

disp(' ');
disp('最小化された変分自由エネルギー:');
disp(Minimized_F);
disp(' ');
