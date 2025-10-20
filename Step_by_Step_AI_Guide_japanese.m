%% 能動的推論モデルの構築と使用に関するステップバイステップの導入

% 論文「能動的推論モデリングとその実証データへの応用に関するステップバイステップチュートリアル」の補足コード

% 著者: Ryan Smith, Karl J. Friston, Christopher J. Whyte
% 更新日: 2024/8/28 (忘却率の実装を変更)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% まず、SPM12、SPM12のDEMツールボックス、および
% このサンプルスクリプトが入っているフォルダをMatlabのパスに追加する必要があります。

clear all
close all      % これらのコマンドはワークスペースをクリアし、すべての図を閉じます

rng('shuffle') % これは乱数生成器を設定し、毎回異なる乱数列を生成します。
               % これにより、シミュレーション結果を繰り返した際にばらつきが生じます。
               % (毎回同じ乱数列を生成するには 'default' に設定することもできます)

% 以下のモデル構築後のシミュレーションオプション:

% Sim = 1 の場合、単一試行をシミュレートします。これは図8を再現します。(ただし、
           % このシミュレーションと以降のシミュレーションでは、ランダムサンプリングのため
           % 結果は毎回異なります)

% Sim = 2 の場合、左のコンテキストが有効な複数試行をシミュレートします
           % (D{1} = [1 0]')。これは図10を再現します。

% Sim = 3 の場合、逆転学習をシミュレートします。初期の試行では左のコンテキストが有効で
           % (D{1} = [1 0]')、後の試行で逆転します
           % (D{1} = [0 1]')。これは図11を再現します。

% Sim = 4 の場合、逆転学習のシミュレーションデータでパラメータ推定を実行します。
           % これは図17の上部パネルを再現します。

% Sim = 5 の場合、複数の参加者からの逆転学習シミュレーションデータを用いて、
           % 異なるモデル（つまり、異なるパラメータ値）下でパラメータ推定を行い、
           % モデル比較を実行します。これは図17の下部パネルを再現します。
           % このオプションは、モデル比較、モデルフィッティング、パラメータ回復可能性分析の結果、
           % およびグループ（PEB）分析に必要な入力を含む2つの構造体も保存します。

rs1 = 4 % リスク追求パラメータ（下の変数rsに設定されます）
           % 図8を再現するには、4または8の値を使用します（Sim = 1の場合）
           % 図10を再現するには、3または4の値を使用します（Sim = 2の場合）
           % 図11を再現するには、3または4の値を使用します（Sim = 3の場合）
           % これはSim = 4またはSim = 5には影響しません

Sim = 2;

% Sim = 5 のとき、PEB = 1 であれば、スクリプトはシミュレートされたグループレベルの
% （パラメトリック経験ベイズ）分析を実行します。

PEB = 0; % 注：GCM_2とGCM_3（PEBへの入力、下記参照）は、Sim = 5の実行後に保存されます。
         % これはPEBを使用するたびに再実行する必要をなくすためです（Sim = 5は時間がかかるため）。
         % 一度Sim = 5を実行した後は、GCM_2とGCM_3をロードして、
         % 後でPEBセクションを個別に実行することができます。

% モデルを構築した後、各セクションをクリックして上の「セクションの実行」をクリックすることで、
% セクションを個別に実行することもできます

%% 1. モデル構造のセットアップ

% 試行内の時間点または「エポック」の数: T
% =========================================================================

% ここでは、3つの時間点（T）を指定します。エージェントは 1) 「開始」状態から始まり、
% 2) 「ヒント」状態または「左を選択」か「右を選択」のスロットマシン状態に移動し、
% 3) 「ヒント」状態から選択状態の一つに移動するか、選択状態の一つから
% 「開始」状態に戻ります。

T = 3;

% 初期状態に関する事前確率: D と d
% =========================================================================

%--------------------------------------------------------------------------
% 生成過程（D）における初期状態に関する事前確率を指定します
% 注：デフォルトでは、これらは生成モデルの事前確率にもなります
%--------------------------------------------------------------------------

% 「コンテキスト」状態因子について、「左が良い」コンテキスト
% （つまり、左のスロットマシンが勝ちやすい）が真のコンテキストであると指定できます：

D{1} = [1 0]';  % {'左が良い','右が良い'}

% 「行動」状態因子について、エージェントは常に試行を「開始」状態で始めると指定できます
% （つまり、スロットマシンを選ぶか、まずヒントを求めるかを選択する前）：

D{2} = [1 0 0 0]'; % {'開始','ヒント','左を選択','右を選択'}

%--------------------------------------------------------------------------
% 生成モデル（d）における初期状態に関する事前信念を指定します
% 注：これは任意であり、指定された場合、状態に関する事前確率の学習をシミュレートします
%--------------------------------------------------------------------------

% これらは技術的には「ディリクレ集中パラメータ」と呼ばれるものであり、
% 0から1の間の値を取る必要はありません。これらの値は、初期状態に関する事後信念に基づいて、
% 各試行後に追加されます。例えば、エージェントが試行1の終わりに
% 「左が良い」コンテキストにいたと信じていた場合、試行2のd{1}は
% d{1} = [1.5 0.5]' となります（ただし、各試行後の値の増加量は学習率に依存します）。
% 一般的に、値が大きいほど初期状態に関する信念の確信度が高いことを示し、
% 信念の変化が遅くなることを意味します（例えば、d{1} = [25 25]' でエンコードされる分布の形状は、
% d{1} = [.5 0.5]' でエンコードされる分布の形状よりも、新しい観測ごとに
% はるかにゆっくりと変化します）。

% コンテキストの信念について、エージェントは両方のコンテキストが等しく可能性があると信じて
% スタートしますが、これらの信念に対する確信度はやや低いと指定できます：

d{1} = [.25 .25]';  % {'左が良い','右が良い'}

% 行動の信念について、エージェントは試行を「開始」状態で始めることを
% 確信していると指定できます：

d{2} = [1 0 0 0]'; % {'開始','ヒント','左を選択','右を選択'}


% 状態-結果のマッピングと信念: A と a
% =========================================================================

%--------------------------------------------------------------------------
% 生成過程（A）における各状態から得られる結果の確率を指定します
% これには、結果のモダリティごとに1つの行列が含まれます
% 注：デフォルトでは、これらは生成モデルの信念にもなります
%--------------------------------------------------------------------------

% まず、状態から観測されるヒントへのマッピングを指定します（結果モダリティ1）。
% ここで、行は観測、列は最初の状態因子（コンテキスト）、
% 3次元目は行動に対応します。各列は合計が1になる確率分布です。

% まず、すべての行動状態において、両方のコンテキストが「ヒントなし」の
% 観測を生成するように指定します：

Ns = [length(D{1}) length(D{2})]; % 各状態因子の状態数（2と4）

for i = 1:Ns(2)

    A{1}(:,:,i) = [1 1; % ヒントなし
                   0 0; % 左のマシンのヒント
                   0 0];% 右のマシンのヒント
end

% 次に、「ヒントを得る」という行動状態が、コンテキスト状態に応じて
% 左または右のスロットマシンが良いというヒントを生成するように指定します。
% この場合、ヒントはpHAの確率で正確です。

pHA = 0.75; % デフォルトでは1に設定しますが、値を変更して
         % モデルの挙動にどう影響するか試してみてください

A{1}(:,:,2) = [0,     0;      % ヒントなし
               pHA, 1-pHA;    % 左のマシンのヒント
               1-pHA, pHA];   % 右のマシンのヒント

% 次に、状態と勝ち負けのマッピングを指定します。最初の2つの
% 行動状態（「開始」と「ヒントを得る」）は、どちらのコンテキストでも勝ちまたは
% 負けの観測を生成しません：

for i = 1:2

    A{2}(:,:,i) = [1 1;  % なし
                   0 0;  % 負け
                   0 0]; % 勝ち
end

% 左のマシンを選択する（行動状態3）と、pWinの確率で勝ちが生成されます。
% これはコンテキスト状態（列）によって異なります：

pWin = .8; % デフォルトでは.8に設定しますが、値を変更して
           % モデルの挙動にどう影響するか試してみてください

A{2}(:,:,3) = [0,      0;      % なし
               1-pWin, pWin;   % 負け
               pWin, 1-pWin]; % 勝ち

% 右のマシンを選択する（行動状態4）と、pWinの確率で勝ちが生成されますが、
% コンテキスト状態へのマッピングは左のマシンを選択した場合と逆になります：

A{2}(:,:,4) = [0,      0;      % なし
               pWin, 1-pWin;   % 負け
               1-pWin, pWin]; % 勝ち

% 最後に、行動状態と観測された行動の間に恒等マッピングを指定し、
% エージェントが行動が計画通りに実行されたことを確実に知るようにします。
% ここで、各行は各行動状態に対応します。

for i = 1:Ns(2)

    A{3}(i,:,i) = [1 1];

end

%--------------------------------------------------------------------------
% 生成モデル（a）における状態-結果マッピングに関する事前信念を指定します
% 注：これは任意であり、指定された場合、状態-結果マッピングの学習をシミュレートします
%--------------------------------------------------------------------------

% ここでは'a'行列の学習はシミュレートしません。
% しかし、初期状態の事前確率の学習と同様に、これは単に
% 生成過程（A）と同じ構造を持つ行列（a）を指定する必要がありますが、
% 生成過程と一致する必要のない信念（およびその信念の確信度）を
% エンコードできるディリクレ集中パラメータを持ちます。学習は、エージェントが特定の状態に
% いたと信じているときに観測された結果に基づいて、行列のエントリの値に
% 加算することに対応します。
% 例えば、エージェントが「左が良い」コンテキストと「左のマシンを選択」の行動状態で
% 勝ちを観測した場合、状態-結果マッピングのその場所に対応する
% 確率値が増加します（例：a{2}(3,1,3)が.8から1.8に変わるかもしれません）。

% この行列を設定する簡単な方法の一つは次のとおりです：

% 1. 最初は生成過程と同一視する
% 2. 行列のすべての側面を学習しないように、値を大きな数で乗算する
%    （これにより分布の形状が非常にゆっくりと変化します）
% 3. 生成過程と異なるようにしたい要素を調整する。

% 例えば、報酬確率の学習をシミュレートするには、次のように指定できます：

    % a{1} = A{1}*200;
    % a{2} = A{2}*200;
    % a{3} = A{3}*200;
    %
    % a{2}(:,:,3) =  [0  0;  % なし
    %                .5 .5;  % 負け
    %                .5 .5]; % 勝ち
    %
    %
    % a{2}(:,:,4) =  [0  0;  % なし
    %                .5 .5;  % 負け
    %                .5 .5]; % 勝ち

% 別の例として、ヒントの正確さを学習することをシミュレートするには、
% 次のように指定するかもしれません：

    % a{1} = A{1}*200;
    % a{2} = A{2}*200;
    % a{3} = A{3}*200;

    % a{1}(:,:,2) =  [0     0;     % ヒントなし
    %                .25   .25;    % 左のマシンのヒント
    %                .25   .25];   % 右のマシンのヒント


% 制御された遷移と遷移信念: B{:,:,u} と b(:,:,u)
%==========================================================================

%--------------------------------------------------------------------------
% 次に、各行動（「制御状態」とも呼ばれる）下での隠れ状態間の
% 確率的遷移を指定する必要があります。
% 注：デフォルトでは、これらは生成モデルの遷移信念にもなります
%--------------------------------------------------------------------------

% 列は時間tの状態。行は時間t+1の状態。

% エージェントはコンテキスト状態を制御できないため、「行動」は1つだけであり、
% コンテキストは試行内で安定していることを示します：

B{1}(:,:,1) = [1 0;  % 「左が良い」コンテキスト
               0 1]; % 「右が良い」コンテキスト

% エージェントは行動状態を制御でき、4つの可能な行動を含めます：

% 他のどの状態からでも開始状態に移動
B{2}(:,:,1) = [1 1 1 1;  % 開始状態
               0 0 0 0;  % ヒント
               0 0 0 0;  % 左のマシンを選択
               0 0 0 0]; % 右のマシンを選択

% 他のどの状態からでもヒント状態に移動
B{2}(:,:,2) = [0 0 0 0;  % 開始状態
               1 1 1 1;  % ヒント
               0 0 0 0;  % 左のマシンを選択
               0 0 0 0]; % 右のマシンを選択

% 他のどの状態からでも左を選択状態に移動
B{2}(:,:,3) = [0 0 0 0;  % 開始状態
               0 0 0 0;  % ヒント
               1 1 1 1;  % 左のマシンを選択
               0 0 0 0]; % 右のマシンを選択

% 他のどの状態からでも右を選択状態に移動
B{2}(:,:,4) = [0 0 0 0;  % 開始状態
               0 0 0 0;  % ヒント
               0 0 0 0;  % 左のマシンを選択
               1 1 1 1]; % 右のマシンを選択

%--------------------------------------------------------------------------
% 生成モデル（b）における状態遷移に関する事前信念を指定します。
% これはBと同じ構造を持つ行列のセットです。
% 注：これは任意であり、指定された場合、状態遷移の学習をシミュレートします。
%--------------------------------------------------------------------------

% この例では、遷移信念の学習はシミュレートしません。
% しかし、dとaの学習と同様に、これは単にディリクレ集中パラメータを
% 蓄積するだけです。ここでは、遷移信念は、エージェントが時間tである状態に
% いて、時間t+1で別の状態にいたと信じている場合に、各試行後に更新されます。

% 選好される結果: C と c
%==========================================================================

%--------------------------------------------------------------------------
% 次に、「事前選好」を、ここでは対数確率としてエンコードして
% 指定する必要があります。
%--------------------------------------------------------------------------

% 結果のモダリティごとに1つの行列。各行は観測、各列は時間点です。
% 負の値は選好が低いことを示し、正の値は選好が高いことを示します。
% 強い選好は、リスクのある選択を促進し、情報探索を減少させます。

% まず、すべての結果に対して選好を0に設定することから始めます：

No = [size(A{1},1) size(A{2},1) size(A{3},1)]; % 各結果モダリティの
                                             % 結果の数

C{1}       = zeros(No(1),T); % ヒント
C{2}       = zeros(No(2),T); % 勝ち/負け
C{3}       = zeros(No(3),T); % 観測された行動

% 次に、時間点2と3での「損失回避」の大きさ（la）と、
% 「報酬追求」（または「リスク追求」）の大きさ（rs）を指定できます。
% ここで、rsは3番目の時間点で2で割られ、スロットマシンを選ぶ前に
% ヒントを得た場合のより小さな勝ち（$4ではなく$2）をエンコードします。

la = 1; % デフォルトでは1に設定しますが、値を変更して
        % モデルの挙動にどう影響するか試してみてください

rs = rs1; % この値はスクリプトの冒頭で設定します。
          % デフォルトでは4に設定しますが、値を変更して
          % モデルの挙動にどう影響するか試してみてください（値が高いほど
          % 本文で説明されているようにリスク追求を促進します）

C{2}(:,:) =    [0  0   0   ;  % なし
                0 -la -la  ;  % 負け
                0  rs  rs/2]; % 勝ち

% ちなみに、これを展開すると、他のC行列は次のようになります：

% C{1} =       [0 0 0;    % ヒントなし
%                0 0 0;    % 左のマシンのヒント
%                0 0 0];   % 右のマシンのヒント
%
% C{3} =       [0 0 0;  % 開始状態
%                0 0 0;  % ヒント
%                0 0 0;  % 左のマシンを選択
%                0 0 0]; % 右のマシンを選択


%--------------------------------------------------------------------------
% 選好の学習をシミュレートするために、選好に関するディリクレ分布（c）を
% 指定することも任意で選択できます。
%--------------------------------------------------------------------------

% ここではシミュレートしません。しかし、これは、結果が観測されるたびに
% その結果に対する選好の大きさを増加させることで機能します。
% ここでの仮定は、より身近な状況に入ることで選好が自然に増加するというものです。
% そのためには、開始時の集中パラメータを指定できます。例えば：

% c{1}       = zeros(No(1),T); % ヒント
% c{2}       = zeros(No(2),T); % 勝ち/負け
% c{3}       = zeros(No(3),T); % 観測された行動
%
% c{2}(:,:) =    [1  1   1  ;  % なし
%                 1  0   0.5;  % 負け
%                 1  2   1.5]; % 勝ち

% 注：これらの値は非負でなければなりません。値が高いほど選好されます。

% 許容される方策: U または V
%==========================================================================

%--------------------------------------------------------------------------
% 各方策は、エージェントが考慮できる時間を通した一連の行動です。
%--------------------------------------------------------------------------

% 方策は、Uで指定されるように「浅い」（一歩先のみを見る）ものとして指定できます。
% または、Vで指定されるように「深い」（試行の終わりまで行動を計画する）ものとして
% 指定できます。UとVの両方は、3番目の行列次元として各状態因子について
% 指定する必要があります。その状態が制御不可能である場合、これはすべて1になります。

% 例えば、Uを指定するには、単に次のようにします：

    % Np = 4; % 方策の数
    % Nf = 2; % 状態因子の数
    %
    % U         = ones(1,Np,Nf);
    %
    % U(:,:,1) = [1 1 1 1]; % コンテキスト状態は制御不可能
    % U(:,:,2) = [1 2 3 4]; % B{2}の4つの行動はすべて許可される

% 今回のシミュレーションでは、Vを指定します。行は時間点に対応し、
% 長さはT-1でなければなりません（ここでは2回の遷移、時間点1から2へ、
% および時間点2から3へ）：

Np = 5; % 方策の数
Nf = 2; % 状態因子の数

V         = ones(T-1,Np,Nf);

V(:,:,1) = [1 1 1 1 1;
            1 1 1 1 1]; % コンテキスト状態は制御不可能

V(:,:,2) = [1 2 2 3 4;
            1 3 4 1 1];

% V(:,:,2)について、列は左から右へ、以下を許可する方策を示します：
% 1. 開始状態に留まる
% 2. ヒントを得てから左のマシンを選択する
% 3. ヒントを得てから右のマシンを選択する
% 4. すぐに左のマシンを選択する（その後、開始状態に戻る）
% 5. すぐに右のマシンを選択する（その後、開始状態に戻る）


% 習慣: E と e
%==========================================================================

%--------------------------------------------------------------------------
% 任意：方策ごとに1つのエントリを持つ列ベクトルで、その方策を選択する
% 事前確率を示します（つまり、他の信念とは独立しています）。
%--------------------------------------------------------------------------

% このシミュレーション例では、エージェントに習慣を装備させませんが、
% 4番目の方策を選択する強い習慣を含めたい場合は、次のように指定できます：

% E = [.1 .1 .1 .6 .1]';

% 選択されるたびに方策がより選択されやすくなる習慣学習を組み込むには、
% eを指定して集中パラメータを指定することもできます。例えば：

% e = [1 1 1 1 1]';

% その他の任意パラメータ
%==========================================================================

% Eta: 学習率（0-1）。各試行後に集中パラメータが更新される際の
% 大きさを制御します（学習が有効な場合）。

    eta = 0.5; % ここではデフォルトで0.5に設定しますが、値を変更して
               % モデルの挙動にどう影響するか試してみてください

% Omega: 忘却率（0-1）。各試行後に集中パラメータの大きさが減少する
% 度合いを制御します（学習が有効な場合）。これは、新しい経験が
% 古い経験から学んだことをどの程度「上書き」できるかを制御します。
% 生成過程の真のパラメータ（事前確率、尤度など）が時間とともに
% 変化する環境で適応的です。omegaの値が高いことは、
% 世界が揮発性であり、条件が時間とともに変化するという事前信念と見なせます。

  omega = 0.0; % ここではデフォルトで0に設定します（忘却なしを示す）、
               % しかし、値を変更してモデルの挙動にどう影響するか試してみてください。
               % 1に近い値はより大きな忘却率を示します。
               % 注：試行1の集中パラメータ値は下限値として設定されます
               % （忘却によってカウントがそれらの値を下回ることはありません -
               % これは公開されたチュートリアル版から変更されています。
               % 下限値を超える集中パラメータは1-OMEGAで乗算されます）

% Beta: 方策に対する期待自由エネルギー（G）の期待精度（正の値で、
% 値が高いほど期待精度が低いことを示します）。
% 値が低いと習慣（E）の影響が大きくなり、それ以外の場合は
% 方策選択がより決定的でなくなります。このシミュレーション例では、
% 単にデフォルト値の1に設定します：

      beta = 1; % デフォルトでは1に設定しますが、値を大きくして
                % 精度を下げ、モデルの挙動にどう影響するか試してみてください

% Alpha: 「逆温度」または「行動精度」パラメータで、
% 行動を選択する際のランダム性の量を制御します（例えば、モデルが
% ヒントを得る行動に最も高い確率を割り当てたとしても、エージェントが
% ヒントを得ないことを選択する頻度）。これは正の数で、値が高いほど
% ランダム性が少なくなります。ここでは高い値に設定します：

    alpha = 32;   % 任意の正の数。1は非常に低く、32はかなり高いです。
                  % 決定論的な行動を指定するために非常に高い値を
                  % 使用することもできます（例：512）

% ERP: このパラメータは、神経応答をシミュレートする際に、試行内の
% 各時間点での信念のリセットの度合いを制御します。値が1の場合は
% リセットがなく、事前信念がスムーズに引き継がれます。値が高いほど、
% 各時間ステップでの事前信念の確信度の喪失度合いが大きくなります。

    erp = 1; % ここではデフォルトで1に設定しますが、値を大きくして
             % シミュレートされた神経（および行動）応答にどう影響するか
             % 試してみてください

% tau: 証拠蓄積の時間定数。このパラメータは、勾配降下の
% 各反復での更新の大きさを制御します。tauの値が大きいと、
% 更新が小さくなり収束時間が遅くなりますが、事後信念の
% 安定性も向上します。

    tau = 12; % ここでは滑らかな生理学的応答をシミュレートするために12に設定しますが、
              % 値を調整してシミュレートされた神経（および行動）応答に
              % どう影響するか試してみてください

% 注：これらの値が指定されていない場合、シミュレーション実行時に
% デフォルト値が割り当てられます。これらのデフォルト値は、
% spm_MDP_VB_Xスクリプト内（およびこのチュートリアルで提供する
% spm_MDP_VB_X_tutorialスクリプト内）で見つけることができます。

% その他の任意の定数
%==========================================================================

% Chi: 深い時間モデルにおける更新閾値のためのオッカムの窓パラメータ。
% 階層モデルでは、このパラメータは、下位レベルの証拠蓄積中に
% 収束がどれだけ早く「打ち切られる」かを制御します。
% 具体的には、不確実性の閾値を設定し、それを下回ると追加の
% 試行エポックはシミュレートされません。デフォルトでは1/64に設定されています。
% より小さい数（例：1/128）は、試行エポックの数が短縮される前に
% より低い不確実性（より高い確信度）が必要であることを示します。

% zeta: 方策のためのオッカムの窓。このパラメータは、方策の自由エネルギーが
% 高くなりすぎた場合（つまり、他の方策に比べて考慮するには
% 不合理になった場合）に、その方策が考慮されなくなる閾値を制御します。
% デフォルトでは3の値に設定されています。値が高いほど閾値が高くなります。
% 例えば、値が6であれば、ある方策が「プルーニング」（つまり、
% 考慮されなくなる）される前に、その方策と最良の方策との間に
% より大きな差が必要であることを示します。zetaの値が小さいほど、
% 方策はより迅速に除去されます。

% 注：spm_MDP_VB_X関数は、混合（離散および連続）モデルの組み込み、
% プロット、シミュレートされた休息/睡眠中のベイジアンモデルリダクションのシミュレーションなど、
% より広範な機能も備えています。ここではこれらを詳細に説明しませんが、
% 関数の冒頭にあるドキュメントに記載されています。

% 真の状態と結果: s と o
%==========================================================================

%--------------------------------------------------------------------------
% 任意で、sとoを用いて一部またはすべての時間点について真の状態と結果を
% 指定することもできます。指定されない場合、これらは
% 生成過程によって生成されます。
%--------------------------------------------------------------------------

% 例えば、これは時間点1での真の状態が左のコンテキストと開始状態であることを意味します：

    %       s = [1;
    %            1]; % 後の時間点（各状態因子の行）は0で、
    %                 % 指定されていないことを示します。


% そして、これは時間点1での観測が「ヒントなし」、「なし」、
% 「開始」の行動観測であることを意味します。

    %       o = [1;
    %            1;
    %            1]; % 後の時間点（各結果モダリティの行）は
    %                 % 0で、指定されていないことを示します

%% 2. MDP構造の定義
%==========================================================================
%==========================================================================

mdp.T = T;                   % 時間ステップの数
mdp.V = V;                   % 許容される（深い）方策

    %mdp.U = U;                   % 代わりに浅い方策を
                                 % 使用することもできました（Vの代わりにUを指定）。

mdp.A = A;                   % 状態-結果マッピング
mdp.B = B;                   % 遷移確率
mdp.C = C;                   % 選好される状態
mdp.D = D;                   % 初期状態に関する事前確率

mdp.d = d; mdp.d_0 = mdp.d;   % 初期状態に関する事前確率の学習を有効にし、
                             % 集中パラメータの下限を設定します（d_0）
mdp.eta = eta;               % 学習率
mdp.omega = omega;           % 忘却率
mdp.alpha = alpha;           % 行動精度
mdp.beta = beta;             % 方策に対する期待自由エネルギーの期待精度
mdp.erp = erp;               % 各タイムステップでの信念のリセット度合い
mdp.tau = tau;               % 証拠蓄積の時間定数

% 注、ここでは習慣は含んでいません：

    % mdp.E = E;

% または他のパラメータの学習：
    % mdp.a = a;  mdp.a_0 = mdp.a;
    % mdp.b = b;  mdp.b_0 = mdp.b;
    % mdp.c = c;  mdp.c_0 = mdp.c; clear mdp.C = C;
    % mdp.e = e;  mdp.e_0 = mdp.e;

% または真の状態や結果の指定：

    % mdp.s = s;
    % mdp.o = o;

% または他の任意パラメータの指定（上記で説明）：

    % mdp.chi = chi;   % 階層モデルの下位レベルでの証拠蓄積を
                     % 停止するための確信度閾値
    % mdp.zeta = zeta; % 不合理な方策を考慮しなくなるためのオッカムの窓

% 後続のプロットのために、状態、結果、行動にラベルを追加できます：

label.factor{1}   = 'contexts';   label.name{1}    = {'left-better','right-better'};
label.factor{2}   = 'choice states';     label.name{2}    = {'start','hint','choose left','choose right'};
label.modality{1} = 'hint';    label.outcome{1} = {'null','left hint','right hint'};
label.modality{2} = 'win/lose';  label.outcome{2} = {'null','lose','win'};
label.modality{3} = 'observed action';  label.outcome{3} = {'start','hint','choose left','choose right'};
label.action{2} = {'start','hint','left','right'};
mdp.label = label;

clear beta
clear alpha
clear eta
clear omega
clear la
clear rs % これらをクリアして、後のシミュレーションで再指定できるようにします

%--------------------------------------------------------------------------
% スクリプトを使用して、すべての行列の次元が正しいかチェックします：
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);


if Sim ==1
%% 3. 単一試行のシミュレーション

%--------------------------------------------------------------------------
% 生成過程とモデルが生成されたので、spm_MDP_VB_Xスクリプトを使用して
% 単一試行をシミュレートできます。ここでは、このチュートリアルに特化したバージョン
% spm_MDP_VB_X_tutorialを提供します。これは、初期状態の事前確率（d）の学習率（eta）と
% 忘却率（omega）を追加するもので、現在のSPMバージョン（21/05/08現在）には
% 含まれていません。
%--------------------------------------------------------------------------

MDP = spm_MDP_VB_X_tutorial(mdp);

% 標準的なプロットルーチンを使用して、シミュレートされた神経応答を
% 可視化できます

spm_figure('GetWin','Figure 1'); clf   % 行動を表示
spm_MDP_VB_LFP(MDP);

%  そして事後信念と行動を表示します：

spm_figure('GetWin','Figure 2'); clf   % 行動を表示
spm_MDP_VB_trial(MDP);

% 図の解釈については本文を参照してください

elseif Sim == 2
%% 4. 複数試行のシミュレーション

% 次に、mdp構造を拡張して複数の試行を含めます

N = 30; % 試行数

MDP = mdp;

[MDP(1:N)] = deal(MDP);

MDP = spm_MDP_VB_X_tutorial(MDP);

% 再びシミュレートされた神経応答を可視化できます

%spm_figure('GetWin','Figure 3'); clf   % 行動を表示
%spm_MDP_VB_game_tutorial(MDP);
% 1. 図のハンドルを取得
F = spm_figure('GetWin','Figure 3');
clf;

% 2. 図を描画
spm_MDP_VB_game_tutorial(MDP);

% 3. 高解像度画像としてファイルに保存 ✨
%    -dpng : ファイル形式をPNGに指定
%    -r300 : 解像度を300 DPIに指定（数値を大きくするとより高精細に）
%    'MyHighResFigure.png' : 保存するファイル名（自由に変更可能）
print(F, '-dpng', '-r300', 'MyHighResFigure.png');


elseif Sim == 3
%% 5. 逆転学習のシミュレーション

N = 32; % 試行数（8の倍数である必要があります）

MDP = mdp;

[MDP(1:N)] = deal(MDP);

    for i = 1:N/8
        MDP(i).D{1}   = [1 0]'; % 初期の試行では「左が良い」コンテキストで
                                % 開始します
    end

    for i = (N/8)+1:N
        MDP(i).D{1}   = [0 1]'; % 残りの試行では「右が良い」コンテキストに
                                % 切り替えます
    end

MDP = spm_MDP_VB_X_tutorial(MDP);

% 再びシミュレートされた神経応答を可視化できます

spm_figure('GetWin','Figure 4'); clf   % 行動を表示
spm_MDP_VB_game_tutorial(MDP);

elseif Sim == 4
%% 6. パラメータ（行動精度とリスク追求）を回復するためのモデル反転
%==========================================================================
%==========================================================================

close all

% 特定のパラメータ値下でのシミュレートされた行動の生成：
%==========================================================================

% 再び逆転学習バージョンを使用します

N = 32; % 試行数

MDP = mdp;

[MDP(1:N)] = deal(MDP);

    for i = 1:N/8
        MDP(i).D{1}   = [1 0]'; % 初期の試行では「左が良い」コンテキストで
                                % 開始します
    end

    for i = (N/8)+1:N
        MDP(i).D{1}   = [0 1]'; % 残りの試行では「右が良い」コンテキストに
                                % 切り替えます
    end

%==========================================================================
% 真のパラメータ値（推定時に回復を試みる）：
%==========================================================================

alpha = 4; % 事前値（16）より低い行動精度（4）を指定
la = 1;    % 損失回避は値1のまま
rs = 6;    % 事前値（5）より高いリスク追求（6）を指定

C_fit = [0  0   0 ;    % なし
         0 -la -la  ;  % 負け
         0  rs  rs/2]; % 勝ち

[MDP(1:N).alpha] = deal(alpha);

for i = 1:N
    MDP(i).C{2} = C_fit;
end


% 必要であれば、他のパラメータの真の値を同様に調整することもできます。
% 例えば：

    % beta = 5; % 事前値（1）より低い期待方策精度（5）を指定
    % [MDP(1:N).beta] = deal(beta);

    % eta = .9; % 事前値（.5）より高い学習率（.9）を指定
    % [MDP(1:N).eta] = deal(eta);

%==========================================================================

MDP = spm_MDP_VB_X_tutorial(MDP);


% モデルを反転し、元のパラメータを回復しようと試みる：
%==========================================================================

%--------------------------------------------------------------------------
% ここでモデル反転を行います。モデル反転は変分ベイズに基づいています。
% ここでは、自由パラメータ（ここではalphaとrs）に関して（負の）変分自由エネルギーを
% 最大化します。これは、これらのパラメータ下でのデータの尤度を最大化する
% （つまり、精度を最大化する）ことと、同時にパラメータの事前分布からの
% 大きな逸脱を罰すること（つまり、複雑さを最小化する）に対応し、
% 過学習を防ぎます。
%
% 各パラメータの事前平均と分散は、Estimate_parametersスクリプトの
% 冒頭で指定できます。
%--------------------------------------------------------------------------
mdp.la_true = la;    % 推定スクリプトで使用するために真のla値を引き継ぎます
mdp.rs_true = rs;    % 推定スクリプトで使用するために真のrs値を引き継ぎます

DCM.MDP   = mdp;               % 推定されるMDPモデル
DCM.field = {'alpha','rs'};    % 最適化するパラメータ（フィールド）名

% 注：他のパラメータをフィットさせたい場合は、単にそれらの
% フィールド名を追加するだけです。例：

 % DCM.field = {'alpha','rs','eta'}; % 学習率をフィットさせたい場合

% これには、それらのパラメータがEstimate_parametersスクリプトで指定された
% 可能なパラメータにも含まれている必要があります。

% 次に、（シミュレートされた）参加者の真の観測と行動を追加します

DCM.U     = {MDP.o};           % （実在またはシミュレートされた）参加者によって
                               % 行われた観測を含めます

DCM.Y     = {MDP.u};           % （実在またはシミュレートされた）参加者によって
                               % 行われた行動を含めます

DCM       = Estimate_parameters(DCM); % パラメータ推定関数を実行

subplot(2,2,3)
xticklabels(DCM.field),xlabel('parameter')
subplot(2,2,4)
xticklabels(DCM.field),xlabel('parameter')

% 事前および事後平均の偏差と事後共分散を確認
%==========================================================================

%--------------------------------------------------------------------------
% 値を再変換し、事前推定値と事後推定値を比較します
%--------------------------------------------------------------------------

field = fieldnames(DCM.M.pE);
for i = 1:length(field)
    if strcmp(field{i},'eta')
        prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    elseif strcmp(field{i},'omega')
        prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    else
        prior(i) = exp(DCM.M.pE.(field{i}));
        posterior(i) = exp(DCM.Ep.(field{i}));
    end
end

figure, set(gcf,'color','white')
subplot(2,1,1),hold on
title('Means')
bar(prior,'FaceColor',[.5,.5,.5]),bar(posterior,0.5,'k')
xlim([0,length(prior)+1]),set(gca, 'XTick', 1:length(prior)),set(gca, 'XTickLabel', DCM.field)
legend({'Prior','Posterior'})
hold off
subplot(2,1,2)
imagesc(DCM.Cp),caxis([0 1]),colorbar
title('(Co-)variance')
set(gca, 'XTick', 1:length(prior)),set(gca, 'XTickLabel', DCM.field)
set(gca, 'YTick', 1:length(prior)),set(gca, 'YTickLabel', DCM.field)

% 回復可能性の証拠を示すために、さまざまなパラメータで生成された
% シミュレーションデータからパラメータを推定し、その後、
% 真のパラメータと推定されたパラメータ間の相関の強さをチェックして、
% 合理的に強い関係があることを確認したい場合があります。
% これをセクション7の最後で試みます。

elseif Sim == 5
%% 7. モデル比較
%==========================================================================
%==========================================================================

% これから6人の参加者のデータをシミュレートし、2つのモデルにフィットさせます：
% 1つは行動精度（alpha）とリスク追求（rs）のみをフィットさせるモデル、
% もう1つは学習率（eta）もフィットさせるモデルです。

% 結果を保存するためのベクトル/行列を作成

F_2_params = [];
F_3_params = [];

avg_LL_2_params = [];
avg_prob_2_params = [];
avg_LL_3_params = [];
avg_prob_3_params = [];

GCM_2 = {};
GCM_3 = {};

Sim_params_2 = [];
true_params_2 = [];
Sim_params_3 = [];
true_params_3 = [];

% 以前のように逆転学習試行を設定

N = 32; % 試行数

MDP = mdp;

[MDP(1:N)] = deal(MDP);

    for i = 1:N/8
        MDP(i).D{1}   = [1 0]'; % 初期の試行では「左が良い」コンテキストで
                                % 開始します
    end

    for i = (N/8)+1:N
        MDP(i).D{1}   = [0 1]'; % 残りの試行では「右が良い」コンテキストに
                                % 切り替えます
    end

% 2パラメータモデル（etaなし）のモデルフィットに対する自由エネルギーを生成

rs_sequence = [4 6];     % 異なる真のリスク追求値を指定（事前=5）
alpha_sequence = [4 16 24]; % 異なる真の行動精度を指定（事前=16）


for rs = rs_sequence  % 異なる真のリスク追求値を指定（事前=5）
    for alpha = alpha_sequence   % 異なる真の行動精度を指定（事前=16）


MDP_temp = MDP;

la = 1;    % 損失回避は値1のまま

C_fit = [0  0   0 ;    % なし
         0 -la -la  ;  % 負け
         0  rs  rs/2]; % 勝ち

[MDP_temp(1:N).alpha] = deal(alpha);

for i = 1:N
    MDP_temp(i).C{2} = C_fit;
end

mdp.la_true = la;    % 推定スクリプトで使用するために真のla値を引き継ぎます
mdp.rs_true = rs;    % 推定スクリプトで使用するために真のrs値を引き継ぎます

MDP_temp = spm_MDP_VB_X_tutorial(MDP_temp);

spm_figure('GetWin','Figure 5'); clf   % フィットさせる行動を表示
spm_MDP_VB_game_tutorial(MDP_temp);

DCM.MDP   = mdp;               % 推定されるMDPモデル
DCM.field = {'alpha','rs'};    % 最適化するパラメータ（フィールド）名

DCM.U     = {MDP_temp.o};      % （実在またはシミュレートされた）参加者によって
                               % 行われた観測を含めます

DCM.Y     = {MDP_temp.u};      % （実在またはシミュレートされた）参加者によって
                               % 行われた行動を含めます

DCM       = Estimate_parameters(DCM); % パラメータ推定関数を実行

% パラメータを対数またはロジット空間から元に戻す

field = fieldnames(DCM.M.pE);
for i = 1:length(field)
    if strcmp(field{i},'eta')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    elseif strcmp(field{i},'omega')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    else
        DCM.prior(i) = exp(DCM.M.pE.(field{i}));
        DCM.posterior(i) = exp(DCM.Ep.(field{i}));
    end
end

F_2_params = [F_2_params DCM.F];% 各参加者のモデルの自由エネルギーを取得

GCM_2   = [GCM_2;{DCM}]; % 各参加者のDCMを保存

% 最適フィットモデルの対数尤度と行動確率を取得

MDP_best = MDP;

[MDP_best(1:N).alpha] = deal(DCM.posterior(1));

C_fit_best = [0  0   0 ;                               % なし
              0 -la -la  ;                             % 負け
              0  DCM.posterior(2)  DCM.posterior(2)/2]; % 勝ち

for i = 1:N
    MDP_best(i).C{2} = C_fit_best;
end

for i = 1:N
    MDP_best(i).o = MDP_temp(i).o;
end

for i = 1:N
    MDP_best(i).u = MDP_temp(i).u;
end

MDP_best   = spm_MDP_VB_X_tutorial(MDP_best); % 最適なパラメータ値でモデルを実行

% 試行にわたる各行動の対数尤度の合計を取得

L      = 0; % モデルを与えられた行動の（対数）確率を0で開始
total_prob = 0;

for i = 1:numel(MDP_best) % 各試行の真の行動の確率を取得
    for j = 1:numel(MDP_best(1).u(2,:)) % 2番目（制御可能）の状態因子の確率のみ取得

        L = L + log(MDP_best(i).P(:,MDP_best(i).u(2,j),j)+ eps); % 各行動の（対数）確率を合計する
                                                               % 特定のパラメータ値のセットが与えられた場合
        total_prob = total_prob + MDP_best(i).P(:,MDP_best(i).u(2,j),j); % 各行動の（対数）確率を合計する
                                                                       % 特定のパラメータ値のセットが与えられた場合

    end
end

% 各参加者の平均対数尤度と、最適フィットパラメータ下での各参加者の
% 平均行動確率を取得

avg_LL_2 = L/(size(MDP_best,2)*2);

avg_LL_2_params = [avg_LL_2_params; avg_LL_2];

avg_prob_2 = total_prob/(size(MDP_best,2)*2);

avg_prob_2_params = [avg_prob_2_params; avg_prob_2];

% 回復可能性を評価するために真のパラメータと推定パラメータを保存

Sim_params_2 = [Sim_params_2; DCM.posterior];% 事後分布を取得
true_params_2 = [true_params_2; [alpha rs]];% 真のパラメータを取得

clear DCM
clear MDP_temp
clear MDP_best

    end
end

% 真のパラメータとシミュレートされたパラメータを別々に保存

True_alpha_2 = true_params_2(:,1);
Estimated_alpha_2 = Sim_params_2(:,1);
True_rs_2 = true_params_2(:,2);
Estimated_rs_2 = Sim_params_2(:,2);

% 3パラメータモデル（etaあり）のモデルフィットに対する自由エネルギーを生成

for rs = rs_sequence  % 異なる真のリスク追求値を指定（事前=2）
    for alpha = alpha_sequence   % 異なる真の行動精度を指定（事前=16）

MDP_temp = MDP;

la = 1;    % 損失回避は値1のまま

if rs == rs_sequence(1,1)
    eta = .2; % 3人の参加者に対して、推定事前値（.5）より低いetaの値を設定
elseif rs == rs_sequence(1,2)
    eta = .8; % 3人の参加者に対して、推定事前値（.5）より高いetaの値を設定
end


C_fit = [0  0   0 ;    % なし
         0 -la -la  ;  % 負け
         0  rs  rs/2]; % 勝ち

[MDP_temp(1:N).alpha] = deal(alpha);
[MDP_temp(1:N).eta] = deal(eta);

for i = 1:N
    MDP_temp(i).C{2} = C_fit;
end

mdp.la_true = la;    % 推定スクリプトで使用するために真のla値を引き継ぎます
mdp.rs_true = rs;    % 推定スクリプトで使用するために真のrs値を引き継ぎます

MDP_temp = spm_MDP_VB_X_tutorial(MDP_temp);

spm_figure('GetWin','Figure 6'); clf   % フィットさせる行動を表示
spm_MDP_VB_game_tutorial(MDP_temp);

DCM.MDP   = mdp;               % 推定されるMDPモデル
DCM.field = {'alpha','rs','eta'}; % 最適化するパラメータ（フィールド）名

DCM.U     = {MDP_temp.o};      % （実在またはシミュレートされた）参加者によって
                               % 行われた観測を含めます

DCM.Y     = {MDP_temp.u};      % （実在またはシミュレートされた）参加者によって
                               % 行われた行動を含めます

DCM       = Estimate_parameters(DCM); % パラメータ推定関数を実行

% パラメータを対数またはロジット空間から元に戻す

field = fieldnames(DCM.M.pE);
for i = 1:length(field)
    if strcmp(field{i},'eta')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    elseif strcmp(field{i},'omega')
        DCM.prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
        DCM.posterior(i) = 1/(1+exp(-DCM.Ep.(field{i})));
    else
        DCM.prior(i) = exp(DCM.M.pE.(field{i}));
        DCM.posterior(i) = exp(DCM.Ep.(field{i}));
    end
end


F_3_params = [F_3_params DCM.F]; % 各参加者のモデルの自由エネルギーを取得

GCM_3   = [GCM_3;{DCM}]; % 各参加者のDCMを保存

% 最適フィットモデルの対数尤度と行動確率を取得

MDP_best = MDP;

[MDP_best(1:N).alpha] = deal(DCM.posterior(1));

C_fit_best = [0  0   0 ;                               % なし
              0 -la -la  ;                             % 負け
              0  DCM.posterior(2)  DCM.posterior(2)/2]; % 勝ち

for i = 1:N
    MDP_best(i).C{2} = C_fit_best;
end

if rs == rs_sequence(1,1)
    eta = .2; % 3人の参加者に対して、推定事前値（.5）より低いetaの値を設定
elseif rs == rs_sequence(1,2)
    eta = .8; % 3人の参加者に対して、推定事前値（.5）より高いetaの値を設定
end

[MDP_best(1:N).eta] = deal(eta);


for i = 1:N
    MDP_best(i).o = MDP_temp(i).o;
end

for i = 1:N
    MDP_best(i).u = MDP_temp(i).u;
end

MDP_best   = spm_MDP_VB_X_tutorial(MDP_best); % 最適なパラメータ値でモデルを実行

% 試行にわたる各行動の対数尤度の合計を取得

L      = 0; % モデルを与えられた行動の（対数）確率を0で開始
total_prob = 0;

for i = 1:numel(MDP_best) % 各試行の真の行動の確率を取得
    for j = 1:numel(MDP_best(1).u(2,:)) % 2番目（制御可能）の状態因子の確率のみ取得

        L = L + log(MDP_best(i).P(:,MDP_best(i).u(2,j),j)+ eps); % 各行動の（対数）確率を合計する
                                                               % 特定のパラメータ値のセットが与えられた場合
        total_prob = total_prob + MDP_best(i).P(:,MDP_best(i).u(2,j),j); % 各行動の（対数）確率を合計する
                                                                       % 特定のパラメータ値のセットが与えられた場合

    end
end

% 各参加者の平均対数尤度と、最適フィットパラメータ下での各参加者の
% 平均行動確率を取得

avg_LL_3 = L/(size(MDP_best,2)*2);

avg_LL_3_params = [avg_LL_3_params; avg_LL_3];

avg_prob_3 = total_prob/(size(MDP_best,2)*2);

avg_prob_3_params = [avg_prob_3_params; avg_prob_3];

% 回復可能性を評価するために真のパラメータと推定パラメータを保存

Sim_params_3 = [Sim_params_3; DCM.posterior];% 事後分布を取得
true_params_3 = [true_params_3; [alpha rs eta]];% 真のパラメータを取得

clear DCM
clear MDP_temp
clear MDP_best

    end
end

% 真のパラメータとシミュレートされたパラメータを別々に保存

True_alpha_3 = true_params_3(:,1);
Estimated_alpha_3 = Sim_params_3(:,1);
True_rs_3 = true_params_3(:,2);
Estimated_rs_3 = Sim_params_3(:,2);
True_eta_3 = true_params_3(:,3);
Estimated_eta_3 = Sim_params_3(:,3);

clear alpha

% ランダム効果ベイジアンモデル比較（各参加者の最適フィットモデルの
% 自由エネルギーの比較）：

F_2_params = F_2_params';
F_3_params = F_3_params'; % 自由エネルギーを列ベクトルに変換

[alpha,exp_r,xp,pxp,bor] = spm_BMS([F_2_params F_3_params]);

disp(' ');
disp(' ');
disp('Protected exceedance probability (pxp):');
disp(pxp);
disp(' ');

% pxp値は保護超過確率（pxp）であり、
% 各モデルが最適フィットモデルである確率を提供します。例えば、
% pxp = [.37 .63]は、3パラメータモデルの確率が高いことを示します

%--------------------------------------------------------------------------

% 2パラメータおよび3パラメータモデル下での行動の平均確率と対数尤度（LL）も
% 計算できます：

average_LL_2p = mean(avg_LL_2_params);
average_action_probability_2p = mean(avg_prob_2_params);
average_LL_3p = mean(avg_LL_3_params);
average_action_probability_3p = mean(avg_prob_3_params);

disp(' ');
fprintf('Average log-likelihood under the 2-parameter model: %.2g\n',average_LL_2p);
fprintf('Average action probability under the 2-parameter model: %.2g\n',average_action_probability_2p);
disp(' ');
fprintf('Average log-likelihood under the 3-parameter model: %.2g\n',average_LL_2p);
fprintf('Average action probability under the 3-parameter model: %.2g\n',average_action_probability_2p);
disp(' ');

%% 回復可能性に関するセクション6の簡単な続き
%==========================================================================
% ここでは、回復可能性を確認するために、真のパラメータと推定されたパラメータ間の
% 関係の強さを計算することもできます。
%==========================================================================

% 相関のための行列を組み立てる（2パラメータモデル）
recover_check_alpha_2 = [True_alpha_2 Estimated_alpha_2];
recover_check_rs_2 = [True_rs_2 Estimated_rs_2];

% 相関と有意性を取得
[Correlations_alpha_2, Significance_alpha_2] = corrcoef(recover_check_alpha_2);
[Correlations_rs_2, Significance_rs_2] = corrcoef(recover_check_rs_2);

% この場合、rsの相関は非常に高く、alphaの相関は中程度に見えます

disp(' ');
disp('2-parameter model:');
disp(' ');
fprintf('Alpha recoverability: r = %.2g\n',Correlations_alpha_2(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_alpha_2(1,2));
disp(' ');
fprintf('Risk-seeking recoverability: r = %.2g\n',Correlations_rs_2(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_rs_2(1,2));
disp(' ');

% 相関のための行列を組み立てる（3パラメータモデル）
recover_check_alpha_3 = [True_alpha_3 Estimated_alpha_3];
recover_check_rs_3 = [True_rs_3 Estimated_rs_3];
recover_check_eta_3 = [True_eta_3 Estimated_eta_3];

% 相関と有意性を取得
[Correlations_alpha_3, Significance_alpha_3] = corrcoef(recover_check_alpha_3);
[Correlations_rs_3, Significance_rs_3] = corrcoef(recover_check_rs_3);
[Correlations_eta_3, Significance_eta_3] = corrcoef(recover_check_eta_3);

% この場合、rsとalphaの相関は高く、学習率は中程度に見えます。
% ただし、実際の研究で回復可能性を確認するには、より広い範囲の値を
% シミュレートし、より多くの被験者で確認する必要があります。

disp(' ');
disp('3-parameter model:');
disp(' ');
fprintf('Alpha recoverability: r = %.2g\n',Correlations_alpha_3(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_alpha_3(1,2));
disp(' ');
fprintf('Risk-seeking recoverability: r = %.2g\n',Correlations_rs_3(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_rs_3(1,2));
disp(' ');
fprintf('Learning rate recoverability: r = %.2g\n',Correlations_eta_3(1,2));
fprintf('Correlation significance: p = %.2g\n',Significance_eta_3(1,2));
disp(' ');
%% 結果の整理と保存

two_parameter_model_estimates.alpha_true = recover_check_alpha_2(:,1);
two_parameter_model_estimates.alpha_estimated = recover_check_alpha_2(:,2);
two_parameter_model_estimates.risk_seeking_true = recover_check_rs_2(:,1);
two_parameter_model_estimates.risk_seeking_estimated = recover_check_rs_2(:,2);
two_parameter_model_estimates.final_log_likelihoods = avg_LL_2_params;
two_parameter_model_estimates.final_action_probabilities = avg_prob_2_params;
two_parameter_model_estimates.protected_exceedance_probability = pxp;

three_parameter_model_estimates.alpha_true = recover_check_alpha_3(:,1);
three_parameter_model_estimates.alpha_estimated = recover_check_alpha_3(:,2);
three_parameter_model_estimates.risk_seeking_true = recover_check_rs_3(:,1);
three_parameter_model_estimates.risk_seeking_estimated = recover_check_rs_3(:,2);
three_parameter_model_estimates.learning_rate_true = recover_check_eta_3(:,1);
three_parameter_model_estimates.learning_rate_estimated = recover_check_eta_3(:,2);
three_parameter_model_estimates.final_log_likelihoods = avg_LL_3_params;
three_parameter_model_estimates.final_action_probabilities = avg_prob_3_params;
three_parameter_model_estimates.protected_exceedance_probability = pxp;

save('Two_parameter_model_estimates','two_parameter_model_estimates');
save('Three_parameter_model_estimates','three_parameter_model_estimates');
save('GCM_2','GCM_2');
save('GCM_3','GCM_3');

figure
scatter(two_parameter_model_estimates.alpha_true,two_parameter_model_estimates.alpha_estimated,'filled')
lsline
title('Recoverability: Alpha (two-parameter model)')
xlabel('True (Generative) Alpha')
ylabel('Estimated Alpha')
[Corr_alpha_2, Sig_alpha_2] = corrcoef(two_parameter_model_estimates.alpha_true,two_parameter_model_estimates.alpha_estimated);
text(1, 23, ['r = ' num2str(Corr_alpha_2(1,2))])
text(1, 22, ['p = ' num2str(Sig_alpha_2(1,2))])

figure
scatter(two_parameter_model_estimates.risk_seeking_true,two_parameter_model_estimates.risk_seeking_estimated,'filled')
lsline
title('Recoverability: Risk-Seeking (two-parameter model)')
xlabel('True (Generative) Risk-Seeking')
ylabel('Estimated Risk-Seeking')
[Corr_rs_2, Sig_rs_2] = corrcoef(two_parameter_model_estimates.risk_seeking_true,two_parameter_model_estimates.risk_seeking_estimated);
text(4.1, 6.75, ['r = ' num2str(Corr_rs_2(1,2))])
text(4.1, 6.5, ['p = ' num2str(Sig_rs_2(1,2))])

figure
scatter(three_parameter_model_estimates.alpha_true,three_parameter_model_estimates.alpha_estimated,'filled')
lsline
title('Recoverability: Alpha (three-parameter model)')
xlabel('True (Generative) Alpha')
ylabel('Estimated Alpha')
[Corr_alpha_3, Sig_alpha_3] = corrcoef(three_parameter_model_estimates.alpha_true,three_parameter_model_estimates.alpha_estimated);
text(1, 29, ['r = ' num2str(Corr_alpha_3(1,2))])
text(1, 27, ['p = ' num2str(Sig_alpha_3(1,2))])

figure
scatter(three_parameter_model_estimates.risk_seeking_true,three_parameter_model_estimates.risk_seeking_estimated,'filled')
lsline
title('Recoverability: Risk-Seeking (three-parameter model)')
xlabel('True (Generative) Risk-Seeking')
ylabel('Estimated Risk-Seeking')
[Corr_rs_3, Sig_rs_3] = corrcoef(three_parameter_model_estimates.risk_seeking_true,three_parameter_model_estimates.risk_seeking_estimated);
text(4.1, 6.75, ['r = ' num2str(Corr_rs_3(1,2))])
text(4.1, 6.5, ['p = ' num2str(Sig_rs_3(1,2))])

figure
scatter(three_parameter_model_estimates.learning_rate_true,three_parameter_model_estimates.learning_rate_estimated,'filled')
lsline
title('Recoverability: Learning Rate (three-parameter model)')
xlabel('True (Generative) Learning Rate')
ylabel('Estimated Learning Rate')
[Corr_lr_3, Sig_lr_3] = corrcoef(three_parameter_model_estimates.learning_rate_true,three_parameter_model_estimates.learning_rate_estimated);
text(.25, .53, ['r = ' num2str(Corr_lr_3(1,2))])
text(.25, .52, ['p = ' num2str(Sig_lr_3(1,2))])


if PEB == 1
%% 10. 階層ベイズ (被験者間)
%==========================================================================
% 第2レベル分析のために保存されたGCMをクリアして再ロードします

% これにより、後でGCMデータを再ロードしてPEBを使用できるようになり、
% 'Sim = 5'オプションを再実行する必要がなくなります。

clear GCM_2
clear GCM_3
load('GCM_2.mat')
load('GCM_3.mat')
%==========================================================================

%--------------------------------------------------------------------------
% PEBを使用して、すべてのパラメータでグループ差を仮定する「完全な」モデルと、
% 1つ以上のパラメータで差がないと仮定する単純なモデルの
% エビデンスをテストできます。
%
% これにより、推定されたパラメータの差（または差がないこと）のエビデンスを
% テストできます。PEBは一般線形（ランダム効果）モデルを使用し、
% これにより個人差の効果（例：年齢、症状の重症度）のエビデンスも
% テストできます。

% これらのルーチンに関する関連文献を参照してください。例：
% Friston, Litvak, Oswal, Razi, Stephan, van Wijk, Ziegler, & Zeidman, 2016
% Zeidman, P., Jafarian, A., Seghier, M. L., Litvak, V., et al., 2019
%--------------------------------------------------------------------------

% 第2レベルモデル

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% まず、2パラメータモデルと3パラメータモデルのどちらを使用するかを指定します：

GCM_PEB = GCM_3; % GCM_2またはGCM_3のいずれか

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% デフォルトのPEBパラメータと被験者間モデル（M）を指定

M       = struct();
M.alpha = 1;       % 事前PEBパラメータ分散 = 1/alpha
M.beta  = 16;      % 被験者間変動の事前期待値（ランダム効果精度） = 1/(事前DCMパラメータ分散/M.beta)
M.hE    = 0;       % デフォルト
M.hC    = 1/16;    % デフォルト
M.Q     = 'all';   % 共分散成分: {'single','fields','all','none'}
M.X     = [];      % 一般線形モデルの計画行列

M.X = ones(length(GCM_PEB),1); % 一般線形モデルの最初の列は全参加者の平均

for i = 1:length(GCM_PEB)
    if i < (length(GCM_PEB)/2)+1 % このシミュレーションでは、グループ1はシミュレートされたサンプルの前半
        M.X(i,2) = 1;            % そしてグループ2は後半です
    else
        M.X(i,2) = -1;
    end
end

M.X(:,3) = 30 + 5.*randn(size(M.X,1),1); % 年齢の範囲をシミュレート（平均=30、SD=5）

M.X(:,2) = detrend(M.X(:,2),'constant'); % グループ値を0中心化
M.X(:,3) = detrend(M.X(:,3),'constant'); % 年齢値を0中心化

PEB_model  = spm_dcm_peb(GCM_PEB,M); % PEBモデルを指定
PEB_model.Xnames = {'Mean','Group','Age'}; % 共変量名を指定

[BMA,BMR] = spm_dcm_peb_bmc(PEB_model); % PEBモデルを推定

spm_dcm_peb_review(BMA,GCM_PEB); % 結果をレビュー

% 「第2レベル効果 - グループ」を選択すると、rsが
% グループ間で有意に異なることがわかります

% 結果の図をどのように解釈するかについての詳細は、
% 本文を参照してください

end

end

%==========================================================================
% これでチュートリアルスクリプトは完了です。これらのスクリプトを応用することで、
% タスクの生成モデルを構築し、シミュレーションを実行し、パラメータの
% 回復可能性を評価し、ベイジアンモデル比較を行い、階層的な
% ベイジアンのグループ分析を行うことができます。これらのステップの
% 他の側面に関するさらなる説明については、本文を参照してください。
