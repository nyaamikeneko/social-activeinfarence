% --- このコードをコピーしてください ---

try
    % あなたのSPMがインストールされているフォルダのパスを指定してください
    % ※以前の情報から推測していますが、もし違っていたら修正してください
    spm_path = 'C:\Users\dmasu\Downloads\spm_25.01.02\spm';

    fprintf('Searching for demo files in %s ...\n', spm_path);

    % SPMの関数を使って 'DEM_'で始まり'.m'で終わるファイルを再帰的に検索
    file_list = spm_select('FPListRec', spm_path, '^DEM_.*\.m$');

    % --- ここからが出力部分 ---
    if isempty(file_list)
        fprintf('DEM_ で始まるファイルは見つかりませんでした。\n');
    else
        % 出力ファイル名を指定
        output_filename = 'spm_demo_list.txt';

        % ファイルを開いて書き込み
        fileID = fopen(output_filename, 'w');
        for i = 1:size(file_list, 1)
            fprintf(fileID, '%s\n', strtrim(file_list(i, :)));
        end
        fclose(fileID);

        % 完了メッセージを表示
        fprintf('\n=========================================================\n');
        fprintf('✅ 出力結果を %s に保存しました。\n', output_filename);
        fprintf('このスクリプトと同じフォルダにあるテキストファイルを開いて、\n');
        fprintf('中身をコピーしてください。\n');
        fprintf('=========================================================\n');
    end

catch ME
    fprintf('エラーが発生しました。\n');
    fprintf('SPMがOctaveのパスに正しく追加されているか確認してください。\n');
    fprintf('エラーメッセージ: %s\n', ME.message);
end

% --- ここまで ---
