%% function for normalizing 'a' to get likelihood matrix 'A'
function A_normed = col_norm(A_norm)
aa = A_norm;
norm_constant = sum(aa,1); % create normalizing constant from sum of columns
aa = aa./norm_constant; % divide columns by constant
A_normed = aa;
end
