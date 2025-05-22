function [tmp_feat_set, tmp_k_set] = RNEOC_train(train_data, ensemble_size, kmax)

[~, c] = size(train_data);

min_k = 2;
max_k = kmax;

tmp_feat_set = zeros(ensemble_size, ceil(c/2));
tmp_k_set = zeros(ensemble_size, 1);
for i = 1:ensemble_size
    
    tmp_feat_set(i,:) = sort(randsample(c, ceil(c/2)));
    tmp_k_set(i,1) = randsample((min_k:max_k), 1);
    
end;

end