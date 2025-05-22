function [ensemble_testing_fitted_y] = RNEOC_test(train_data, test_data, ensemble_size, tmp_feat_set, tmp_k_set)

r = size(test_data,1);
ensemble_testing_fitted_y = zeros(r, ensemble_size);
for i = 1:ensemble_size
    
    tmp_feat = tmp_feat_set(i,:);
    k = tmp_k_set(i,1);
    
    shrinked_data = train_data(:,tmp_feat);
    shrinked_test_data = test_data(:,tmp_feat);
    
    dist_train = squareform(pdist(shrinked_data));
    [testing_fitted_y] = scaled_K2_fit(shrinked_data, shrinked_test_data, k, dist_train);
    ensemble_testing_fitted_y(:,i) = testing_fitted_y;
    
end;

end
