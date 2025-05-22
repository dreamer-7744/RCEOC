%%%  experiment_SSKR: tr_data, te_data, rbf_bandwidth, lambda, max_label_size_ratio, max_iter %%%
function [testing_fitted_y] = LDOF_fit(train_data, test_data, k, dist_train)

dist_test_to_train = pdist2(test_data, train_data);
knn_set = knnsearch(train_data, test_data, 'K', k);

testing_fitted_y = zeros(size(test_data, 1), 1);
for i = 1:size(test_data, 1)
    
    dist_set = dist_test_to_train(i,:);
    sorted_dist_set = sort(dist_set);
    
    D_mean = mean(sorted_dist_set(1:k));
    D_scatter = sum(sum(dist_train(sort(knn_set(i,:)), sort(knn_set(i,:))))) / (2*k * (k-1));
    
    testing_fitted_y(i) = D_mean / (D_scatter);
    
end;



end