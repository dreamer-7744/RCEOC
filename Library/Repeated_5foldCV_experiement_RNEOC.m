function [final_AUC_result_all_cl] = Repeated_5foldCV_experiement_RNEOC(data1, max_cross_iter, K, kmax)

[r,p] = size(data1);

data = data1(:,1:(p-1));
class = data1(:,p);

unique_class_size = length(unique(class));
class_index = (1:unique_class_size);

init_y = zeros(r, unique_class_size);
j = 1;
for i = 1:unique_class_size
    init_y(class == i,i) = 1;
end;


final_AUC_result_all_cl = zeros(unique_class_size, max_cross_iter);
for cl_ind = 1:unique_class_size
    
    target_index = cl_ind;
    anomaly_ind = class_index;
    anomaly_ind(target_index) = [];
    
    target_data = [];
    target_class = [];
    for i = 1:length(target_index)
        target_data = [target_data;data(class == target_index(i), :)];
        target_class = [target_class;init_y(class == target_index(i), :)];
    end;
    
    anomaly_data = [];
    anomaly_class = [];
    for i = 1:length(anomaly_ind)
        anomaly_data = [anomaly_data;data(class == anomaly_ind(i), :)];
        anomaly_class = [anomaly_class;init_y(class == anomaly_ind(i), :)];
    end;
    
    final_AUC_result_mat = zeros(1,max_cross_iter);
    for cross_iter = 1:max_cross_iter
        
        tmp_final_AUC_result_mat = zeros(K, 1);
        
        cross_valind = crossvalind('Kfold', size(target_data,1), K);
        for cross_ind = 1:K
            
            target_train_data = target_data(cross_valind ~= cross_ind, :);
            target_train_class = target_class(cross_valind ~= cross_ind, :);
            
            target_test_data = target_data(cross_valind == cross_ind, :);
            target_test_class = target_class(cross_valind == cross_ind, :);
            
            train_data1 = target_train_data;
            
            test_data1 = [target_test_data; anomaly_data];
            test_class = [target_test_class; anomaly_class];
            
            
            data1_mean = mean(train_data1);
            data1_std = std(train_data1);
            train_data = zeros(size(train_data1,1),p-1);
            for i = 1:size(train_data1,1)
                train_data(i,:) = (train_data1(i,:) - data1_mean) ./ data1_std ;
            end;
            train_data = train_data(:,data1_std~=0);
            
            
            test_data = zeros(size(test_data1,1),p-1);
            for i = 1:size(test_data1,1)
                test_data(i,:) = (test_data1(i,:) - data1_mean) ./ data1_std ;
            end;
            test_data = test_data(:,data1_std~=0);
            
            ensemble_size = 100;
            [tmp_feat_set, tmp_k_set] = RNEOC_train(train_data, ensemble_size, kmax);
            [testing_fitted_y1] = RNEOC_test(train_data, test_data, ensemble_size, tmp_feat_set, tmp_k_set);
            
            testing_fitted_y1(testing_fitted_y1 == Inf) = "NaN";
            testing_fitted_y = mean(testing_fitted_y1', "omitnan");
            
            [~,~,~,AUC] = perfcurve(1-test_class(:,target_index), testing_fitted_y, 1);
            
            tmp_final_AUC_result_mat(cross_ind,1) = AUC;
        end

        final_AUC_result_mat(1,cross_iter) = mean(tmp_final_AUC_result_mat);
    end
    
    final_AUC_result_all_cl(cl_ind,:) = final_AUC_result_mat;
    
end

end