max_cross_iter = 5;
K = 5;
kmax = 20;

data1 = dlmread('C:\Users\user\Desktop\MyWork\data\논문에 쓴 데이터\wine.csv');
[final_AUC_result_all_cl] = Repeated_5foldCV_experiement_RNEOC(data1, max_cross_iter, K, kmax);