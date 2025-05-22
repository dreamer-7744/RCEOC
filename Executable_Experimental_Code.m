% This MATLAB code is for executable code for repeated 5 fold cross validation of RCE-OC algorithm. The users can change apply different datasets for the experiment by chaing the path of dataset %

max_cross_iter = 5;
K = 5;
kmax = 20;

data1 = dlmread('C:\Users\user\Desktop\Codes and Datasets\Datasets\wine.csv');
[final_AUC_result_all_cl] = Repeated_5foldCV_experiement_RNEOC(data1, max_cross_iter, K, kmax);
