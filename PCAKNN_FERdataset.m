% created by Priyanka Goenka
clc
clear
close all
%fourfold cross-validation of dataset.You can change it to fivefold
%crossvalidation.
load('dataFER.mat');
data=a3;
labels=data(:,end);
data=data(:,1:end-1);
ratio=0.01;% ratio between min and max eig value of scatter matrix.(user defined). For different ratio value we get different accuracy.
[w,v,e]=PCA(data,ratio);%PCA is applied for feature reduction
numCores = 4;%specify based on your PC spec.
if isempty(gcp('nocreate'))%parallel processing
    parpool(numCores)
end
i=i-1;
vnew=v(:,1:i);
data=data*w;%data after feture reduction
testdata=data(1:15000,:);
ytest=labels(1:15000,:);
traindata=data(15001:end,1:end);
ytrain=labels(15001:end,1:end);
ypredicted=zeros(size(testdata,1),1);
k=50;% Positive interger (neighbor). Value of k will vary in kNN classifier. For different values of k we get different results. 
parfor i = 1:size(testdata,1)
    ypredicted(i)=knn(testdata(i,:),traindata,ytrain,k);;%KNN classification
end
confumat=confusionmat(ytest,ypredicted);
accuracy=sum(diag(confumat))/size(testdata,1)*100;%misclassified/total*100
