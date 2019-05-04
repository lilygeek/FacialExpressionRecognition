% created by Priyanka Goenka
clc
clear
close all
load('dataCK.mat');
labels=data(:,end);
data=data(:,1:end-1);
%fourfold cross-validation of dataset.You can change it to fivefold
%crossvalidation.
testdata1=data(1:400,1:end);
traindata1=data(401:end,1:end);
testdata2=data(401:800,1:end);
traindata2=data(801:end,1:end);
testdata3=data(801:1200,1:end);
traindata3=data(1201:end,1:end);
testdata4=data(801:1200,1:end);
traindata4=data(1201:end,1:end);
ratio=0.1;% ratio between min and max eig value of scatter matrix.(user defined). For different ratio value we get different accuracy.
[w,v,e]=PCA(data,ratio);%PCA is applied for feature reduction
i=1;
while e(i)/max(e)>ratio
    i=i+1;
end
i=i-1;
vnew=v(:,1:i);
data=data*w;%data after feture reduction
testdata=data(1:400,:);
ytest=labels(1:400,:);
traindata=data(401:end,1:end);
ytrain=labels(401:end,1:end);
ypredicted=zeros(size(testdata,1),1);
k=1;% Positive interger (neighbor). Value of k will vary in kNN classifier. For different values of k we get different results. 
for i = 1:size(testdata,1)
    ypredicted(i)=knn(testdata(i,:),traindata,ytrain,k);%KNN classification
end
confumat=confusionmat(ytest,ypredicted);
accuracy=sum(diag(confumat))/size(testdata,1)*100;%misclassified/total*100
