function y=knn(input,D,Y,k)
% created by Priyanka Goenka
Y=Y';
%Y is label vector;
%data without labels;
%k no of nearest neighbors
%y predicted label
distance=pdist2(input,D,'euclidean');
[distance, in]=sort(distance,'ascend');%sort neighbors
Y=Y(in);
[M,F]=mode(Y(1,1:k));
if F==1 && k~=1
    msg = 'Change k';% if k is not sufficient i.e, same no of nearest neighbors for more than one class
    error(msg)
else
    y=M;%predicted label
end
