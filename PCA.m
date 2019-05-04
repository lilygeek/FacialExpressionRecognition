function [vreduced,v,c]=PCA(D,ratio)
% created by Priyanka Goenka
% D is N x p dataset;
%ratio is the eig_min/eig_max of scatter matrix that the user chooses
m=mean(D);
m=m';
S=zeros(size(D,2));
size(S)
for i = 1:length(D)
    S=S+(D(i,:)'-m)*(D(i,:)'-m)';%scatter matrix
end
[v,e]=eig(S);
[c, ind]=sort(diag(abs(e)),'descend');%sorting eig values in decending order
v=v(:,ind);% eig vectors corresponding to sorted eig values
i=1;
while c(i)/max(c)>ratio
    i=i+1;
end
i=i-1;
vreduced=v(:,1:i);%priciple components


