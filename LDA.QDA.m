clc
close all
clear

%-------
% mixture models
%-------
num_clusters=5;
Bcentr=randn(2,num_clusters); % locations of centers of clusters for green class
Rcentr=randn(2,num_clusters); % -"- red class

%-------
% training data
%-------
N=200; % total number of data samples
samples=zeros(2,N); % locations of samples in 2 dimensions
class_samples=zeros(N,1); % class of each one (green or red)
cluster_variance=0.1; % variance of each cluster around its mean
for n=1:N/2
    Bcluster=ceil(rand(1)*num_clusters); % select green cluster
    Rcluster=ceil(rand(1)*num_clusters); % -"- red
    samples(:,n)=Bcentr(:,Bcluster)+sqrt(cluster_variance)*randn(2,1); % generate green sample
    samples(:,n+N/2)=Rcentr(:,Rcluster)+sqrt(cluster_variance)*randn(2,1); % -"- red
    class_samples(n)=1; % green
    class_samples(n+N/2)=0; % red
end

%-------
% test data - basically a 2-D grid
%-------
grid=-3:0.1:3; % will scan along this grid in each dimension
test_samples=zeros(2,length(grid)^2); % locations of test samples
for n1=1:length(grid)
    for n2=1:length(grid)
        test_samples(1,n1+length(grid)*(n2-1))=grid(n1); % first coordinate x
        test_samples(2,n1+length(grid)*(n2-1))=grid(n2); % second y
    end
end

%-------
% nearest neighbors
%-------
num_neighbors=5; % number of neighbors used
test_NN=zeros(length(grid)^2,1); % classification results on test data
for n1=1:length(grid)
    for n2=1:length(grid)
        distances=(grid(n1)-samples(1,:)).^2+(grid(n2)-samples(2,:)).^2; % distances to training samples
        [distances_sort,distances_index]=sort(distances);
        neighbors=distances_index(1:num_neighbors);
        class_predicted=(sum(class_samples(neighbors))/num_neighbors>0.5); % NN classifier
        test_NN(n1+length(grid)*(n2-1))=class_predicted; % store classification
    end
end

% identify location indices (in test grid) that are red and green
r_locations=find(test_NN==0);
g_locations=find(test_NN==1);


%We consider 2 classes Blue and Red

Blue=samples(:,1:N/2)';
Red=samples(:,N/2+1:N)';
datapoints=samples';

%mean of all datapoints
Tmean=mean(datapoints);

%variance
Tc=cov(datapoints);

%mean of datapoints of each of the 2 classes
Bmean= mean(Blue);
Rmean=mean(Red);

%Covariances for the two classes
Bc=cov(Blue);
Rc=cov(Red);

rng(1);
X = [mvnrnd(Bmean,Tc,1000);mvnrnd(Rmean,Tc,1000)];
obj=fitgmdist(X,2);

%Plotting Probability Density Function
scatter(X(:,1),X(:,2),10,'.')
hold on
h = ezcontour(@(x,y)pdf(obj,[x y]),[-3 3],[-3 3]);

%posterior probability will be P(Blue/samples(i)) proportional to
%P(samples(i)/Blue)*P(Blue)

%Posterior Probability for LDA

P=posterior(obj,X);
scatter(X(:,1),X(:,2),10,P(:,1));
hb = colorbar;
ylabel(hb,'Blue Probability')
title('PDF and Posterior Probability for LDA');
%Finding Decision Boundary with Bayesian Classification
%decision boundary would be the solution of g1x=g2x
%Linear Discriminant Analysis
syms x1 x2;
g1 = -0.5*([x1;x2]-Bmean')'*(Tc\([x1;x2]-Bmean')-log(2*pi)-0.5*log(det(Tc)));
g2 = -0.5*([x1;x2]-Rmean')'*(Tc\([x1;x2]-Rmean')-log(2*pi)-0.5*log(det(Tc)));

g=g1-g2;
figure
ezplot(g,[[-3,3],[-3,3]]);
hold on
plot(samples(1,1:N/2),samples(2,1:N/2),'b*',...
    samples(1,N/2+1:N),samples(2,N/2+1:N),'ro',...  
    test_samples(1,g_locations),test_samples(2,g_locations),'b.',... 
    test_samples(1,r_locations),test_samples(2,r_locations),'r.')
   
title('Bayesian Decision Regions with LDA');

%Quadratic Discriminant Ananlysis

rng(1);
X = [mvnrnd(Bmean,Bc,1000);mvnrnd(Rmean,Rc,1000)];
obj=fitgmdist(X,2);

%Plotting Probability Density Function
figure
scatter(X(:,1),X(:,2),10,'.')
hold on
h = ezcontour(@(x,y)pdf(obj,[x y]),[-3 3],[-3 3]);

%Posterior Probability for QDA

P=posterior(obj,X);
scatter(X(:,1),X(:,2),10,P(:,1));
hb = colorbar;
ylabel(hb,'Blue Probability')
title('PDF and Posterior Probability for QDA');

%Using 2 different covariances
syms B R;
g1 = -0.5*([B;R]-Bmean')'*(Bc\([B;R]-Bmean')-log(2*pi)-0.5*log(det(Bc)));
g2 = -0.5*([B;R]-Rmean')'*(Rc\([B;R]-Rmean')-log(2*pi)-0.5*log(det(Rc)));

g=g1-g2;
figure
ezplot(g,[[-3,3],[-3,3]]);
hold on
plot(samples(1,1:N/2),samples(2,1:N/2),'b*',... 
    samples(1,N/2+1:N),samples(2,N/2+1:N),'ro',...
    test_samples(1,g_locations),test_samples(2,g_locations),'b.',...
    test_samples(1,r_locations),test_samples(2,r_locations),'r.')
   
title('Bayesian Decision Regions with QDA');
