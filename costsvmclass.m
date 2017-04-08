function [cost,Alpsupaux,Alpsupaux2,w0aux,w0aux2,delta_k,delta_k2] = costsvmclass(K,StepSigma,DirSigma,Sigma,Alpsup,Alpsup2,C,yapp,option)


global nbcall
nbcall=nbcall+1;

[n]=length(yapp);

Sigma = Sigma+ StepSigma * DirSigma;
kerneloption.matrix=sumKbeta(K,Sigma);
% kernel='numerical';
% span=1;
% lambdareg=option.lambdareg;
% verbosesvm=option.verbosesvm;
m1 = size(find(yapp==1),1);
m2 = size(find(yapp==-1),1);
% alphainit=zeros(2*m1+m2,1);
% temp=[ones(m1,1);yapp];


% alphainit=zeros(size(yapp));
% alphainit(indsup)=yapp(indsup).*Alpsup;
% [xsup,Alpsupaux,w0aux,posaux,timeps,alpha,cost] = svmclass([],yapp,C,lambdareg,kernel,kerneloption,verbosesvm,span,alphainit);
option.train='p';  
alphainit=Alpsup;
[nsvm, pi ,bias,obj,delta_k] = EpsilonTSVM4MKLnew(kerneloption.matrix,yapp,option,C,alphainit);
option.train='n';
alphainit=Alpsup2;
[nsvm2, pi2 ,bias2,obj2,delta_k2] = EpsilonTSVM4MKLnew(kerneloption.matrix,yapp,option,C,alphainit);

cost=obj+obj2;
Alpsupaux=pi;
Alpsupaux2=pi2;
w0aux=bias;
w0aux2=bias2;
