% Example of how to use the mkl_npsvm for  classification
%
%

clear
close all
rand('state',-1);

nbiter=1;
ratio=0.8;
data='ionosphere.mat';
C = 256;
verbose=1;

options.algo='svmclass'; 
%------------------------------------------------------
% choosing the stopping criterion
% here, we only use the variation of weights for stopping criterion, and 
% other two methods need to be done
%------------------------------------------------------
options.stopvariation=1; % use variation of weights for stopping criterion 
options.stopKKT=0;       % set to 1 if you use KKTcondition for stopping criterion    
options.stopdualitygap=0; % set to 1 for using duality gap for stopping criterion

%------------------------------------------------------
% choosing the stopping criterion value
%------------------------------------------------------
options.seuildiffsigma=1e-3;        % stopping criterion for weight variation 
options.seuildiffconstraint=0.1;    % stopping criterion for KKT
options.seuildualitygap=0.01;       % stopping criterion for duality gap

%------------------------------------------------------
% Setting some numerical parameters 
%------------------------------------------------------
options.goldensearch_deltmax=1e-3; % initial precision of golden section search
options.numericalprecision=1e-3;   % numerical precision weights below this value
                                   % are set to zero 
options.lambdareg = 1e-8;          

%------------------------------------------------------
% some algorithms paramaters
%------------------------------------------------------
options.firstbasevariable='first'; % tie breaking method for choosing the base 
                                   % variable in the reduced gradient method 
options.nbitermax=500;             % maximal number of iteration  
options.seuil=0;                   % forcing to zero weights lower than this 
options.seuilitermax=10;           % value, for iterations lower than this one 

options.miniter=0;                 % minimal number of iterations 
options.verbosesvm=0;              % verbosity of inner svm algorithm 
options.efficientkernel=0;         % use efficient storage of kernels 

options.ee=0.01;                   % use for npsvm

%------------------------------------------------------------------------
%                   Building the kernels parameters
%------------------------------------------------------------------------

kernelt={'gaussian','poly'};
kerneloptionvect={[1/8 1/4  1/2 1 2 4 8 16 32 64 ] [1 2 3]};
variablevec={'all' 'all' };

classcode=[1 -1];
load([data ]);
x=mapminmax(x);
[nbdata,dim]=size(x);

nbtrain=floor(nbdata*ratio);


for i=1: nbiter
    [xapp,yapp,xtest,ytest,indice]=CreateDataAppTest(x, y, nbtrain,classcode);
    [xapp,xtest]=normalizemeanstd(xapp,xtest);

    [kernel,kerneloptionvec,variableveccell]=CreateKernelListWithVariable(variablevec,dim,kernelt,kerneloptionvect);
    [Weight,InfoKernel]=UnitTraceNormalization(xapp,kernel,kerneloptionvec,variableveccell);
    K=mklkernel(xapp,InfoKernel,Weight,options);
    
    tic
    [beta,w,w2,b,b2,story(i),delta_k,delta_k2,objective] = mklsvm(K,yapp,C,options,verbose);
    timelasso(i)=toc;
    
    m = length(yapp);
    m1 = size(find(yapp==1),1);
    m2 = size(find(yapp==-1),1);

    Kt=mklkernel(xtest,InfoKernel,Weight,options,xapp,beta);
    out1 = Kt(:,1:m1)*[w(1:m1)-w(m1+1:2*m1)] - Kt(:,m1+1:m)*w(2*m1+1:2*m1+m2)+b;
    out2 = Kt(:,m1+1:m)*[w2(1:m2)-w2(m2+1:2*m2)] - Kt(:,1:m1)*w2(2*m2+1:2*m2+m1)+b2;
    Y=[];
    for i=1:length(ytest)
        if (abs(out1(i)/delta_k)-abs(out2(i)/delta_k2)<0)
            Y(i) = 1;
        else
            Y(i) = -1;
        end
    end
    
    accuracy  = 1-length(find(ytest'-Y~=0))/length(ytest);

end



