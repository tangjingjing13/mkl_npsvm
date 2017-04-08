function [nsvm, pi,bias,obj,delta_k] = EpsilonTSVM4MKLnew(K,yapp,option,C,alphainit)

epsilon = option.lambdareg;
ee=option.ee;
verbose=option.verbosesvm;
AA =[];  AB=[]; BB=[];
if(option.train=='p')
    
    m1 = size(find(yapp==1),1);
    m2 = size(find(yapp==-1),1);
    AA = K(find(yapp==1),find(yapp==1));
    AB = K(find(yapp==1),find(yapp==-1));
    BB = K(find(yapp==-1),find(yapp==-1));
else
   
    m1 = size(find(yapp==-1),1);
    m2 = size(find(yapp==1),1);
    AA = K(find(yapp==-1),find(yapp==-1));
    AB = K(find(yapp==-1),find(yapp==1));
    BB = K(find(yapp==1),find(yapp==1));
end

H1 = [AA, -AA;
    -AA, AA];
H2 = [AB;-AB];
H3 = BB;
H = [H1,-H2;
    -H2', H3];

H = H+1e-10*eye(size(H));   %[m,n] = size(X);


Aeq = [-ones(1,m1), ones(1,m1), ones(1,m2)];  
beq = 0;
lb = zeros(2*m1+m2,1);  
ub = C*ones(2*m1+m2,1); 
x0 = alphainit;
% x0 = zeros(2*m1+m2,1);   
f=[ee*(ones(1,m1)), ee*(ones(1,m1)), -ones(1,m2)]';

t = cputime;  
% options = optimset;
% options.LargeScale = 'off';
    
options = optimset('Algorithm','interior-point-convex','Display','off');
% options = optimset('Display','iter','TolFun',1e-32);
% options = optimset('LargeScale','off','MaxIter',1e4);
% options.MaxIter =1000;
[pi,fval,~,~,~]=quadprog(H,f,[],[],Aeq,beq,lb,ub,x0,options); 
obj=-fval;

delta_k=abs(fval-f'*pi);
if verbose ~= 0 
    fprintf('CPU time:%4.1f\n',cputime - t);
end
T=cputime - t;


nnsvm = find( pi> epsilon);
nsvm = length(nnsvm);
if verbose ~= 0 
    fprintf('nb of sv : %d %f\n',nsvm, nsvm/(m1+m2));
end


b=[]; h1=[];
for i=1:m1
    if(pi(i)>epsilon & pi(i)<(C-epsilon))
        for k=1:m1
            h1(k) = AA(k,i);
        end
        for g=1:m2
            temp=AB';
            h2(g) = temp(g,i);
        end
        b1 = h1*[pi(1:m1)-pi(m1+1:2*m1)] - h2*pi(2*m1+1:2*m1+m2);
        b1=-b1-ee;
        b=[b b1];
    end
end
if(isempty(b))
 for i=1:m1
    if(pi(i+m1)>epsilon & pi(i+m1)<(C-epsilon))
        for k=1:m1
            h1(k) = AA(k,i);
        end
        for g=1:m2
            temp=AB';
            h2(g) = temp(g,i);;
        end
        b1 = h1*[pi(1:m1)-pi(m1+1:2*m1)] - h2*pi(2*m1+1:2*m1+m2);
        b1=-b1+ee;
        b=[b b1];
    end
end   
end

bias=1/(length(b))*sum(b);

end
