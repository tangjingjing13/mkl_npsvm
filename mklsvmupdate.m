function [Sigma,Alpsup,Alpsup2,w0,w02,CostNew,delta_k,delta_k2] = mklsvmupdate(K,Sigma,Alpsup,Alpsup2,C,yapp,GradNew,CostNew,option)


%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%

d = length(Sigma);
gold = (sqrt(5)+1)/2 ;


SigmaInit = Sigma ;
SigmaNew  = SigmaInit ; 
descold = zeros(1,d);

%---------------------------------------------------------------
% Compute Current Cost and Gradient
%%--------------------------------------------------------------
% switch option.algo
%     case 'svmclass'
%       %  CostNew = costsvmclass(K,0,descold,SigmaNew,pos,Alpsup,C,yapp,option) ;
%       %  GradNew = gradsvmclass(K,pos,Alpsup,C,yapp,option) ;
%     case 'svmreg'
%       %  CostNew = costsvmreg(K,0,descold,SigmaNew,pos,Alpsup,C,yapp,option) ;
%       %  GradNew = gradsvmreg(K,Alpsup,yapp) ;
% end;

NormGrad = GradNew*GradNew';
GradNew=GradNew/sqrt(NormGrad);
CostOld=CostNew;
%---------------------------------------------------------------
% Compute reduced Gradient and descent direction
%%--------------------------------------------------------------

switch option.firstbasevariable
    case 'first'
        [val,coord] = max(SigmaNew) ;

    case 'random'
        [val,coord] = max(SigmaNew) ;
        coord=find(SigmaNew==val);
        indperm=randperm(length(coord));
        coord=coord(indperm(1));
    case 'fullrandom'
        indzero=find(SigmaNew~=0);
        if ~isempty(indzero)
        [mini,coord]=min(GradNew(indzero));
        coord=indzero(coord);
    else
        [val,coord] = max(SigmaNew) ;
    end;
        
end;
GradNew = GradNew - GradNew(coord) ;
desc = - GradNew.* ( (SigmaNew>0) | (GradNew<0) ) ;
desc(coord) = - sum(desc);  % NB:  GradNew(coord) = 0



%----------------------------------------------------
% Compute optimal stepsize
%-----------------------------------------------------
stepmin  = 0;
costmin  = CostOld ;
costmax  = 0 ;
%-----------------------------------------------------
% maximum stepsize
%-----------------------------------------------------
ind = find(desc<0);
stepmax = min(-(SigmaNew(ind))./desc(ind));
deltmax = stepmax;
if isempty(stepmax) | stepmax==0
    Sigma = SigmaNew ;
    w0=0;w02=0;delta_k=0;delta_k2=0;
    return
end,
if stepmax > 0.1
     stepmax=0.1;
end;

%-----------------------------------------------------
%  Projected gradient
%-----------------------------------------------------



while costmax<costmin;
    switch option.algo
        case 'svmclass'            
            [costmax,Alpsupaux,Alpsupaux2,w0aux,w0aux2,delta_k,delta_k2] = costsvmclass(K,stepmax,desc,SigmaNew,Alpsup,Alpsup2,C,yapp,option) ;
           
        case 'svmreg'
            [costmax,Alpsupaux,w0aux,posaux] = costsvmreg(K,stepmax,desc,SigmaNew,pos,Alpsup,C,yapp,option) ;
            
    end;
    if costmax<costmin
        costmin = costmax;
        SigmaNew  = SigmaNew + stepmax * desc;
%-------------------------------
%       Numerical cleaning
%-------------------------------
%       SigmaNew(find(abs(SigmaNew<option.numericalprecision)))=0;
%      SigmaNew=SigmaNew/sum(SigmaNew);
        % SigmaNew  =SigmaP;
        % project descent direction in the new admissible cone
        % keep the same direction of descent while cost decrease
        %desc = desc .* ( (SigmaNew>0) | (desc>0) ) ;
        desc = desc .* ( (SigmaNew>option.numericalprecision) | (desc>0) ) ;
        desc(coord) = - sum(desc([[1:coord-1] [coord+1:end]]));  
        ind = find(desc<0);
        Alpsup=Alpsupaux;
        Alpsup2=Alpsupaux2;
        w0=w0aux;
        w02=w0aux2;
        if ~isempty(ind)
            stepmax = min(-(SigmaNew(ind))./desc(ind));
            deltmax = stepmax;
            costmax = 0;
        else
            stepmax = 0;
            deltmax = 0;
        end;
        
    end;
end;


%-----------------------------------------------------
%  Linesearch
%-----------------------------------------------------

Step = [stepmin stepmax];
Cost = [costmin costmax];
[val,coord] = min(Cost);
% optimization of stepsize by golden search
while (stepmax-stepmin)>option.goldensearch_deltmax*(abs(deltmax))  & stepmax > eps;
    stepmedr = stepmin+(stepmax-stepmin)/gold;
    stepmedl = stepmin+(stepmedr-stepmin)/gold;                     
    switch option.algo
        case 'svmclass'
            [costmedr,Alpsupr,Alpsupr2,w0r,w0r2,delta_kr,delta_kr2] = costsvmclass(K,stepmedr,desc,SigmaNew,Alpsup,Alpsup2,C,yapp,option) ;
            [costmedl,Alpsupl,Alpsupl2,w01,w012,delta_kl,delta_kl2] = costsvmclass(K,stepmedl,desc,SigmaNew,Alpsupr,Alpsupr2,C,yapp,option) ;
        case 'svmreg'
            [costmedr,Alpsupr,w0r,posr] = costsvmreg(K,stepmedr,desc,SigmaNew,pos,Alpsup,C,yapp,option) ;
            [costmedl,Alpsupl,w01,posl] = costsvmreg(K,stepmedl,desc,SigmaNew,posr,Alpsupr,C,yapp,option) ;
            
    end;
    Step = [stepmin stepmedl stepmedr stepmax];
    Cost = [costmin costmedl costmedr costmax];
    [val,coord] = min(Cost);
    switch coord
        case 1
            stepmax = stepmedl;
            costmax = costmedl;
            Alpsup=Alpsupl;
            Alpsup2=Alpsupl2;
            w0=w01;w02=w012;delta_k=delta_kl;delta_k2=delta_kl2;
        case 2
            stepmax = stepmedr;
            costmax = costmedr;
            Alpsup=Alpsupr;
            Alpsup2=Alpsupr2;
            w0=w0r;w02=w0r2;delta_k=delta_kr;delta_k2=delta_kr2;
        case 3
            stepmin = stepmedl;
            costmin = costmedl;
            Alpsup=Alpsupl;
            Alpsup2=Alpsupl2;
            w0=w01;w02=w012;delta_k=delta_kl;delta_k2=delta_kl2;
        case 4
            stepmin = stepmedr;
            costmin = costmedr;
            Alpsup=Alpsupr;
            Alpsup2=Alpsupr2;
            w0=w0r;w02=w0r2;delta_k=delta_kr;delta_k2=delta_kr2;
    end;
end;


%---------------------------------
% Final Updates
%---------------------------------

CostNew = Cost(coord) ;
step = Step(coord) ;
% Sigma update
if CostNew < CostOld ;
    SigmaNew = SigmaNew + step * desc;  
    
end;       

Sigma = SigmaNew ;
