function [grad] = gradsvmclass4MKL(K,pi,pi2,C,yapp,option);

[n] = length(yapp);
m1=size(find(yapp==1),1);
m2=size(find(yapp==-1),1);


aa=pi(1:m1)-pi(m1+1:2*m1);
bb=pi(2*m1+1:2*m1+m2);

if ~isstruct(K)
    d=size(K,3);

    for k=1:d;
        grad(k)=-0.5*aa'*K(find(yapp==1),find(yapp==1),k)*aa...
        +aa'*K(find(yapp==1),find(yapp==-1),k)*bb...
        -0.5*bb'*K(find(yapp==-1),find(yapp==-1),1)*bb;
        %  grad(k) = - 0.5*Alpsup'*Kaux(indsup,indsup)*(Alpsup)  ;
%         grad(k) = - 0.5*Alpsup'*K(indsup,indsup,k)*(Alpsup)  ;
    end;

end


%负类计算方法同上述正类
yapp=-yapp;
[n] = length(yapp);
m1=size(find(yapp==1),1);
m2=size(find(yapp==-1),1);


aa=pi2(1:m1)-pi2(m1+1:2*m1);
bb=pi2(2*m1+1:2*m1+m2);

if ~isstruct(K)
    d=size(K,3);

    for k=1:d;
        grad2(k)=-0.5*aa'*K(find(yapp==1),find(yapp==1),k)*aa...
        +aa'*K(find(yapp==1),find(yapp==-1),k)*bb...
        -0.5*bb'*K(find(yapp==-1),find(yapp==-1),1)*bb;
        %  grad(k) = - 0.5*Alpsup'*Kaux(indsup,indsup)*(Alpsup)  ;
%         grad(k) = - 0.5*Alpsup'*K(indsup,indsup,k)*(Alpsup)  ;
    end;

end

% 正负单独算的时候，二者的grad计算方法不一样
% aa=pi2(1:m2)-pi2(m2+1:2*m2);
% bb=pi2(2*m2+1:2*m2+m1);
% 
% if ~isstruct(K)
%     d=size(K,3);
%     for k=1:d;
%         grad2(k)=-0.5*aa'*K(find(yapp==-1),find(yapp==-1),k)*aa...
%         -aa'*K(find(yapp==-1),find(yapp==1),k)*bb...
%         -0.5*bb'*K(find(yapp==1),find(yapp==1),1)*bb;
%         %  grad(k) = - 0.5*Alpsup'*Kaux(indsup,indsup)*(Alpsup)  ;
% %         grad(k) = - 0.5*Alpsup'*K(indsup,indsup,k)*(Alpsup)  ;
%     end;
% 
% end
grad=grad+grad2;

