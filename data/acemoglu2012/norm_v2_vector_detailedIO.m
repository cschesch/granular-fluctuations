%% Norm Calculations for Detailed IO Matrices
% Input to code:
% BEA Commodity by Commodity Detailed Matrices for years 1972-2002
% Output of code:
% ||vnd||2: Norm of v2 vector for detailed IO matrices
% March 2012
%%

%% Load data
load CC_allyears_new.mat

%% Computations for each year
% Note: We normalize intermediate input shares to sum to one for every
% sector; for computations without normalization comment out loops and let
% X=CCYear

X=CC02;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC02=XX;
clear XX

X=CC97;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC97=XX;
clear XX

X=CC92;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC92=XX;
clear XX

X=CC87;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC87=XX;
clear XX

X=CC82;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC82=XX;
clear XX

X=CC77;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC77=XX;
clear XX

X=CC72;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC72=XX;
clear XX


%% Norm of v2 vector for detailed IO matrices
alpha=0.3333;
commodities=max(size(CC02));
W=CC02';
v2_02 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_02_2d=norm(v2_02);

alpha=0.3333;
commodities=max(size(CC97));
W=CC97';
v2_97 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_97_2d=norm(v2_97);

alpha=0.3333;
commodities=max(size(CC92));
W=CC92';
v2_92 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_92_2d=norm(v2_92);

alpha=0.3333;
commodities=max(size(CC87));
W=CC87';
v2_87 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_87_2d=norm(v2_87);

alpha=0.3333;
commodities=max(size(CC82));
W=CC82';
v2_82 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_82_2d=norm(v2_82);

alpha=0.3333;
commodities=max(size(CC77));
W=CC77';
v2_77 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_77_2d=norm(v2_77);

alpha=0.3333;
commodities=max(size(CC72));
W=CC72';
v2_72 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_72_2d=norm(v2_72);

%Output ||vnd||2
norm_all_nd=[norm_72_2d,norm_77_2d,norm_82_2d,norm_87_2d,norm_92_2d,norm_97_2d,norm_02_2d]