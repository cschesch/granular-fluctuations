%% Norm Calculations for Summary IO Matrices
% Input to code:
% BEA Commodity by Commodity Summary Matrices for years 1972-2002
% Output of code:
% ||vns||2: Norm of v2 vector for summary IO matrices
% March 2012
%%

%% Load data
load CC_2d_allyears_new.mat

%% Computations for each year
% Note: We normalize intermediate input shares to sum to one for every
% sector; for computations without normalization comment out loops and let
% X=CCYear

X=CC02_2d;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC02_2dn=XX;
clear XX

X=CC97_2d;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC97_2dn=XX;
clear XX

X=CC92_2d;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC92_2dn=XX;
clear XX

X=CC87_2d;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC87_2dn=XX;
clear XX

X=CC82_2d;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC82_2dn=XX;
clear XX

X=CC77_2d;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC77_2dn=XX;
clear XX

X=CC72_2d;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
CC72_2dn=XX;
clear XX


%% Norm of v2 vector for summary IO matrices
alpha=0.3333;
commodities=max(size(CC02_2d));
W=CC02_2dn';
v2_02 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_02_2d=norm(v2_02);

alpha=0.3333;
commodities=max(size(CC97_2d));
W=CC97_2dn';
v2_97 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_97_2d=norm(v2_97);

alpha=0.3333;
commodities=max(size(CC92_2d));
W=CC92_2dn';
v2_92 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_92_2d=norm(v2_92);

alpha=0.3333;
commodities=max(size(CC87_2d));
W=CC87_2dn';
v2_87 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_87_2d=norm(v2_87);

alpha=0.3333;
commodities=max(size(CC82_2d));
W=CC82_2dn';
v2_82 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_82_2d=norm(v2_82);

alpha=0.3333;
commodities=max(size(CC77_2d));
W=CC77_2dn';
v2_77 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_77_2d=norm(v2_77);

alpha=0.3333;
commodities=max(size(CC72_2d));
W=CC72_2dn';
v2_72 = (alpha/commodities)*inv(eye(commodities)-(1-alpha)*W')*ones(commodities,1); 
norm_72_2d=norm(v2_72);

%Output ||vns||2
norm_all_nd2=[norm_72_2d,norm_77_2d,norm_82_2d,norm_87_2d,norm_92_2d,norm_97_2d,norm_02_2d]