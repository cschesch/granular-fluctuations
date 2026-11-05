%% 2nd Order Weighted Outdegrees Graphs
% Input to code:
% BEA Commodity by Commodity Matrices for years 1972-2002
% Output of code:
% First Order Weighted Outdegree Statistics & Graphs
% Uses: 
% 1) w_out_2ord function
% 2) ksr.m function [Copyright (c) 2009, Yi Cao] from Matlab FileExchange
% March 2012
%%

%% Set tail cutoff criterion
cut_off=0.2;

%% Load Data
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
X=XX;
clear XX;

[n_sectors_02,f_wo_02,xi_wo_02,wo_02_sorted,b_wo_02,bint_wo_02, rstats_wo_02,avg_slope_nw_wo_02,b_wo_gi_02,se_b_wo_gi_02]=w_out_2ord(X, cut_off);

X=CC97;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
X=XX;
clear XX;

[n_sectors_97,f_wo_97,xi_wo_97,wo_97_sorted,b_wo_97,bint_wo_97, rstats_wo_97,avg_slope_nw_wo_97,b_wo_gi_97,se_b_wo_gi_97]=w_out_2ord(X, cut_off);

X=CC92;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
X=XX;
clear XX;

[n_sectors_92,f_wo_92,xi_wo_92,wo_92_sorted,b_wo_92,bint_wo_92, rstats_wo_92,avg_slope_nw_wo_92,b_wo_gi_92,se_b_wo_gi_92]=w_out_2ord(X, cut_off);

X=CC87;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
X=XX;
clear XX;

[n_sectors_87,f_wo_87,xi_wo_87,wo_87_sorted,b_wo_87,bint_wo_87, rstats_wo_87,avg_slope_nw_wo_87,b_wo_gi_87,se_b_wo_gi_87]=w_out_2ord(X, cut_off);

X=CC82;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
X=XX;
clear XX;

[n_sectors_82,f_wo_82,xi_wo_82,wo_82_sorted,b_wo_82,bint_wo_82, rstats_wo_82,avg_slope_nw_wo_82,b_wo_gi_82,se_b_wo_gi_82]=w_out_2ord(X, cut_off);

X=CC77;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
X=XX;
clear XX;

[n_sectors_77,f_wo_77,xi_wo_77,wo_77_sorted,b_wo_77,bint_wo_77, rstats_wo_77,avg_slope_nw_wo_77,b_wo_gi_77,se_b_wo_gi_77]=w_out_2ord(X, cut_off);

X=CC72;
for i=1:max(size(X))
    if sum(X(:,i))>0
        XX(:,i)=X(:,i)./sum(X(:,i));
    elseif sum(X(:,i))==0
        XX(:,i)=X(:,i);
    end
end
X=XX;
clear XX;
[n_sectors_72,f_wo_72,xi_wo_72,wo_72_sorted,b_wo_72,bint_wo_72, rstats_wo_72,avg_slope_nw_wo_72,b_wo_gi_72,se_b_wo_gi_72]=w_out_2ord(X, cut_off);

%% Output Results
% OLS tail estimates of ZETA_HAT and standard error as per Gabaix-Ibragimov
ols_gi_outd2=[b_wo_gi_72(2),b_wo_gi_77(2),b_wo_gi_82(2),b_wo_gi_87(2),b_wo_gi_92(2),b_wo_gi_97(2),b_wo_gi_02(2);
              se_b_wo_gi_72,se_b_wo_gi_77,se_b_wo_gi_82,se_b_wo_gi_87,se_b_wo_gi_92,se_b_wo_gi_97,se_b_wo_gi_02]

% Total number of sectors
n_sectors_outd2=[n_sectors_72,n_sectors_77,n_sectors_82,n_sectors_87,n_sectors_92,n_sectors_97,n_sectors_02]

% Nadaraya-Watson nonparametric regression: average slope
nwslope_outd2=[avg_slope_nw_wo_72,avg_slope_nw_wo_77,avg_slope_nw_wo_82,avg_slope_nw_wo_87,avg_slope_nw_wo_92,avg_slope_nw_wo_97,avg_slope_nw_wo_02]

%% MLE Robustness check: calls Clauset, Shalizi and Newman (2009) codes
%%Please refer to readme file for instructions
%%MLE -Hill tail parameter with endogenous cutoff
% [alpha_MLE_72, xmin_72]=plfit(nonzeros(wo_72_sorted),'finite');
% ntail_72=length(find(wo_72_sorted>xmin_72));
% std_alpha_72=(alpha_MLE_72-1)/sqrt(ntail_72);
% [alpha_MLE_77, xmin_77]=plfit(nonzeros(wo_77_sorted),'finite');
% ntail_77=length(find(wo_77_sorted>xmin_77));
% std_alpha_77=(alpha_MLE_77-1)/sqrt(ntail_77);
% [alpha_MLE_82, xmin_82]=plfit(nonzeros(wo_82_sorted),'finite');
% ntail_82=length(find(wo_82_sorted>xmin_82));
% std_alpha_82=(alpha_MLE_82-1)/sqrt(ntail_82);
% [alpha_MLE_87, xmin_87]=plfit(nonzeros(wo_87_sorted),'finite');
% ntail_87=length(find(wo_87_sorted>xmin_87));
% std_alpha_87=(alpha_MLE_87-1)/sqrt(ntail_87);
% [alpha_MLE_92, xmin_92]=plfit(nonzeros(wo_92_sorted),'finite');
% ntail_92=length(find(wo_92_sorted>xmin_92));
% std_alpha_92=(alpha_MLE_92-1)/sqrt(ntail_92);
% [alpha_MLE_97, xmin_97]=plfit(nonzeros(wo_97_sorted),'xmin',1.5);
% ntail_97=length(find(wo_97_sorted>xmin_97));
% std_alpha_97=(alpha_MLE_97-1)/sqrt(ntail_97);
% [alpha_MLE_02, xmin_02]=plfit(nonzeros(wo_02_sorted),'nosmall');
% ntail_02=length(find(wo_02_sorted>xmin_02));
% std_alpha_02=(alpha_MLE_02-1)/sqrt(ntail_02);
% 
%%Output MLE results
% mle_outd2=[alpha_MLE_72-1,alpha_MLE_77-1,alpha_MLE_82-1,alpha_MLE_87-1,alpha_MLE_92-1,alpha_MLE_97-1,alpha_MLE_02-1;
%            std_alpha_72,std_alpha_77,std_alpha_82,std_alpha_87,std_alpha_92,std_alpha_97,std_alpha_02;
%            ntail_72,ntail_77,ntail_82,ntail_87,ntail_92,ntail_97,ntail_02]
% 

%% Output Graphs

%2 panel version with non parametric on LHS
a=log(wo_02_sorted(1:398));
b=log((1:1:398)./398);
r=ksr(a,b);
subplot(1,2,1), loglog(exp(a),exp(b),'o',exp(r.x),exp(r.f),'r-','MarkerSize',3,'linewidth',2);
legend('2002','Location','NE');
axis([0.000001 100 0 1])
xlabel('Second Order Weighted Outdegree');
ylabel('Empirical CCDF');
axis square;
subplot(1,2,2,'align'),loglog(wo_72_sorted,(1:1:n_sectors_72)./n_sectors_72,'bx',wo_77_sorted,(1:1:n_sectors_77)./n_sectors_77,'r+',wo_82_sorted,(1:1:n_sectors_82)./n_sectors_82,'k*',wo_87_sorted,(1:1:n_sectors_87)./n_sectors_87,'go',wo_92_sorted,(1:1:n_sectors_92)./n_sectors_92,'ms',wo_97_sorted,(1:1:n_sectors_97)./n_sectors_97,'b^', 'MarkerSize',3);
axis([0.000001 100 0 1])
xlabel('Second Order Weighted Outdegree');
ylabel('Empirical CCDF');
legend('1972','1977','1982','1987','1992','1997','2002','Location','NE');
axis square;

