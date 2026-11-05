function [n_sectors, f_wo2,xi_wo2,wo2_sorted,b_wo2,bint_wo2, rstats_wo2,avg_slope_nw_wo2, b_wo_gi,se_b_wo_gi] =w_out_2ord(X,pct_cut)

%% Function w_out_2ord
% Used by weighted_outdegrees_2ndorder_final.m program
% Inputs:
% Outdegree Sequence and cut-off criteria 
% Outputs:
% number of sectors; non-parametric density of 2nd order outdegree sequence and its average slope at the tail; ols
% tail estimates of its CCDF given the tail cutoff criteria; ols with
% Gabaix-Ibragimov correction
% August 2011
%%

wo=sum(X');

for i=1:max(size(wo))
    wij(i,:)=X(i,:);
    wo2(i)=(wij(i,:)*wo');
end

n_sectors=max(size(X));

% Weighted outdegrees, 2002, density
[f_wo2,xi_wo2,u] = ksdensity(nonzeros(wo2),'support',[0,max(wo2)+2]);

% Weighted outdegrees, 2002, CCDF
wo2_sorted=sort(wo2,'descend');

% 20% cutoff OLS regression
cut=round(pct_cut*n_sectors);
[b_wo2,bint_wo2,r,rint,rstats_wo2]=regress(log((1:1:cut)'), [ones(cut,1),log(wo2_sorted(1:cut))']);

% 20% cutoff OLS regression with Gabaix Ibragimov Correction
cut=round(pct_cut*n_sectors);
[b_wo_gi,bint_wo_gi,r_gi,rint_gi,rstats_wo_gi]=regress(log((1-0.5:1:cut-0.5)'), [ones(cut,1),log(wo2_sorted(1:cut))']);
[se_b_wo_gi]=(sqrt(2/cut))*b_wo_gi(2);

%Nadaraya-Watson nonparametric regression
a=log(wo2_sorted(1:398));
b=log((1:1:398)./398);
r=ksr(a,b);

btemp=log(wo2_sorted(cut));
temp_index=find(r.x>=btemp);

avg_slope_nw_wo2=(r.f(end)-r.f(temp_index(1)))/(r.x(end)-r.x(temp_index(1)));



