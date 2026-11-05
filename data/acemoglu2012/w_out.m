function [n_sectors, f_wo,xi_wo,wo_sorted,b_wo,bint_wo, rstats_wo,avg_slope_nw_wo, b_wo_gi,se_b_wo_gi ] =w_out(X,pct_cut)

%% Function w_out
% Used by weighted_outdegrees_final.m program
% Inputs:
% Outdegree Sequence and cut-off criteria 
% Outputs:
% number of sectors; non-parametric density of outdegree sequence and its average slope at the tail; ols
% tail estimates of its CCDF given the tail cutoff criteria; ols with
% Gabaix-Ibragimov correction
% August 2011
%%

wo=sum(X');
n_sectors=max(size(X));

% Weighted outdegrees density
[f_wo,xi_wo,u] = ksdensity(nonzeros(wo),'support',[0,max(wo)+2]);
[f_wo,xi_wo,u] = ksdensity(nonzeros(wo),'support',[0,max(wo)+2],'width',u*3);

% Weighted outdegrees, CCDF
wo_sorted=sort(wo,'descend');

% 20% cutoff OLS regression
cut=round(pct_cut*n_sectors);
[b_wo,bint_wo,r,rint,rstats_wo]=regress(log((1:1:cut)'), [ones(cut,1),log(wo_sorted(1:cut))']);

% 20% cutoff OLS regression with Gabaix-Ibragimov correction
cut=round(pct_cut*n_sectors);
[b_wo_gi,bint_wo_gi,r_gi,rint_gi,rstats_wo_gi]=regress(log((1-0.5:1:cut-0.5)'), [ones(cut,1),log(wo_sorted(1:cut))']);
[se_b_wo_gi]=(sqrt(2/cut))*b_wo_gi(2);


% Nadaraya-Watson nonparametric regression
a=log(wo_sorted(1:400));
b=log((1:1:400)./400);
r=ksr(a,b);

btemp=log(wo_sorted(cut));
temp_index=find(r.x>=btemp);

avg_slope_nw_wo=(r.f(end)-r.f(temp_index(1)))/(r.x(end)-r.x(temp_index(1)));


