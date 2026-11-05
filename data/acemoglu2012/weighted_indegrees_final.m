%% Weighted Indegrees: Data Summary & Graphs
% Input to code:
% BEA Commodity by Commodity Matrices for years 1972-2002
% Output of code:
% Weighted Indegree Statistics & Graphs
% Uses w_ind function
% March 2012
%%

%% Set tail cutoff criterion
cut_off=0.2;

%% Load Data
load CC_allyears_new.mat

%% Computations for each year

X=CC02;
[mean_share_02,n_sectors_02,f_wi_02,xi_wi_02, u_wi_02,wi_02_sorted,b_wi_02,bint_wi_02, rstats_wi_02]=w_ind(X, cut_off);

X=CC97;
[mean_share_97,n_sectors_97,f_wi_97,xi_wi_97, u_wi_97,wi_97_sorted,b_wi_97,bint_wi_97, rstats_wi_97]=w_ind(X, cut_off);

X=CC92;
[mean_share_92,n_sectors_92,f_wi_92,xi_wi_92, u_wi_92,wi_92_sorted,b_wi_92,bint_wi_92, rstats_wi_92]=w_ind(X, cut_off);

X=CC87;
[mean_share_87,n_sectors_87,f_wi_87,xi_wi_87, u_wi_87,wi_87_sorted,b_wi_87,bint_wi_87, rstats_wi_87]=w_ind(X, cut_off);

X=CC82;
[mean_share_82,n_sectors_82,f_wi_82,xi_wi_82, u_wi_82,wi_82_sorted,b_wi_82,bint_wi_82, rstats_wi_82]=w_ind(X, cut_off);

X=CC77;
[mean_share_77,n_sectors_77,f_wi_77,xi_wi_77, u_wi_77,wi_77_sorted,b_wi_77,bint_wi_77, rstats_wi_77]=w_ind(X, cut_off);

X=CC72;
[mean_share_72,n_sectors_72,f_wi_72,xi_wi_72, u_wi_72,wi_72_sorted,b_wi_72,bint_wi_72, rstats_wi_72]=w_ind(X, cut_off);

%% Output Results
Mean_Input_Share=[mean_share_72,mean_share_77,mean_share_82,mean_share_87,mean_share_92,mean_share_97,mean_share_02]
Std_Input_Share= [std(wi_72_sorted),std(wi_77_sorted),std(wi_82_sorted),std(wi_87_sorted),std(wi_92_sorted),std(wi_97_sorted),std(wi_02_sorted)]
Bandwith=[ u_wi_72,u_wi_77, u_wi_82,u_wi_87, u_wi_92,u_wi_97, u_wi_02]

%Number/Percentage of Sectors within 1 std from mean input share
Pct_1std_mean_72=length(find(wi_72_sorted <= (1*std(wi_72_sorted)+mean(wi_72_sorted)) & wi_72_sorted >= (mean(wi_72_sorted)-1*std(wi_72_sorted))))./n_sectors_72;
Pct_1std_mean_77=length(find(wi_77_sorted <= (1*std(wi_77_sorted)+mean(wi_77_sorted)) & wi_77_sorted >= (mean(wi_77_sorted)-1*std(wi_77_sorted))))./n_sectors_77;
Pct_1std_mean_82=length(find(wi_82_sorted <= (1*std(wi_82_sorted)+mean(wi_82_sorted)) & wi_82_sorted >= (mean(wi_82_sorted)-1*std(wi_82_sorted))))./n_sectors_82;
Pct_1std_mean_87=length(find(wi_87_sorted <= (1*std(wi_87_sorted)+mean(wi_87_sorted)) & wi_87_sorted >= (mean(wi_87_sorted)-1*std(wi_87_sorted))))./n_sectors_87;
Pct_1std_mean_92=length(find(wi_92_sorted <= (1*std(wi_92_sorted)+mean(wi_92_sorted)) & wi_92_sorted >= (mean(wi_92_sorted)-1*std(wi_92_sorted))))./n_sectors_92;
Pct_1std_mean_97=length(find(wi_97_sorted <= (1*std(wi_97_sorted)+mean(wi_97_sorted)) & wi_97_sorted >= (mean(wi_97_sorted)-1*std(wi_97_sorted))))./n_sectors_97;
Pct_1std_mean_02=length(find(wi_02_sorted <= (1*std(wi_02_sorted)+mean(wi_02_sorted)) & wi_02_sorted >= (mean(wi_02_sorted)-1*std(wi_02_sorted))))./n_sectors_02;
Pct_1std_mean=[Pct_1std_mean_72,Pct_1std_mean_77,Pct_1std_mean_82,Pct_1std_mean_87,Pct_1std_mean_92,Pct_1std_mean_97,Pct_1std_mean_02]


%% Output Graphs
% 2 panel version, PDF only
subplot(1,2,1), plot(xi_wi_02,f_wi_02,'b.');
legend('2002','Location','NE');
xlabel('Weighted Indegree');
ylabel('Empirical Density');
axis([0 1 0 3])
axis square;
subplot(1,2,2,'align'),plot(xi_wi_72,f_wi_72,'bx',xi_wi_77,f_wi_77,'r+',xi_wi_82(1:end-1),f_wi_82(1:end-1),'r*',xi_wi_87(1:end-1),f_wi_87(1:end-1),'go',xi_wi_92,f_wi_92,'ms',xi_wi_97,f_wi_97,'b^', 'MarkerSize',3);
legend('1972','1977','1982','1987','1992','1997','2002','Location','NE');
xlabel('Weighted Indegree');
ylabel('Empirical Density');
axis([0 1 0 3])
axis square;

