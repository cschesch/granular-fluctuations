%% Weighted Indegrees: Data Summary & Graphs
% Input to code:
% NBER Manufacturing database 1958-2005; Sector TFP growth (5factor) 
% Output of code:
% Average Standard deviation of TFP growth with and without controlling for
% linear time trends in TFP growth (respectively STD_TFPg_RES and
% STD_TFPg_RAW)
% March 2012
%%

%% Load Raw NBER-CES Manufacturing data and select dtfp5, 5 factor TFP
%% growth
load NBER_Manuf_TFP_source_data.mat
%5 factor TFP growth
tfpg=data(:,21);
% Manipulate Matrix to get TFP growth in sector x year format
tfpg_mat(1,:)=tfpg(1:48);
for i=1:1:458
    tfpg_mat(i+1,:)=tfpg(i*48+1:48*(i+1));

end

%% Calculate Raw Standard deviation of TFP growth
for i=1:459
    std_tfpg(i)=nanstd(tfpg_mat(i,2:end));
end
% Mean across sectors
STD_TFPg_RAW=nanmean(std_tfpg)

%% Account for linear time trends in TFP growth
%Obtain residuals of regression of each sector's TFP growth on linear time trend.
linear_t=[1:(size(tfpg_mat,2)-1)];
for i=1:1:459
    [b,bint,r]=regress(tfpg_mat(i,2:end)',linear_t');
    res(i,:)=r';
end

%Calculate Standard deviation of residuals
for i=1:459
    std_tfp_res(i)=nanstd(res(i,1:end));
end

STD_TFPg_RES=nanmean(std_tfp_res)
