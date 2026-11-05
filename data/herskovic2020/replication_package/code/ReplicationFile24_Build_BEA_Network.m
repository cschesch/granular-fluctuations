clear, addpath(genpath('functions\'))

%**************************************************************************
% Use I-O Table
%**************************************************************************

% The use table shows the uses of commodities by intermediate and final
% users. In contrast to the make table, the rows in the use table present the commodities
% or products, and the columns display the industries and final users that utilize
% them. The sum of the entries in a row is the output of that commodity. The columns
% show the products consumed by each industry and the three components of
% ?value added??compensation of employees, taxes on production and imports less
% subsidies, and gross operating surplus. Value added is the difference between an
% industry?s output and the cost of its intermediate inputs, and total value added is
% equal to GDP. The sum of the entries in a column is that industry?s output. 

for yr = 1997:2015

% I-O data
sheet = num2str(yr)
range = 'C8:BO72';
table = xlsread('../data/IOUse_After_Redefinitions_PRO_1997-2015_Summary.xlsx',sheet,range);
table = table(1:65,1:65);
table(isnan(table)) = 0;

range = 'C90:BO90';
output_ind   = xlsread('../data/IOUse_After_Redefinitions_PRO_1997-2015_Summary.xlsx',sheet,range);
output_ind   = output_ind';

range = 'CV8:CV72';
output_com   = xlsread('../data/IOUse_After_Redefinitions_PRO_1997-2015_Summary.xlsx',sheet,range);

eval(['IO_table.yr' num2str(yr) ' = table;'])
eval(['output_industry.yr' num2str(yr) ' = output_ind;'])
eval(['output_commodity.yr' num2str(yr) ' = output_com;'])
end
save(['../output/CleanData/BEA_1997_2015'],'IO_table','output_industry','output_commodity')
