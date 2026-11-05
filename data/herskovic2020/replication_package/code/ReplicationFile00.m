clear, clc, close all
% This file replicates all tables and figures in 
% "Firm Volatility in Granular Networks" (Journal of Political Economy)
% by Bernard Herskovic, Bryan Kelly, Hanno Lustig, and Stijn Van Nieuwerburgh

%% Creating Directories to save results
try rmdir('../output', 's'),  catch, end
try rmdir('../figures', 's'), catch, end
try rmdir('../tables', 's'),  catch, end
mkdir('../output/CleanData')           % Create empty folder to save clean data used in the empirics
mkdir('../output/EstimationResults')   % Create empty folder to save smm estimation results
mkdir('../output/FigureA1simulations') % Create empty folder to save simulated data used in Figure A1
mkdir('../output/TSFit_Results')       % Create empty folder to save simulated data used in Table 5
mkdir('../figures')                    % Create empty folder to save all figures used in the paper
mkdir('../tables')                     % Create empty folder to save all tables used in the paper

%% Data analysis
disp('ReplicationFile01'), ReplicationFile01_data_Build_Daily_Return_Matrix, clear
disp('ReplicationFile02'), ReplicationFile02_data_Construct_Annual_Stock_RV_Daily_CRSP, clear
disp('ReplicationFile03'), ReplicationFile03_data_Construct_ME_EXCHCD_SIC_Monthly_CRSP, clear
disp('ReplicationFile04'), ReplicationFile04_data_Construct_Annual_Sales_Qrt_Compustat, clear
disp('ReplicationFile05'), ReplicationFile05_data_BuildLinkageMatrices, clear
disp('ReplicationFile06'), ReplicationFile06_data_MergeDatasets, clear
disp('ReplicationFile07'), ReplicationFile07_data_Figures1and2, clear                            % Figures 1 and 2
disp('ReplicationFile08'), ReplicationFile08_data_Table1, clear                                  % Table 1
disp('ReplicationFile09'), ReplicationFile09_data_Build_Matrices_PanelRegressions, clear
disp('ReplicationFile10'), ReplicationFile10_data_Build_Cash_Flow_Growth_Data, clear
disp('ReplicationFile11'), ReplicationFile11_data_Build_Employees_and_Age_Data, clear
disp('ReplicationFile12'), ReplicationFile12_data_Linkage_Data_Analysis, clear
disp('ReplicationFile13'), ReplicationFile13_data_Build_PanelData_PanelRegressions, clear
% File #14 is a .do file with Table 4 panel regressions (runs from stata instead)               % Table 4
disp('ReplicationFile15'), ReplicationFile15_data_MomentsToCalibrate, clear
disp('ReplicationFile16'), ReplicationFile16_data_Figure_C1_hptrend_Time_series_plots, clear    % Figure C1

%% Model estimation and results
disp('ReplicationFile17'), ReplicationFile17_model_FigureA1_savesimulatoins, clear
disp('ReplicationFile18'), ReplicationFile18_model_FigureA1_saveplot, clear                     % Figure A1
disp('ReplicationFile19'), ReplicationFile19_model_smm_and_hessian, clear
disp('ReplicationFile20'), ReplicationFile20_model_decomposition, clear
disp('ReplicationFile21'), ReplicationFile21_model_Table2_3_ModelParameters, clear              % Tables 2 and 3
disp('ReplicationFile22'), ReplicationFile22_model_TS_ModelFit, clear
disp('ReplicationFile23'), ReplicationFile23_model_Table5_FactorModelRegressions, clear         % Table 5

%% Appendix Tables
disp('ReplicationFile24'), ReplicationFile24_Build_BEA_Network, clear
disp('ReplicationFile25'), ReplicationFile25_TableC1_truncation_analysis, clear                 % Table C1
disp('ReplicationFile26'), ReplicationFile26_smm_and_hessian_NoNetwork, clear
disp('ReplicationFile27'), ReplicationFile27_smm_and_hessian_NoID, clear
disp('ReplicationFile28'), ReplicationFile28_TableC2andC3, clear                                % Tables C2 and C3
disp('ReplicationFile29'), ReplicationFile29_model_aggvol, clear
disp('ReplicationFile30'), ReplicationFile30_TableC4, clear                                     % Table C4
disp('ReplicationFile31'), ReplicationFile31_model_AltID, clear
disp('ReplicationFile32'), ReplicationFile32_model_TablesC6_C7_AltSig, clear                    % Tables C6 and C7
disp('ReplicationFile33'), ReplicationFile33_data_Build_NewFirmSeries, clear
disp('ReplicationFile34'), ReplicationFile34_data_MomentsToCalibrateBySubsample, clear
disp('ReplicationFile35'), ReplicationFile35_TableC5A, clear                                    % Table C5 Panel A
disp('ReplicationFile36'), ReplicationFile36_TableC5B, clear                                    % Table C5 Panel B