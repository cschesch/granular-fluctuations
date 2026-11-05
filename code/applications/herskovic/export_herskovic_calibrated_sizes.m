function export_herskovic_calibrated_sizes(project_root, package_root)
% Export exactly the size population from the authors' simulate_model.m.
% No network simulation, parameter change, or empirical inputs are involved.
if nargin < 2
    package_root = fullfile(project_root,'data','herskovic2020','replication_package');
end
path = fullfile(package_root,'code','guess','benchmark_parameters.mat');
parameters = load(path);
theta = parameters.benchmark_parameters(:);
N = 5000;
size_seed = 100;
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',size_seed));
z = normrnd(0,1,N,1);
S = exp(theta(6)+theta(7)*z);
matlab_version = version;
destination = fullfile(project_root,'code','.cache','derived','herskovic2020', ...
    'calibrated_sizes_matlab_seed100.mat');
if ~exist(fileparts(destination),'dir'), mkdir(fileparts(destination)); end
save(destination,'S','z','theta','N','size_seed','matlab_version','-v7');
fprintf('Saved original 5000-firm size population: top share %.9f\n',max(S)/sum(S));
end
