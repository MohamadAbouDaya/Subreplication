function export_exp3_marginals(n_out)
% Reproduce the marginal parameters of Experiment 3 of Neufeld-Xiang
% (exp/CPWACost_Exp/CPWAMM_prepare.m, same seed and same draws) and write
% the first n_out of them to exp3_marginals.json.
marg_num = 100;
marg_comp_num_range = [3, 5];
rng(1000, "combRecursive");
P = cell(marg_num, 1);
for marg_id = 1:marg_num
    mixnorm_comp_num = randi(marg_comp_num_range, 1, 1);
    w = ones(mixnorm_comp_num, 1) / mixnorm_comp_num;
    mu = randn(mixnorm_comp_num, 1) * 3;
    sd = sqrt(1 ./ gamrnd(3, 0.1, mixnorm_comp_num, 1));
    P{marg_id} = struct('weights', w', 'means', mu', 'stds', sd');
end
fid = fopen('exp3_marginals.json', 'w');
fprintf(fid, '%s', jsonencode(struct('lb', -10, 'ub', 10, 'marginals', {P(1:n_out)})));
fclose(fid);
end
