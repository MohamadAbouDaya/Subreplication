function run_basket(tag, sig, K_list, m_list, tol, mc_samp, mc_rep, time_limit)
% Apply the Neufeld-Xiang MMOT1DCPWA solver to the basket-call lower bound
%   inf E[(X_1 + ... + X_n - K)^+],  X_i = (100/n) exp(sig_i sqrt(T) Z - sig_i^2 T / 2)
% with T = 0.5, each marginal truncated to [0, quantile(1 - 1e-12)] and
% equal-probability knots as in their Experiment 3.

repo = fullfile(fileparts(mfilename('fullpath')), '..', 'repo');
addpath(fullfile(repo, 'cutplane'), fullfile(repo, 'probmeas'), ...
    fullfile(repo, 'utils'));

T = 0.5;
n = length(sig);
scale = 100 / n;
s_list = sig(:) * sqrt(T);
mu_list = log(scale) - 0.5 * s_list .^ 2;
trunc_ub = exp(mu_list + s_list * norminv(1 - 1e-12));

out = struct('K', {}, 'm', {}, 'LB', {}, 'UB_mean', {}, 'UB_se', {}, ...
    'iter', {}, 'time', {}, 'timeout', {}, 'knots', {}, 'coefs', {}, ...
    'constant', {}, 'atoms', {}, 'probs', {}, 'LSIP_LB', {}, 'LSIP_UB', {});

for K = K_list(:)'
    costfunc = struct;
    costfunc.plus = {[ones(1, n); zeros(1, n)], [-K; 0]};

    options = struct;
    options.global_formulation = 'LOG';
    options.display = false;
    options.time_limit = time_limit;

    for mi = 1:length(m_list)
        m = m_list(mi);
        knots_cell = cell(n, 1);

        if mi == 1
            marg_cell = cell(n, 1);
        end

        for i = 1:n
            if mi == 1
                marg_cell{i} = ProbMeas1DLogNormal(0, trunc_ub(i), ...
                    mu_list(i), s_list(i));
            end

            knots = marg_cell{i}.evaluateInverseCDF(linspace(0, 1, m + 1)');
            knots_cell{i} = {knots};
        end

        if mi == 1
            for i = 1:n
                marg_cell{i}.setSimplicialTestFuncs(knots_cell{i}{:});
            end

            OT = MMOT1DCPWA(marg_cell, costfunc, options, struct, struct);
            [coup_atoms, testfunc_vals] = OT.generateHeuristicCoupling();
        else
            [coup_atoms, testfunc_vals] = ...
                OT.updateSimplicialTestFuncs(knots_cell);
        end

        init_constr = struct('points', coup_atoms, ...
            'testfunc_vals', testfunc_vals);

        try
            output = OT.run(init_constr, tol);
        catch err
            fprintf('K = %6.2f  m = %4d  FAILED: %s\n', K, m, err.message);
            break;
        end

        LB = OT.getMMOTLowerBound();

        RS = RandStream('combRecursive', 'Seed', 1000 + mi * 100);
        UB_list = zeros(mc_rep, 1);

        for r = 1:mc_rep
            UB_list(r) = OT.getMMOTUpperBound(mc_samp, RS);
        end

        rec = struct;
        rec.K = K;
        rec.m = m;
        rec.LB = LB;
        rec.UB_mean = mean(UB_list);
        rec.UB_se = std(UB_list) / sqrt(mc_rep);
        rec.iter = output.iter;
        rec.time = output.total_time;
        rec.timeout = output.time_limit_exceeded;
        rec.knots = cellfun(@(c) c{1}', knots_cell, 'UniformOutput', false);
        rec.coefs = cellfun(@(c) c', OT.Runtime.PrimalSolution.Coefficients, ...
            'UniformOutput', false);
        rec.constant = OT.Runtime.PrimalSolution.Constant;
        rec.atoms = OT.Runtime.DualSolution.Atoms;
        rec.probs = OT.Runtime.DualSolution.Probabilities';
        rec.LSIP_LB = OT.Runtime.LSIP_LB;
        rec.LSIP_UB = OT.Runtime.LSIP_UB;
        out(end + 1) = rec; %#ok<AGROW>

        fprintf(['K = %6.2f  m = %4d  LB = %.6f  UB = %.6f (se %.1e)  ' ...
            'gap = %.2e  iter = %4d  time = %6.1fs  timeout = %d\n'], ...
            K, m, LB, rec.UB_mean, rec.UB_se, rec.UB_mean - LB, ...
            rec.iter, rec.time, rec.timeout);

        fid = fopen(fullfile(fileparts(mfilename('fullpath')), ...
            [tag, '.json']), 'w');
        fprintf(fid, '%s', jsonencode(struct('sig', sig(:)', 'T', T, ...
            'scale', scale, 'trunc_ub', trunc_ub', 'runs', out)));
        fclose(fid);
    end
end

end
