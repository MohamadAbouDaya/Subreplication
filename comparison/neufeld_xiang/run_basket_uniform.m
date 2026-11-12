function run_basket_uniform(tag, sig, K_list, m_list, delta, tol, mc_samp, mc_rep, time_limit)
% Neufeld-Xiang MMOT1DCPWA solver on the basket-call lower bound with
% equally spaced knots (as in their Experiment 1 and Corollary 6.6). Each
% log-normal marginal is truncated to [quantile(delta), quantile(1 - delta)].

repo = fullfile(fileparts(mfilename('fullpath')), '..', 'repo');
addpath(fullfile(repo, 'cutplane'), fullfile(repo, 'probmeas'), ...
    fullfile(repo, 'utils'));

T = 0.5;
n = length(sig);
scale = 100 / n;
s_list = sig(:) * sqrt(T);
mu_list = log(scale) - 0.5 * s_list .^ 2;
z = -norminv(delta);
trunc_lb = exp(mu_list - s_list * z);
trunc_ub = exp(mu_list + s_list * z);

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
        marg_cell = cell(n, 1);

        for i = 1:n
            marg_cell{i} = ProbMeas1DLogNormal(trunc_lb(i), trunc_ub(i), ...
                mu_list(i), s_list(i));
            knots = linspace(trunc_lb(i), trunc_ub(i), m + 1)';
            knots_cell{i} = knots;
            marg_cell{i}.setSimplicialTestFuncs(knots);
        end

        try
            OT = MMOT1DCPWA(marg_cell, costfunc, options, struct, struct);
            [coup_atoms, testfunc_vals] = OT.generateHeuristicCoupling();
            init_constr = struct('points', coup_atoms, ...
                'testfunc_vals', testfunc_vals);
            output = OT.run(init_constr, tol);
        catch err
            fprintf('K = %6.2f  m = %4d  FAILED: %s\n', K, m, err.message);
            continue;
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
        rec.knots = cellfun(@(c) c', knots_cell, 'UniformOutput', false);
        rec.coefs = cellfun(@(c) c', OT.Runtime.PrimalSolution.Coefficients, ...
            'UniformOutput', false);
        rec.constant = OT.Runtime.PrimalSolution.Constant;
        rec.atoms = OT.Runtime.DualSolution.Atoms;
        rec.probs = OT.Runtime.DualSolution.Probabilities';
        rec.LSIP_LB = OT.Runtime.LSIP_LB;
        rec.LSIP_UB = OT.Runtime.LSIP_UB;
        out(end + 1) = rec; %#ok<AGROW>

        fprintf(['K = %6.2f  m = %4d  LB = %.6f  UB = %.6f (se %.1e)  ' ...
            'iter = %4d  time = %6.1fs  timeout = %d\n'], ...
            K, m, LB, rec.UB_mean, rec.UB_se, rec.iter, rec.time, rec.timeout);

        fid = fopen(fullfile(fileparts(mfilename('fullpath')), ...
            [tag, '.json']), 'w');
        fprintf(fid, '%s', jsonencode(struct('sig', sig(:)', 'T', T, ...
            'scale', scale, 'trunc_ub', trunc_ub', 'runs', out)));
        fclose(fid);
    end
end

end
