function run_exp3_term(n, offsets, m_list, tol, mc_samp, mc_rep, time_limit)
% Neufeld-Xiang solver on one convex term of their Experiment 3 cost,
% |<s, x> - t|, on their first n marginals (their class ProbMeas1DMixNorm,
% their equal-probability knots). s = first n entries of their first
% direction vector rescaled to unit length; t = E<s, x> + offset.

repo = fullfile(fileparts(mfilename('fullpath')), '..', 'repo');
addpath(fullfile(repo, 'cutplane'), fullfile(repo, 'probmeas'), ...
    fullfile(repo, 'utils'));

d = jsondecode(fileread('exp3_marginals.json'));
c = jsondecode(fileread('exp3_cost.json'));
s = c.weights_plus(1, 1:n);
s = s / norm(s);

out = struct('n', {}, 't', {}, 'offset', {}, 'm', {}, 'LB', {}, 'UB_mean', {}, ...
    'UB_se', {}, 'iter', {}, 'time', {}, 'timeout', {}, 'knots', {}, ...
    'coefs', {}, 'constant', {}, 's', {});

for off = offsets(:)'
    for mi = 1:length(m_list)
        m = m_list(mi);
        marg_cell = cell(n, 1);
        knots_cell = cell(n, 1);
        mean_sx = 0;

        for i = 1:n
            if iscell(d.marginals)
                p = d.marginals{i};
            else
                p = d.marginals(i);
            end

            mix = struct('weights', p.weights(:), 'mean_list', p.means(:), ...
                'std_list', p.stds(:));
            marg_cell{i} = ProbMeas1DMixNorm(d.lb, d.ub, mix);
            knots = marg_cell{i}.evaluateInverseCDF(linspace(0, 1, m + 1)');
            knots_cell{i} = knots;

            % mean of the marginal = integral of the identity, which lies
            % in the span of the test functions
            integrals = marg_cell{i}.setSimplicialTestFuncs(knots);
            mean_sx = mean_sx + s(i) * (knots' * integrals);
        end

        t = mean_sx + off;
        costfunc = struct;
        costfunc.plus = {[s; -s], [-t; t]};
        options = struct('global_formulation', 'LOG', 'display', false, ...
            'time_limit', time_limit);

        try
            OT = MMOT1DCPWA(marg_cell, costfunc, options, struct, struct);
            [coup_atoms, testfunc_vals] = OT.generateHeuristicCoupling();
            output = OT.run(struct('points', coup_atoms, ...
                'testfunc_vals', testfunc_vals), tol);
        catch err
            fprintf('offset = %g  m = %d  FAILED: %s\n', off, m, err.message);
            continue;
        end

        LB = OT.getMMOTLowerBound();
        RS = RandStream('combRecursive', 'Seed', 1000 + mi * 100);
        UB_list = zeros(mc_rep, 1);

        for r = 1:mc_rep
            UB_list(r) = OT.getMMOTUpperBound(mc_samp, RS);
        end

        rec = struct('n', n, 't', t, 'offset', off, 'm', m, 'LB', LB, ...
            'UB_mean', mean(UB_list), 'UB_se', std(UB_list) / sqrt(mc_rep), ...
            'iter', output.iter, 'time', output.total_time, ...
            'timeout', output.time_limit_exceeded, ...
            'knots', {cellfun(@(k) k', knots_cell, 'UniformOutput', false)}, ...
            'coefs', {cellfun(@(k) k', OT.Runtime.PrimalSolution.Coefficients, ...
            'UniformOutput', false)}, ...
            'constant', OT.Runtime.PrimalSolution.Constant, 's', s);
        out(end + 1) = rec; %#ok<AGROW>

        fprintf(['t = %8.4f (offset %4.1f)  m = %3d  LB = %.6f  UB = %.6f ' ...
            '(se %.1e)  iter = %4d  time = %6.1fs  timeout = %d\n'], t, off, ...
            m, LB, rec.UB_mean, rec.UB_se, rec.iter, rec.time, rec.timeout);

        fid = fopen(sprintf('exp3_term_nx_%d.json', n), 'w');
        fprintf(fid, '%s', jsonencode(struct('runs', out)));
        fclose(fid);
    end
end

end
