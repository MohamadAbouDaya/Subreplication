function result = gurobi(model, params) %#ok<INUSD>
% Stand-in for the Gurobi MATLAB interface (Gurobi is not installed here).
% Solves the model struct with MATLAB's linprog (LP) or intlinprog (MIP)
% and returns the result fields that the Neufeld-Xiang code reads:
% status, x, objval, objbound, pi, slack.

A = model.A;
rhs = model.rhs(:);
sense = model.sense(:);

if isscalar(sense)
    sense = repmat(sense, size(A, 1), 1);
end

le = sense == '<';
ge = sense == '>';
eq = sense == '=';

Aineq = [A(le, :); -A(ge, :)];
bineq = [rhs(le); -rhs(ge)];
Aeq = A(eq, :);
beq = rhs(eq);
lb = model.lb(:);
ub = model.ub(:);
f = model.obj(:);

objcon = 0;

if isfield(model, 'objcon')
    objcon = model.objcon;
end

result = struct;
is_mip = isfield(model, 'vtype') && any(model.vtype(:) ~= 'C');

if ~is_mip
    % strict dual simplex first; if it fails, solve the same LP with HiGHS
    % (through a small Python script) instead of loosening the tolerances
    opts = optimoptions('linprog', 'Display', 'off', 'Algorithm', ...
        'dual-simplex', 'ConstraintTolerance', 1e-9, ...
        'OptimalityTolerance', 1e-9);
    [x, fval, flag, ~, lam] = linprog(f, Aineq, bineq, Aeq, beq, ...
        lb, ub, opts);

    % linprog sometimes reports success with a wrong answer on these
    % badly scaled problems; accept its answer only if it passes the LP
    % optimality conditions
    if flag == 1 && isempty(Aeq)
        primal_res = max([0; Aineq * x - bineq]);
        dual_res = max(abs(Aineq' * lam.ineqlin + f));
        gap = abs(f' * x + bineq' * lam.ineqlin);

        if primal_res > 1e-7 || dual_res > 1e-8 || any(lam.ineqlin < -1e-9) ...
                || gap > 1e-7 * (1 + abs(fval))
            flag = -99;
        end
    end

    if flag ~= 1
        assert(isempty(Aeq) && all(isinf(lb)) && all(isinf(ub)), ...
            'HiGHS fallback only handles free variables and inequalities');
        in_file = [tempname, '.mat'];
        out_file = [tempname, '.mat'];
        save(in_file, 'f', 'Aineq', 'bineq', '-v7');
        here = fileparts(mfilename('fullpath'));
        cmd = ['python "', ...
            fullfile(here, 'lp_bridge.py'), '" "', in_file, '" "', ...
            out_file, '"'];
        [rc, msg] = system(cmd);
        delete(in_file);

        if rc ~= 0 || ~exist(out_file, 'file')
            disp(['gurobi shim: HiGHS fallback failed: ', msg]);
            result.status = 'LP_FAILED';
            return;
        end

        sol = load(out_file);
        delete(out_file);

        if sol.ok ~= 1
            result.status = 'LP_FAILED';
            return;
        end

        x = sol.x(:);
        fval = sol.fval;
        lam = struct('ineqlin', sol.lam(:), 'eqlin', zeros(0, 1));
    end

    result.status = 'OPTIMAL';
    result.x = x;
    result.objval = fval + objcon;

    % Gurobi sign convention for a minimisation problem
    dual = zeros(size(A, 1), 1);
    nle = nnz(le);
    dual(le) = -lam.ineqlin(1:nle);
    dual(ge) = lam.ineqlin(nle + 1:end);
    dual(eq) = -lam.eqlin;
    result.pi = dual;
    result.slack = rhs - A * x;
else
    vtype = model.vtype(:);
    bin = vtype == 'B';
    lb(bin) = max(lb(bin), 0);
    ub(bin) = min(ub(bin), 1);
    intcon = find(vtype ~= 'C');

    opts = optimoptions('intlinprog', 'Display', 'off', ...
        'RelativeGapTolerance', 0, 'AbsoluteGapTolerance', 1e-10, ...
        'IntegerTolerance', 1e-6, 'ConstraintTolerance', 1e-9, ...
        'LPOptimalityTolerance', 1e-9);
    obj_scale = max(1, max(abs(f)));
    [x, fval, flag, out] = intlinprog(f / obj_scale, intcon, Aineq, ...
        bineq, Aeq, beq, lb, ub, opts);
    fval = fval * obj_scale;

    if isfield(out, 'absolutegap') && ~isempty(out.absolutegap)
        out.absolutegap = out.absolutegap * obj_scale;
    end

    if isempty(x) || flag < 1
        result.status = sprintf('INTLINPROG_EXITFLAG_%d', flag);
        return;
    end

    result.status = 'OPTIMAL';
    result.x = x;
    result.objval = fval + objcon;

    gap = 0;

    if isfield(out, 'absolutegap') && ~isempty(out.absolutegap) ...
            && isfinite(out.absolutegap)
        gap = max(out.absolutegap, 0);
    end

    result.objbound = fval - gap + objcon;
end

end
