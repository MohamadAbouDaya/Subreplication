classdef ProbMeas1DLogNormal < ProbMeas1DInterval
    % Log-normal probability measure truncated to a compact interval;
    % X = exp(mu + s * Z) with Z standard normal. Mirrors the structure of
    % ProbMeas1DMixNorm in the Neufeld-Xiang repository.

    methods(Access = public)
        function obj = ProbMeas1DLogNormal(trunc_lb, trunc_ub, mu, s)
            obj@ProbMeas1DInterval(trunc_lb, trunc_ub);
            obj.Dens.Mu = mu;
            obj.Dens.S = s;
            obj.Dens.UntruncatedCDFAtLB = obj.untruncatedCDF(trunc_lb);
            obj.Dens.NormConst = obj.untruncatedCDF(trunc_ub) ...
                - obj.Dens.UntruncatedCDFAtLB;
        end

        function dens = densityFunction(obj, x)
            inside = obj.checkIfInsideSupport(x);
            dens = zeros(length(x), 1);
            pos = x > 0;
            dens(pos) = lognpdf(x(pos), obj.Dens.Mu, obj.Dens.S);
            dens = (dens / obj.Dens.NormConst) .* inside;
        end

        function x = evaluateInverseCDF(obj, probs, varargin)
            p = probs * obj.Dens.NormConst + obj.Dens.UntruncatedCDFAtLB;
            x = exp(obj.Dens.Mu + obj.Dens.S * norminv(p));
            x = min(max(x, obj.Supp.LowerBound), obj.Supp.UpperBound);
            x(probs <= 1e-15) = obj.Supp.LowerBound;
            x(probs >= 1 - 1e-15) = obj.Supp.UpperBound;
        end

        function cost = computeOTCost(obj)
            if ~isfield(obj.OT, 'Coupled') || ~obj.OT.Coupled
                error('must set the coupled discrete measure first');
            end

            atoms = obj.OT.DiscMeas.Atoms;
            [sorted_atoms, sorted_order] = sort(atoms, 'ascend');
            sorted_probs = obj.OT.DiscMeas.Probs(sorted_order);

            itv_right = obj.evaluateInverseCDF(cumsum(sorted_probs));
            itv_left = [obj.Supp.LowerBound; itv_right(1:end - 1)];
            itv_right_dist = abs(itv_right - sorted_atoms);
            itv_left_dist = abs(itv_left - sorted_atoms);

            itv_center = max(min(sorted_atoms, itv_right), itv_left);
            itv_center_dist = abs(itv_center - sorted_atoms);

            pwalim_lb = [itv_left; itv_center];
            pwalim_ub = [itv_center; itv_right];
            pwalim_diff = pwalim_ub - pwalim_lb;
            pwaval_lb = [itv_left_dist; itv_center_dist];
            pwaval_ub = [itv_center_dist; itv_right_dist];
            pwaval_diff = pwaval_ub - pwaval_lb;
            pwalim_diff(pwalim_diff == 0) = 1;

            slopes = pwaval_diff ./ pwalim_diff;
            piece_integral = obj.partialExpectation(pwalim_lb, ...
                pwalim_ub, slopes, pwaval_lb - slopes .* pwalim_lb);

            cost = sum(piece_integral) / obj.Dens.NormConst;
        end
    end

    methods(Access = protected)
        function probs = untruncatedCDF(obj, x)
            probs = zeros(length(x), 1);
            pos = x > 0;
            probs(pos) = normcdf((log(x(pos)) - obj.Dens.Mu) / obj.Dens.S);
        end

        function vals = partialExpectation(obj, k1, k2, slope, intercept)
            % E[(slope * X + intercept) 1{k1 < X < k2}] for the
            % untruncated log-normal law
            k2 = max(k1, k2);
            d1 = (log(max(k1, 0)) - obj.Dens.Mu) / obj.Dens.S;
            d2 = (log(max(k2, 0)) - obj.Dens.Mu) / obj.Dens.S;
            s = obj.Dens.S;
            m = exp(obj.Dens.Mu + 0.5 * s ^ 2);
            vals = slope .* m .* ProbMeas1DLogNormal.cdfDiff(d1 - s, ...
                d2 - s) + intercept .* ProbMeas1DLogNormal.cdfDiff(d1, d2);
        end

        function integrals = integrateSimplicialTestFuncs(obj, knots)
            knot_num = length(knots);
            knot_diff = diff(knots);
            left_integrals = zeros(knot_num, 1);
            right_integrals = zeros(knot_num, 1);

            left_integrals(2:end) = obj.partialExpectation( ...
                knots(1:end - 1), knots(2:end), ...
                1 ./ knot_diff, -knots(1:end - 1) ./ knot_diff);
            right_integrals(1:end - 1) = obj.partialExpectation( ...
                knots(1:end - 1), knots(2:end), ...
                -1 ./ knot_diff, knots(2:end) ./ knot_diff);

            integrals = (left_integrals + right_integrals) ...
                / obj.Dens.NormConst;
        end
    end

    methods(Static, Access = public)
        function v = cdfDiff(a, b)
            % normcdf(b) - normcdf(a), computed in the tail that avoids
            % cancellation
            v = normcdf(b) - normcdf(a);
            up = a > 0;
            v(up) = normcdf(-a(up)) - normcdf(-b(up));
        end
    end
end
