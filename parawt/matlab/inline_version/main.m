function main
% function main
%
%	Test the fem_waveletT as an inline trafo
%
%	R. Andreev, 2014.04.29

	% Matrix version for reference
	function [M, A, T] = femT_waveletT(TE, nu, scale)
	% function [Mt, At, Tt] = femT_waveletT(TE, nu, scale)
	% R. Andreev, 2013.12.06

		if (nargin <= 1); nu = 2; end
		if (nargin <= 2); scale = true; end

		assert(all( TE == unique(sort(TE)) ));
		[M, A] = femT_assemE(TE);

		IC = 1:length(TE);

		% The rows of T are the coefficients of the new basis
		% with respect to the finest hat basis:
		T = speye(length(TE));

		mc = M; ac = A;

		while (length(IC) >= 3)
			assert(all(IC == sort(IC)));

			% Find most energetic hats
			eta = 1.9; % Relative energy bandwidth
			IF = []; e0 = 0;
			while (length(IC) >= 3)
				assert(length(IC) == min(size(mc)));

				e1 = diag(ac) ./ diag(mc);
				e1([1 end]) = 0;
				[e1, j] = max(e1);
				assert(numel(j) == 1);
				assert(e1 > 0);

				if (e1 <= e0 / eta); break; end;
				if (e0 == 0); e0 = e1; end

				% Neighbors of the fine hat j get coarsened:
				J = j + (-1:1);
				assert( ac(j,j) > 0 );
				dt = speye(3); dt([1,3], 2) = -ac(j+[-1,1], j) / ac(j, j);
				ac(J, :) = dt * ac(J, :); ac(:, J) = ac(:, J) * dt'; 
				mc(J, :) = dt * mc(J, :); mc(:, J) = mc(:, J) * dt';
				ac(j, :) = []; ac(:, j) = []; mc(j, :) = []; mc(:, j) = [];

				T(IC(J), :) = dt * T(IC(J), :);

				IF = [IF, IC(j)]; IC(j) = [];
			end

			assert(~isempty(IF), 'IF should not be empty here');
			assert( max(max(abs(T(IC,:) * M * T(IC,:)' - mc)))  <=  1e-8 );

			% Approximate nu-fold orthogonalization
			P = @(X) X - (X * M * T(IC,:)') * diag(1./sum(mc)) * T(IC,:);
			for k = 1:nu
				T(IF, :) = P( T(IF, :) );
			end

			% Reorder
			assert( (length(IC)+length(IF)) == IC(end) );
			T(1:IC(end), :) = T([IC, IF], :);
			IC = 1:length(IC); 
		end

		if scale
			T = diag(1./sqrt(diag(T * M * T'))) * T;
		end
	end

	T_end = 2;
	K = 8;
	TE = linspace(0, T_end, K+1);
	nu = 1;
	
	[Mt, At, Tt] = femT_waveletT(TE, nu, false);
% 	full(At)
% 	full(Tt * Mt * Vt')
	
	% Inline version
	function v = applyT(v, TE, nu, scale)
		if (nargin <= 1); nu = 2; end
		if (nargin <= 2); scale = true; end
		assert(nu == 1);
		
		v0 = v;

		assert(all( TE == unique(sort(TE)) ));
		[M, A] = femT_assemE(TE);

		IC = 1:length(TE);

		% The rows of T are the coefficients of the new basis
		% with respect to the finest hat basis:
		T = speye(length(TE));

		mc = M; ac = A;

		while (length(IC) >= 3)
			assert(all(IC == sort(IC)));

			% Find most energetic hats
			eta = 1.9; % Relative energy bandwidth
			IF = []; e0 = 0;
			while (length(IC) >= 3)
				assert(length(IC) == min(size(mc)));

				e1 = diag(ac) ./ diag(mc);
				e1([1 end]) = 0;
				[e1, j] = max(e1);
				assert(numel(j) == 1);
				assert(e1 > 0);

				if (e1 <= e0 / eta); break; end;
				if (e0 == 0); e0 = e1; end

				% Neighbors of the fine hat j get coarsened:
				J = j + (-1:1);
				assert( ac(j,j) > 0 );
				dt = speye(3); dt([1,3], 2) = -ac(j+[-1,1], j) / ac(j, j);
				ac(J, :) = dt * ac(J, :); ac(:, J) = ac(:, J) * dt'; 
				mc(J, :) = dt * mc(J, :); mc(:, J) = mc(:, J) * dt';

				T(IC(J), :) = dt * T(IC(J), :);
				v(IC(J)) = dt * v(IC(J));
				[v T*v0]
				
				
				mcj = mc(j, j + [-1,1]);
				s1 = sum(mc(j-1, :));
				s2 = sum(mc(j+1, :));
				
				full(mcj)
				[v(IC(j-1)); v(IC(j+1))]
				v(IC(j)) = v(IC(j)) - mcj * [v(IC(j-1))/s1; v(IC(j+1))/s2];

				ac(j, :) = []; ac(:, j) = []; mc(j, :) = []; mc(:, j) = [];
				IF = [IF, IC(j)]; IC(j) = [];
			end

			assert(~isempty(IF), 'IF should not be empty here');
			assert( max(max(abs(T(IC,:) * M * T(IC,:)' - mc)))  <=  1e-8 );

			% Approximate nu-fold orthogonalization
			P = @(X) X - (X * M * T(IC,:)') * diag(1./sum(mc)) * T(IC,:);
			for k = 1:nu
				full(T(IF, :) * M * T(IC,:)')
				v(IF) = v(IF) - T(IF, :) * M * T(IC,:)' * diag(1./sum(mc)) * v(IC);
				v(IC)
				T(IF, :) = P( T(IF, :) );
				[v T*v0]
				pause
			end

			% Reorder
			assert( (length(IC)+length(IF)) == IC(end) );
			T(1:IC(end), :) = T([IC, IF], :);
			v(1:IC(end)) = v([IC, IF]);
			IC = 1:length(IC); 
		end

% 		if scale
% 			T = diag(1./sqrt(diag(T * M * T'))) * T;
% 		end
	end

	v = rand(K+1, 1);
	full(Tt * v)
	full(applyT(v, TE, nu, false));
end
