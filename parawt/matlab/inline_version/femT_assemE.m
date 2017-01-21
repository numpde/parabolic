function [MtE, AtE] = femT_assemE(TE)
% function [MtE, AtE] = femT_assemE(TE)
%
%   Creates P1 Lagrange FEM matrices on 1d mesh TE
%
%       MtE = (u, v)   and   AtE = (u', v')
%
%   including boundary functions
%
%   TE is a 1 x N vector
%
%   MtE and AtE are N x N matrices
%
%
%   See Sec 6.2 in
%
%       R. Andreev
%       Space-time discretization of the heat equation
%       Numerical Algorithms, 2014

%   R. Andreev, 2012.10.17

	assert(isequal(size(TE), [1 numel(TE)]), 'TE should be a row vector');

	K = length(TE);
	h = diff(TE); 
	g = 1./h;

	MtE = spdiags([h 0; 0 h]' * [1 2 0; 0 2 1]/6, -1:1, K, K);
	AtE = spdiags([g 0; 0 g]' * [-1 1 0; 0 1 -1], -1:1, K, K);
end
