% Given: Real vector TE = [ a < b < ... ], real nu > 1
% Computes: Inverse wavelet transformation matrix Tt

% Temporal mass and stiffness matrix on the mesh TE
N = length(TE); h = diff(TE); g = 1./h;
Mt = spdiags([h 0; 0 h]' * [1 2 0; 0 2 1]/6, -1:1, N, N);
At = spdiags([g 0; 0 g]' * [-1 1 0; 0 1 -1], -1:1, N, N);

% Rows of Tt are coefficients of wavelets wrt finest hats
Tt = speye(N);

mc = Mt; ac = At; dt = speye(3); IC = 1:N; % Aux for current coarse level
while (length(IC) >= 3)
    % Part I. Identify most energetic hats
    IF = []; % Indices for fine hats
    e0 = 0; eta = 1.9; % Reference energy, rel energy bandwidth
    while (length(IC) >= 3)
        e1 = diag(ac) ./ diag(mc); 
        [e1, j] = max([0; e1(2:end-1); 0]); % Non-boundary max
        if (e1 <= e0/eta) break; else e0 = max(e0, e1); end
        
        % Neighbors of the fine hat j get coarsened
        dt([1;3], 2) = -ac(j+[-1;1], j) / ac(j, j);
        J = j + (-1:1);
        ac(J,:) = dt * ac(J,:); ac(:,J) = ac(:,J) * dt';
        mc(J,:) = dt * mc(J,:); mc(:,J) = mc(:,J) * dt';
        Tt(IC(J),:) = dt * Tt(IC(J),:);
        % There is one more fine hat and one less coarse hat
        IF = [IF, IC(j)]; IC(j) = [];
        ac(j,:) = []; ac(:,j) = []; mc(j,:) = []; mc(:,j) = [];
    end
    
    % Part II. Approximate nu-fold orthogonalization
    P = @(X) X - (X * Mt * Tt(IC,:)') * diag(1./sum(mc)) * Tt(IC,:);
    for k = 1:nu; Tt(IF,:) = P(Tt(IF,:)); end
    
    % Optional: reorder rows (first coarse then fine)
    Tt(1:IC(end),:) = Tt([IC, IF],:); IC = 1:length(IC);
end
