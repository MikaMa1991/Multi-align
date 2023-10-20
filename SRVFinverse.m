% Compute the inverse map from SRVF of a given function.
% Input: 
%   q (1 x T): Square-Root-Velocity-Function (SRVF) of a function.
%   f0: The starting point of original function.
% Output:
%   f (1 x T): The inverse map from SRVF.
function f = SRVFinverse(t, q, f0)
    dt = mean(diff(t));
    qq = [0, q(1:end-1)];
    f = f0 + cumsum(qq .* abs(qq)) * dt;
end