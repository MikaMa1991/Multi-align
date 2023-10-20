% Compute the SRVF of a given function.
% Input: 
%   f (1 x T): Input function.
% Output:
%   q (1 x T): Square-Root-Velocity-Function (SRVF) of the input.
function q = SRVF(t, f)
    dt = mean(diff(t));
    g = gradient(f, dt);
    q = sign(g) .* sqrt(abs(g));
end