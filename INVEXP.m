% Compute the Inverse Exp of a given function.
% Input: 
%   q (1 x T): Input sqvf function.
% Output:
%   v (1 x T): tangent expression of the sqvf.
function v = INVEXP(t, q)
    theta = acos(trapz(t, q));
    v = theta/sin(theta)*(q-cos(theta));
end