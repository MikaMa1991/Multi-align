function [prob] = dirpdf(x, alpha)
    prob = (gamma(sum(alpha)) / prod(gamma(alpha))) * prod(x .^ (alpha - 1));
end