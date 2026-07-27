function values = NODDI_erfi(x)
%NODDI_ERFI Imaginary error function on the real NODDI Watson domain.
%
% This package-authored implementation is not MATLAB File Exchange
% submission 18238. It preserves the numerical branches used by NODDI
% v1.05 for real, nonnegative floating-point inputs while avoiding eager
% evaluation of inactive branches.

if ~(isa(x, 'double') || isa(x, 'single')) || ~isreal(x)
    error('NODDI_erfi:InvalidInput', ...
        'x must be a real single- or double-precision array.');
end

values = nan(size(x), 'like', x);
finiteNonnegative = isfinite(x) & x >= 0;
values(finiteNonnegative & x == 0) = 0;

lower = finiteNonnegative & x > 0 & x < 5.7;
values(lower) = imag(gammainc(-(x(lower).^2), 0.5));

upper = finiteNonnegative & x >= 5.7;
values(upper) = exp(x(upper).^2) ./ (x(upper) .* sqrt(pi));
end
