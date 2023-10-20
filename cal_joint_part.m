function [prob] = cal_joint_part(sigma1, q1, q2, t, g)
    N = length(t);
    temp_in = cumtrapz(t, EXP(t, g).^2);
    temp_in = round(temp_in/temp_in(end)*(N-1))+1;
    SSE = (norm(q1 - q2(temp_in).*EXP(t,g')))^2;
%     SSE = trapz(t, (q1 - q2(temp_in).*EXP(t,g')).^2);
    prob = exp(-1/(2*sigma1.^2)*SSE);
end