
function ds_dt = LIN_arm(s, a)

global  Dt psi q_min q_max

q = s(1:2, :);
q_vel = s(3:4, :);
GAMMA_4by100 = psi(2)*sin(q(2,:)).*[-q_vel(2,:);  -(q_vel(1,:)+q_vel(2,:)); q_vel(1,:); zeros(size(q(1,:)))];
    GAMMA_q_vel = zeros(size(q_vel));
    GAMMA_q_vel(1,:) = GAMMA_4by100(1,:).*q_vel(1,:) + GAMMA_4by100(2,:).*q_vel(2,:);
    GAMMA_q_vel(2,:) = GAMMA_4by100(3,:).*q_vel(1,:) + GAMMA_4by100(4,:).*q_vel(2,:);
M = [psi(1)+2*psi(2)   psi(3)+psi(2); psi(3)+psi(2)   psi(3)];    
q_acc = inv(M)*(a - GAMMA_q_vel);

q_vel = q_vel + Dt*q_acc;
q = q+  Dt*q_vel;

for i = 1:numel(size(s(1,:)))
    
q_bounded(:,i) = max(q_min, min(q_max, q(:,i)));
q_vel(:,i) = (q(:,i) == q_bounded(:,i)).*q_vel(:,i);
q(:,i) = q_bounded(:,i);

end

ds_dt = [q_vel; q_acc];
    
end



