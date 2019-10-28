clear variables;
clc;
clf;
global Dt psi q_min q_max

psi(1) = 0.02;
psi(2) = 0.0065;
psi(3) = 0.0056;
n_q = 2;
n_a = 2;
n_s = 2*n_q;
q_min = [-1.5; 0];
q_max = [1.5; 2.5];

Dt = 0.01;
dur = 0.3;
n_steps = floor(1 + dur/Dt);
folder = 'desktop/'; % folder with A3_infile
infile = sprintf('%sA3_infile.mat', folder);
load(infile);


eta_mu = 3e-6;
tau = 3e-5;
a_sd = 0.1;
n_rolls = 100;
n_m = 100;
adam_1 = 0.9;
adam_2 = 0.999;

n_buf = 10000;
s = 2*(rand(n_s, n_buf) - 0.5);
q = s(1:n_q, :);
q_vel = s(n_q+1:n_s, :);
a = 2*(rand(n_a, n_buf) - 0.5);
c = @(s, a) sum(s(1:n_q,:).*s(1:n_q,:), 1) + 0.001*sum(a.*a, 1);
BUFFER = [s; a; c(s, a)];
buf_i0 = 0;

s_path = zeros(n_s, n_m, n_steps);
a_path = zeros(n_a, n_m, n_steps);
dC_da_path = zeros(n_a, n_m, n_steps);

for rollout = 1:n_rolls
    
  % Test policies
  if (rollout == 1) || (rollout == 10)||(rollout == 20)||(rollout == 30)||(rollout == 40)||(rollout == 50)
    
    % Run test batch
    s = s_test;
    cost_test = zeros(1, n_m);
    for t = 1:n_steps
      mu = forward_relu(mu, s);
      a = mu.y{end};
      cost_test = cost_test + c(s, a);
      s = s + Dt*LIN_arm(s,a);
    end  % for t
    cost_test = cost_test*Dt;
    fprintf('At rollout %d, mean cost = %s\n', rollout, mean(cost_test))
  end
  
  s = 2*(rand(n_s, n_m) - 0.5);
  
   for t = 1:n_steps
      
    % Compute off-policy rewards
    mu = forward_relu(mu, s);
    a = mu.y{end};
    s_path(:, :, t) = s;
    a_path(:, :, t) = a;
    a_off = a + a_sd*randn(n_a, n_m);
    
     % Store in the buffer
    buf_i0 = mod(buf_i0 + n_m, n_buf);
    BUFFER(:, buf_i0 + 1:buf_i0 + n_m) = [s; a_off; c(s, a_off)];
    
    % Choose a minibatch from the buffer %%%%do we really need this
    i = ceil(n_buf*rand(1, n_m));
    mb = BUFFER(:, i);
    s_buffer = mb(1:n_s, :);
    a_buffer = mb(n_s + 1:n_s + n_a, :);
    c_buffer = mb(n_s + n_a + 1, :);
    
    % Update s
    s = s + Dt*LIN_arm(s, a);
    
   end % for t
   
   % Sweep back
  t = n_steps;
  s = s_path(:, :, t);
  a = a_path(:, :, t);
  dc_ds = [2*s(1:n_q,:);zeros(n_q, n_m)];
  dc_da = 0.002*a;
  lam = Dt*dc_ds;
  for t = (n_steps - 1):-1:1
    s = s_path(:, :, t);
    a = a_path(:, :, t);
    f_est = forward_relu(f_est, [s; a]);
    dc_ds = [2*s(1:n_q,:);zeros(n_q, n_m)];
    dc_da = 0.002*a;
    lam_grad_f = grad_net_relu(f_est, lam);
    lam_df_da = lam_grad_f(n_s + 1:end, :);
    dC_da_path(:, :, t) = Dt*(dc_da + lam_df_da);
    lam_df_ds = lam_grad_f(1:n_s, :);
    lam = lam + Dt*(dc_ds + lam_df_ds);
  end  % for t
  
  
   % adjust mu
  s_block = reshape(s_path, n_s, n_m*n_steps);
  dC_da_block = reshape(dC_da_path, n_a, n_m*n_steps);
  mu = forward_relu(mu, s_block);
  a_block = mu.y{end};
  mu = backprop_relu_adam(mu, dC_da_block, eta_mu, adam_1, adam_2);
  
end % for rollouts


   
   
    
    
    

