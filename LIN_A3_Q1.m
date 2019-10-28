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

% create network
Q_est = create_net([n_s+n_a; 400; 400; 1], [0; 1; 1; 1]);
Q_tgt = Q_est;
folder = 'desktop/'; % folder with A3_infile
infile = sprintf('%sA3_infile.mat', folder);
load(infile);
mu_tgt = mu;

%set up learning
eta_mu = 3e-8;
eta_Q = 1e-4;
tau = 3e-5;
a_sd = 0.1;
n_rolls = 50;
n_m = 100;
adam_1 = 0.9;
adam_2 = 0.999;

%fill replay buffer
n_buf = 10000;
s = 2*(rand(n_s, n_buf) - 0.5);
q = s(1:n_q, :);
q_vel = s(n_q + 1:n_s, :);
a = 2*(rand(n_a, n_buf) - 0.5);
c = @(s, a) sum(s(1:n_q,:).*s(1:n_q,:), 1);
BUFFER = [s; a; c(s,a); s + Dt*LIN_arm(s, a)];
buf_i0 = 0;



for rollout = 1:n_rolls
    
  %q = 2*(rand(n_q, n_m) - 0.5);
  %q_vel = 2*(rand(n_q, n_m) - 0.5);
  %s = [q; q_vel];

  % Test policy
  if (rollout == 1) || (mod(rollout, 10) == 0)
    s = s_test;
    q = s(1:n_q, :);
    q_vel = s(n_q + 1:n_s, :);
    cost_test = zeros(1, n_m);
    for t = 1:n_steps
      mu = forward_relu(mu, s);
      a = mu.y{end};
      c = @(s, a) sum(s(1:n_q,:).*s(1:n_q,:), 1);
      cost_test = cost_test + c(s, a);
      s = s + Dt*LIN_arm(s,a);
    end  % for t
    cost_test = cost_test*Dt;
    fprintf('At rollout %d, mean cost = %s\n', rollout, mean(cost_test))
  end
  
  s = 2*(rand(n_s, n_m) - 0.5);
  %q = s(1:n_q, :);
  %q_vel = s(n_q + 1:n_s, :);
  
  for t = 1:n_steps

    % Compute off-policy transitions
    mu = forward_relu(mu, s);
    a = mu.y{end};
    c = @(s, a) sum(s(1:n_q,:).*s(1:n_q,:), 1);
    a_off = a + a_sd*randn(n_a, n_m);  % off-policy
    s_next = s + Dt*LIN_arm(s, a_off);

    % Store in the buffer
    buf_i0 = mod(buf_i0 + n_m, n_buf);
    BUFFER(:, buf_i0 + 1:buf_i0 + n_m) = [s; a_off; c(s, a_off); s_next];

    % Choose a minibatch from the buffer
    i = ceil(n_buf*rand(1, n_m));
    mb = BUFFER(:, i);
    ss = mb(1:n_s, :);
    aa = mb(n_s + 1:n_s + n_a, :);
    cc = mb(n_s + n_a + 1:n_s + n_a + 1, :);
    ss_next = mb(n_s + n_a + 2:end, :);
    
    % Adjust critic to minimize Bellman loss over the buffer-minibatch
    Q_est = forward_relu(Q_est, [ss; aa]);
    mu_tgt = forward_relu(mu_tgt, ss_next);
    aa_next = mu_tgt.y{end}; 
    Q_tgt = forward_relu(Q_tgt, [ss_next; aa_next]);
    Q_err = Q_est.y{end} - cc - Q_tgt.y{end};  % Bellman
    Q_est = backprop_relu_adam(Q_est, Q_err, eta_Q, adam_1, adam_2); 
    
    % Adjust actor to minimize Q over sweep-minibatch with on-policy actions 
    Q_est = forward_relu(Q_est, [s; a]);  % prepare for grad_net
    grad_Q = grad_net_relu(Q_est, ones(1, n_m));
    dQ_da = grad_Q(n_s + 1:end, :);
    mu = backprop_relu_adam(mu, dQ_da, eta_mu, adam_1, adam_2);
           
    % Nudge target nets toward learning ones
    for l = 2:Q_est.n_layers
      Q_tgt.W{l} = Q_tgt.W{l} + tau*(Q_est.W{l} - Q_tgt.W{l});
      Q_tgt.b{l} = Q_tgt.b{l} + tau*(Q_est.b{l} - Q_tgt.b{l});
    end
    for l = 2:mu.n_layers
      mu_tgt.W{l} = mu_tgt.W{l} + tau*(mu.W{l} - mu_tgt.W{l});
      mu_tgt.b{l} = mu_tgt.b{l} + tau*(mu.b{l} - mu_tgt.b{l});
    end
    
    % Update s
    s = s + Dt*LIN_arm(s, a);

  end  % for t

end  % for rollout
