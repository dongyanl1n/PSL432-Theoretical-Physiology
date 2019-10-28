function net = create_net(n_neurons, W_scale)

net.n_neurons = n_neurons;
net.n_layers = size(n_neurons, 1);  % #layers including input
[net.W, net.W_, net.W__, net.b, net.b_, net.b__, net.v, net.v_, net.v__, ...
  net.gamma, net.gamma_, net.gamma__, net.beta, net.beta_, net.beta__, ...   
  net.y, net.delta, net.dL_dW, net.dL_db, net.dL_dgamma, net.dL_dbeta] ...
  = deal(cell(net.n_layers, 1));

l = 1;
net.gamma{l} = ones(n_neurons(l), 1);
net.gamma_{l} = zeros(n_neurons(l), 1);
net.gamma__{l} = zeros(n_neurons(l), 1);
net.beta{l} = zeros(n_neurons(l), 1);
net.beta_{l} = zeros(n_neurons(l), 1);
net.beta__{l} = zeros(n_neurons(l), 1);
for l = 2:net.n_layers
  m = W_scale(l)/sqrt(n_neurons(l - 1));
  net.gamma{l} = ones(n_neurons(l), 1);
  net.gamma_{l} = zeros(n_neurons(l), 1);
  net.gamma__{l} = zeros(n_neurons(l), 1);
  net.beta{l} = zeros(n_neurons(l), 1);
  net.beta_{l} = zeros(n_neurons(l), 1);
  net.beta__{l} = zeros(n_neurons(l), 1);
  %net.v_avg_test{l} = zeros(n_neurons(l), 1);
  %net.v_var_test{l} = ones(n_neurons(l), 1);
  net.W{l} = m*(rand(n_neurons(l), n_neurons(l - 1)) - 0.5);
  net.W_{l} = zeros(n_neurons(l), n_neurons(l - 1));
  net.W__{l} = zeros(n_neurons(l), n_neurons(l - 1));
  net.b{l} = 0.2*rand(n_neurons(l), 1);
  net.b_{l} = zeros(n_neurons(l), 1);
  net.b__{l} = zeros(n_neurons(l), 1);
end
net.adam_1t = 1.0;
net.adam_2t = 1.0;
   
end