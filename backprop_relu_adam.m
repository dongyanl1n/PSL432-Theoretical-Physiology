function net = backprop_relu_adam(net, delta_out, eta, adam_1, adam_2)

% Compute gradients
l = net.n_layers;
net.delta{l} = 2*delta_out;  % transpose of dL/dv
net.dL_dW{l} = net.delta{l}*net.y{l - 1}';
net.dL_db{l} = sum(net.delta{l}, 2);
for l = net.n_layers - 1:-1:2
  net.delta{l} = (net.W{l + 1}'*net.delta{l + 1}) .* sign(net.y{l});
  net.dL_dW{l} = net.delta{l}*net.y{l - 1}';
  net.dL_db{l} = sum(net.delta{l}, 2);
end
net.delta{1} = net.W{2}'*net.delta{2};

% Adjust parameters
net.adam_1t = net.adam_1t*adam_1;
net.adam_2t = net.adam_2t*adam_2;
eta_t = eta*sqrt(1 - net.adam_2t)/(1 - net.adam_1t);
for l = 2:net.n_layers
  net.W_{l} = adam_1*net.W_{l} + (1 - adam_1)*net.dL_dW{l};
  net.W__{l} = adam_2*net.W__{l} + (1 - adam_2)*(net.dL_dW{l}.*net.dL_dW{l});
  net.W{l} = net.W{l} - eta_t*net.W_{l}./(sqrt(net.W__{l}) + 1e-8);
  net.b_{l} = adam_1*net.b_{l} + (1 - adam_1)*net.dL_db{l};
  net.b__{l} = adam_2*net.b__{l} + (1 - adam_2)*(net.dL_db{l}.*net.dL_db{l});
  net.b{l} = net.b{l} - eta_t*net.b_{l}./(sqrt(net.b__{l}) + 1e-8);
end

end

