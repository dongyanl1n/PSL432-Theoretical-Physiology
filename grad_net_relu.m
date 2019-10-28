function z_grad = grad_net_relu(net, z)

net.delta{net.n_layers} = z;
for l = net.n_layers - 1:-1:2
  net.delta{l} = (net.W{l + 1}'*net.delta{l + 1}) .* sign(net.y{l});
end
z_grad = net.W{2}'*net.delta{2};  % net.delta{1} = transpose of z'*dy{end}/dy{1}

end

