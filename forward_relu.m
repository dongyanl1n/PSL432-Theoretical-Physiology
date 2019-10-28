function net = forward_relu(net, x)

net.y{1} = x;
for l = 2:net.n_layers - 1
  net.y{l} = max(0, net.W{l}*net.y{l - 1} + net.b{l});
end
l = net.n_layers;
net.y{l} = net.W{l}*net.y{l - 1} + net.b{l};  % neurons in final layer are affine

end