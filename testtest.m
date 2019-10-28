%s = rand(4,3);
%B = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
%c = tanh(sum(s.*(B*s), 1));
%disp(c)

q = [1 2 3; 4 5 6];
c = sum(q.*q, 1);
disp(c)
