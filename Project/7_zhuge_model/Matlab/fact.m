function output = fact(n)
%FACT Calculate factorial of a given positive integer.
output = 1;
for i = 1:n,
    output = output * i;
end
