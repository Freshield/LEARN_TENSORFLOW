clear all;
close all;
a = zeros(10);
for i = 1:10,
    for j=1:10,
        a(i,j)=j-1+10*(i-1);
    end
end
size(a)
b=[1,2,3,4]
minmax(b)