clear all;
close all;
a = zeros(10);
for i = 1:10,
    for j=1:10,
        a(i,j)=j-1+10*(i-1);
    end
end
size(a)
%csvwrite('testlist.csv',a)
b=1:1:20
c=[];
c={a a}
c{1}(1:1:20,:)
