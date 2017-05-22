clear all;
close all;

a = csvread('ciena1000.csv');

first_line = a(1:10,:);

size(first_line)

netCD = []
fiber_length=[]

netCD = [netCD;first_line(:,6201)]

fiber_length = [fiber_length;first_line(1:5,6202:6221)]

launch_pow = first_line(1:5,6222:6241)

fiber_type = first_line(1:5,6242:6261)