%% Plot Eig Values
function plot_eigValues
load("eigs200.mat");

d_200 = sqrt(sort(abs(diag(eigs200{2})), "descend"))/sqrt(max(abs(diag(eigs200{2}))));

semilogy(d_200); grid on;
title("Eigenvalue Magnitude Decay");
xlabel("Magnitude Descend");
ylabel("|Eig Val|")
end