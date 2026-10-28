% -----------------------------------------------------------
% file: test_asymmetric_lsap.m
% -----------------------------------------------------------
% authors: Sebastien Bougleux
% institution: Normandie Univ, CNRS - UNICAEN - ENSICAEN, GREYC UMR 6072 
% ----------------------------------------------------------- 
% This file is part of LSAPE.
% LSAPE is free software: you can redistribute it and/or modify
% it under the terms of the CeCILL-C License. See README file 
% for more details.
% ----------------------------------------------------------- 

clear; clc;
library_directory = '../../matlab/';
addpath(library_directory);
disp('Asymmetric LSAP test');
disp(' ');

% ------------------------------------------
% A simple example
% Cost matrix filled with random values in {1,...,n}

n = 5;
m = 8;
C = int32(randCost(n,m,10));
disp('Cost matrix with random integers');
disp(num2str(C));

% hungarian
% rho is the optimal permutation and (u,v) is the dual solution
[rho,u,v] = hungarianLSAP(C);
mincost = sum(u)+sum(v);

disp('A solution to the LSAP is given by:');
disp([num2str((1:n)'),repmat('->',n,1),num2str(rho)]);
disp('A solution to its dual:');
disp('u=');
disp(num2str(u));
disp(['v = ',num2str(v)]);
disp(['Minimal cost = ',num2str(mincost)]);

% Another simple example
% Cost matrix mxn filled with random values in {1,...,n}
disp(' ');
n = 5;
m = 8;
C = int32(MacholWien(m,n,10));
disp('Machol-Wien instance');
disp(num2str(C));

% hungarian
% rho is the optimal permutation and (u,v) is the dual solution
[rho,u,v] = hungarianLSAP(C);
mincost = sum(u)+sum(v);

disp('A solution to the LSAP is given by:');
disp([num2str((1:m)'),repmat('->',m,1),num2str(rho)]);
disp('A solution to its dual:');
disp('u=');
disp(num2str(u));
disp(['v = ',num2str(v)]);
disp(['Minimal cost = ',num2str(mincost)]);

% ------------------------------------------
% Time comparison between C and C'
% for worst-case time complexity
% Machol-Wien instance c_{ij} = i*j
% YOU HAVE TO WAIT SOME SECONDS

disp(' ');
disp('C vs C^T with Machol-Wien instance, wait ... !');

n = 500;
bg = 100;
stp = 100;
nd = 1000;
k = 1;
nb = (nd-bg)/stp;
tpsC = zeros(1,nb); tpsCp = zeros(1,nb);

for m = bg:stp:nd

    C = int32(MacholWien(n,m));
    % with Hungarian algorithm
    tic;
    phi = hungarianLSAP(C);
    tpsC(k) = toc;
    
    C = C';
    tic;
    phi = hungarianLSAP(C);
    tpsCp(k) = toc;
    k = k+1;
    
end

figure;
plot(bg:stp:nd,tpsC,'-r');
hold on;
plot(bg:stp:nd,tpsCp,'--b');
title('Cost Matrix vs its transpose');
xlabel('number m (n=500)');
ylabel('time (in s)');
legend('C','C^T','Location','northwest');

% ------------------------------------------
% Optimal vs approximation
disp(' ');
disp('Optimal vs greedy with random instance, wait ...!');


k = 1;
n=2000;
nd = n;
bg = 100;
nb = (nd-bg)/stp;

tpsG1 = zeros(1,nb); tpsC = zeros(1,nb);
solG1 = zeros(1,nb); solC = zeros(1,nb);
tpsG2 = zeros(1,nb); solG2 = zeros(1,nb);
tpsG3 = zeros(1,nb); solG3 = zeros(1,nb);
tpsG4 = zeros(1,nb); solG4 = zeros(1,nb);
tpsG5 = zeros(1,nb); solG5 = zeros(1,nb);

for m = bg:stp:nd

    C = int32(randCost(n,m,1));
    
    % with Hungarian algorithm
    tic;
    [phi,u,v] = hungarianLSAP(C);
    tpsC(k) = toc;
    solC(k) = sum(u)+sum(v);
    
    tic;
    [phi,solG1(k)] = greedyLSAP(C,0); % Basic greedy algorithm
    tpsG1(k) = toc;
    
    tic;
    [phi,solG2(k)] = greedyLSAP(C,1); % Refined greedy algorithm
    tpsG2(k) = toc;
    
    tic;
    [phi,solG3(k)] = greedyLSAP(C,2); % Loss greedy algorithm
    tpsG3(k) = toc;
    
    tic;
    [phi,solG4(k)] = greedyLSAP(C,3); % Basic sort greedy algorithm
    tpsG4(k) = toc;
    
    tic;
    [phi,solG5(k)] = greedyLSAP(C,4); % Counting sort greedy algorithm
    tpsG5(k) = toc;
    
    k = k+1;
    
end

figure;
plot(bg:stp:nd,tpsC,'-r');
hold on;
plot(bg:stp:nd,tpsG1,'--b');
plot(bg:stp:nd,tpsG2,'--k');
plot(bg:stp:nd,tpsG3,'--g');
plot(bg:stp:nd,tpsG4,'--y');
plot(bg:stp:nd,tpsG5,'--p');
title('Hungarian vs Greedy LSAP: time');
xlabel('number n');
ylabel('time (in s)');
legend('Hungarian LSAP','Basic Greedy LSAP','Refined Greedy LSAP','Loss Greedy LSAP','Basic sort Greedy LSAP','Counting sort Greedy LSAP','Location','northwest');

figure;
plot(bg:stp:nd,solC,'-r');
hold on;
plot(bg:stp:nd,solG1,'--b');
plot(bg:stp:nd,solG2,'--k');
plot(bg:stp:nd,solG3,'--g');
plot(bg:stp:nd,solG4,'--y');
plot(bg:stp:nd,solG5,'--p');
title('Hungarian vs Greedy LSAP: costs');
xlabel('number n');
ylabel('cost');
legend('Hungarian LSAP','Basic Greedy LSAP','Refined Greedy LSAP','Loss Greedy LSAP','Basic sort Greedy LSAP','Counting sort Greedy LSAP','Location','northwest');
