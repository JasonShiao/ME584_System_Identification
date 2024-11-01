function u = inputSignalGen(k)
    % Parameters
    u = [ones(1, k/2) zeros(1, k/2)];
    u = u(1:k); 
end

N = 50;

% Generate u(k), y(k)
Ts = 0.1;
u = inputSignalGen(N);
sys = ss([1.5 -0.7;1 0], [1;0], [1 0.5], 0, Ts);
[y, t, x] = lsim(sys, u, (0:N-1) * Ts); % t will be k * Ts, corresponding to each time step
% Plot the output y(k)
figure;
plot(t, y);
xlabel('Time (s)');
ylabel('Output y(k)');
title('Output of Discrete-Time System');
grid on;

% Model Parameterization
x_0 = [0; 0]; % Initial condition
theta_1 = 0:0.02:1.5;
theta_2 = -2:0.02:2;
C = [0 1];
vec_B = [0.5; 1];
vec_D = 0;
theta_BD = [x_0 ; vec_B; vec_D];

tic;  % Start the execution timer

[Theta_1, Theta_2] = meshgrid(theta_1, theta_2);
J_1 = zeros(size(Theta_1));
J_2 = zeros(size(Theta_1));
for th1_idx = 1:size(Theta_1, 1)
    for th2_idx = 1:size(Theta_2, 2)
        th1 = Theta_1(th1_idx, th2_idx);
        th2 = Theta_2(th1_idx, th2_idx);
        A_1 = [0 -th1; 1 -th2];    % First way of parameterization
        A_2 = [0 th1*th2; 1 -th2]; % Second way of parameterization
        
        % Precompute middle column of time k=1 to N to reduce duplication and time complexity
        Mid_cols_1 = zeros(size(y,2), size(vec_B,1), N);
        Mid_cols_2 = zeros(size(y,2), size(vec_B,1), N);
        for i=2:(N-1)
            Mid_cols_1(:,:,i) = kron(u(i-1), C)*A_1^0 + Mid_cols_1(:,:,i-1) * A_1;
            Mid_cols_2(:,:,i) = kron(u(i-1), C)*A_2^0 + Mid_cols_2(:,:,i-1) * A_2;
        end

        % Accumulate each time step for cost function J
        for i=1:N
            phi = [C*A_1^(i-1),  Mid_cols_1(:,:,i),  kron(u(i),eye(size(y,2)))];
            J_1(th1_idx, th2_idx) = J_1(th1_idx, th2_idx) + 1/N*norm(y(i) - phi*theta_BD)^2;
        end
        for i=1:N
            phi = [C*A_2^(i-1),  Mid_cols_2(:,:,i),  kron(u(i),eye(size(y,2)))];
            J_2(th1_idx, th2_idx) = J_2(th1_idx, th2_idx) + 1/N*norm(y(i) - phi*theta_BD)^2;
        end
        % Cap for large values (e.g. 50)
        J_1(th1_idx, th2_idx) = min(J_1(th1_idx, th2_idx), 50);
        J_2(th1_idx, th2_idx) = min(J_2(th1_idx, th2_idx), 50);
    end
end

elapsedTime = toc;  % Stop the timer and get elapsed time in seconds
disp(['Run time: ', num2str(elapsedTime), ' seconds']);

figure;
surf(Theta_1, Theta_2, J_1);
title('{}_J_1');
xlabel('\theta_1');
ylabel('\theta_2');
zlabel('J_N');
figure;
surf(Theta_1, Theta_2, J_2);
title('{}_2J_N');
xlabel('\theta_1');
ylabel('\theta_2');
zlabel('J_N');

[minValue, rowIdx] = min(J_1); 
[minJ1, colIdx] = min(minValue);
rowIdx = rowIdx(colIdx);
disp(['Minimum value of J_1: ', num2str(minJ1), ...
    ' occurs at ', num2str(Theta_1(rowIdx, colIdx)), ',', num2str(Theta_2(rowIdx, colIdx))]);

[minValue, rowIdx] = min(J_2); 
[minJ2, colIdx] = min(minValue);
rowIdx = rowIdx(colIdx);
disp(['Minimum value of J_2: ', num2str(minJ2), ...
    ' occurs at ', num2str(Theta_1(rowIdx, colIdx)), ',', num2str(Theta_2(rowIdx, colIdx))]);
