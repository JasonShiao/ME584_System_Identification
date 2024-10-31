function u = inputSignalGen(k)
    % Parameters
    period = 200; % period of 20s with sampling rate of 0.1s
    high_phase_length = period / 2;
    low_phase_length = period / 2;
    cycles = ceil(k/200);         % Number of cycles of the square wave
    one_period = [ones(1, high_phase_length), zeros(1, low_phase_length)];
    % Repeat the pattern for the desired number of cycles
    u = repmat(one_period, 1, cycles);
    % Crop to k signals only
    u = u(1:k); 
end

function [W,V] = noiseSignalGen(R, Q, k)
    % Generate process noise and measurement noise from Q and R
    Q = [0.01, 0.0; 0.0, 0.01];
    R = [0.0125];
    n_w = size(Q, 1);
    n_v = size(R, 1);
    L_w = chol(Q, 'lower');
    L_v = chol(R, 'lower');
    % Transform normal distributed noise to match covariance
    rand = randn(n_w, k);
    W = L_w * rand;
    %V = L_v * randn(n_v, k);
    V = [0 0.5] * W;
    W = [0.0975 0; 0.0024 0.0975] * W; % Actual w after multiplying the prefix matrix
    
    %figure;
    %hold on;
    %plot(1:k, W(1, :), 'DisplayName', 'W_1 (first row)');
    %plot(1:k, W(2, :), 'DisplayName', 'W_2 (second row)');
    %hold off;
    %title('Noise Signal Components over Time');
    %xlabel('Time step');
    %ylabel('Amplitude');
    %legend;
end

function [X_pred, X_est, P_pred, P_est] = KalmanFilter(k, A,B,C,D, Y,U, R,S,Q)
    X_pred = zeros(size(A,1), k); % x(k|k-1)
    X_est  = zeros(size(A,1), k);  % x(k|k)
    P_pred = zeros(size(A,1), size(A,1), k); % P(k|k-1)
    K_est  = zeros(size(A,1), size(C,1), k); % Kalman gain K'(k)
    K_pred = zeros(size(A,1), size(C,1), k); % Kalman gain K(k)

    % Initial value
    P_pred(:,:,1) = eye(size(A,1));
    %X_pred(:,1) = ...

    for i=1:k
        K_pred(:,:,i) = (S + A*P_pred(:,:,i)*C') / (R + C*P_pred(:,:,i)*C'); % use b/A instead of b*inv(A)
        if i < k
            P_pred(:,:,i+1) = A*P_pred(:,:,i)*A' + Q - K_pred(:,:,i)*(S+A*P_pred(:,:,i)*C')';
            X_pred(:,i+1) = A*X_pred(:,i) + B*U(:,i) + K_pred(:,:,i)*(Y(:,i) - C*X_pred(:,i));
        end
        K_est(:,:,i) = P_pred(:,:,i)*C' / (R+C*P_pred(:,:,i)*C'); % use b/A instead of b*inv(A)
        P_est(:,:,i) = P_pred(:,:,i) - K_est(:,:,i)*C*P_pred(:,:,i);
        X_est(:,i) = X_pred(:,i) + K_est(:,:,i)*(Y(:,i) - C*X_pred(:,i));
    end
end

% [Optional] convert from continuous time to discrete time
A_c = [-1/2 0; 1/2 -1/2];
B_c = [1; 0];
C_c = [0 1];
D_c = [0];
%sys_d = c2d(sys(A_c, B_c, C_c, D_c), 0.1, 'zoh');
%A_d = sys_d.A
%B_d = sys_d.B
%C_d = sys_d.C
%D_d = sys_d.D
k = 800; % Number of time steps

% Provided discrete time state model
A_d = [0.9512 0; 0.0476 0.9512];
B_d = [0.0975; 0.0024];
C_d = [0 1];
D_d = [0];

% Generate process noise and measurement noise from Q and R
Q = [0.01, 0.0; 0.0, 0.01];
R = [0.0125];
S = [0; 0.005];
[W, V] = noiseSignalGen(R,Q,k);
Q_2 = [0.0975 0; 0.0024 0.0975]*[0.0975 0; 0.0024 0.0975]';
S_2 = [0.0975 0; 0.0024 0.0975]*S;

% True output signals
X = zeros(2,k);
U = inputSignalGen(k);
% Initial condition x(0) = [1.5; 1.5]
X(:,1) = [1.5; 1.5];
for i = 1:(k-1)
    X(:,i+1) = A_d*X(:,i) + B_d * U(i) + W(:,i);
end
Y = C_d*X + V;

% Convention Kalman Filter
[X_pred, X_est, P_pred, P_est] = KalmanFilter(k, A_d,B_d,C_d,D_d, Y,U, R,S_2,Q_2);


% Plot
figure
subplot(4,1,1);
plot(1:k, X(1,:), 'DisplayName', 'true x_1');
hold on;
plot(1:k, X_est(1,:), 'DisplayName', 'est x_1')
grid on;
ylabel('x_1');
legend;
subplot(4,1,2);
plot(1:k, X(2,:), 'DisplayName', 'true x_2');
hold on;
plot(1:k, X_est(2,:), 'DisplayName', 'est x_2');
grid on;
ylabel('x_2');
legend;

subplot(4,1,3);
plot(1:k, X(1,:)-X_est(1,:), 'DisplayName', 'x_1_e');
grid on;
ylabel('x_1_e');
legend;

subplot(4,1,4);
plot(1:k, X(2,:)-X_est(2,:), 'DisplayName', 'x_2_e');
grid on;
ylabel('x_2_e');
legend;

hold off

%P = P_est(:,:,k);
%A_d*P*A_d' + Q_2 - (S_2+A_d*P*C_d')/(C_d*P*C_d'+R)*(S_2+A_d*P*C_d')'

[P, ~, ~] = idare(A_d', C_d', Q_2, R, S_2, []);