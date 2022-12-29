function [X,R,stats] = ram_demo(N, target, x, R, alpha_opt)
% RAM_DEMO  An example implementation of the Robust Adaptive Metropolis 
%
% Usage: X = ram_demo(N)
%        X = ram_demo(N, target, x)
%        [X,R,stats] = ram_demo(N, target, x, [R, [alpha_opt]])
%
% In:
%   N      -- Number of samples
%   target -- Function giving value of log-target density at a point
%             Default: log_banana_target
%   x      -- Initial value; column vector
%             Default: [0;0]
%   R      -- Initial shape matrix; upper triangular
%             Default: eye(length(x))
%   alpha_opt -- The desired acceptance rate 
%             Default: 0.234
%
% Out:
%   X -- The simulated samples
%   R -- The shape matrix at the end of simulation
%   stats  -- Struct of stats, with fields
%     .acc_rate -- Overall acceptance rate 
%     .sq_jump  -- Average square jump size
%
% Calling the function with one argument runs the 'demo mode' targetting
% a banana-shaped target:
% 
% >> X = ram_demo(2000);  plot(X(1,:), X(2,:), '.')
%
% You could do this to simulate from a standard Gaussian
%
% >> f = inline('-x^2', 'x');  X = ram_demo(2000, f, 0);  hist(X)
%
% Or you could write your 2-dimensional pdf to a file "target.m" 
% (located in the current working directory or in your path) and then call
%
% >> [X,R,stats] = ram_demo(2000, 'target', [0;0], eye(2), 0.3);

% Copyright (c) Matti Vihola 2010-2015
% 
% The code in this file is free software: you can redistribute it and/or
% modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation, either version 3 of the 
% License, or (at your option) any later version.
% 
% The code is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You can see a copy of the GNU General Public License at
% <http://www.gnu.org/licenses/>.

% Check inputs
if ~(nargin == 1 | (nargin >= 3 & nargin <= 5))
  error('See help ram_demo for usage')
end

if nargin == 1
  % The 'demo' mode
  target = @log_banana_target;
  x = [0;0];
  R = eye(2);
end
if nargin < 5
  alpha_opt = 0.234;
end
if nargin < 4
  R = eye(length(x));
end

% ... and outputs
if nargout > 3
  error('See help ram_demo for usage')
end

% Dimension of the target
D = length(x);

% Sanity checks on the arguments
if size(x,2) > 1
  error('The initial value must be a column vector')
end
if ~istriu(R) | size(R,1) ~= D
  error('The initial shape matrix must be upper diagonal and same dimension as x')
end

% Initialise output
X = zeros(D,N);
log_p_x = feval(target, x);

% Draw proposal and uniform random samples at once
rn = randn(D, N);
ru = rand(N, 1);
acc = 0; sq_jump = 0;
for k=1:N
  % Pick untransformed proposal
  u = rn(:,k);
  
  % Form proposal
  y = x + R'*u;
  
  % Calculate acceptance probability
  log_p_y = feval(target, y);
  alpha = min(1, exp(log_p_y - log_p_x));
  if ru(k) < alpha
    % Accept
    curr_sq_jump = sum((y-x).^2);
    acc = acc + 1;
    x = y; log_p_x = log_p_y;
  else
    curr_sq_jump = 0;
  end
  sq_jump = (k-1)/k*sq_jump + curr_sq_jump/k;
  
  % Store state
  X(:,k) = x;
  
  % The step size -- you can try to modify this for efficient adaptation
  eta = D/k^0.75;
  
  % Do the adaptation of shape matrix
  R = ram_adapt(R, u, alpha, eta, alpha_opt);
end

% The stats struct
stats = struct('acc_rate', acc/N, 'sq_jump', sq_jump);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = ram_adapt(R, u, alpha, eta, alpha_opt)
% RAM_ADAPT  Single adaptation step of the shape matrix in the RAM algorithm
%
% Usage: R_new = adapt(R_old, u, alpha, eta, alpha_opt)
%
% In:
%   R_old -- Old value of the shape matrix
%   u       -- The increment from the proposal (such as standard normal)
%   alpha -- The current observed acceptance probability
%   eta     -- Adaptation step size
%
% Out:
%   R_new -- New value of the shape matrix

% Ensure step size smaller than one
eta = min(0.9,2*eta);

% Calculate the normalised direction vector
nu = norm(u);
if nu==0
  u(:) = 0;
  u(1) = 1;
else
  u = u/nu;
end

% The increment vector
z = sqrt(eta*abs(alpha-alpha_opt))*R'*u;

% Rank one Cholesky update -- much faster than full Cholesky in higher
% dimensions
if alpha >= alpha_opt
  R = cholupdate(R, z,'+');
else
  R = cholupdate(R, z,'-');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE REST OF THE CODE IS FOR THE LOG BANANA EXAMPLE; YOU DO NOT NEED 
% ANYT OF THAT IF YOU USE YOUR OWN TARGET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = log_banana_target(x)
% LOG_BANANA_TARGET  Log-density of the infamous 2D 'banana' target
%
% Usage: p = log_banana_target(x)
%
% In:
%   x -- 2-by-n matrix of 2D points
%
% Out:
%   p -- Vector of density values at x

a = 1; b = 1;

y1 = x(1,:)/a;
y2 = x(2,:)*a + a*b*(x(1,:).^2 + a^2);

C = [1 .9; .9 1];

p = normal_log_density([y1;y2], zeros(2,1), C);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function p = normal_log_density(x, m, C, Cinv)
% NORMAL_LOG_DENSITY  Multidimensional Gaussian density function, logarithm
%  
% Usage: p = normal_density(x, m, C)
%        p = normal_density(x, m, C, 'inv')
%
% In:
%   x -- D*N Data points where to evaluate the density, where N is the number
%        of data points and D the dimension.
%   m -- D*1 Mean vector of the distribution.
%   C -- D*D Covariance matrix, or the inverse covariance matrix, if the
%        fourth argument is supplied.
%
% Out:
%   p -- 1*N row vector of the logarithms of the density values.
%
% The function evaluates the values of Gaussian density with given mean and
% covariance matrix at the given points.

N = size(x, 2);
D = size(x, 1);
c = -0.91893853320467274178*D; % log(1/sqrt(2*pi))*D;

if nargin == 1
  m = zeros(D,1);
  C = eye(D);
end

dx = x - m*ones(1,N); % repmat(m, 1, N);

if nargin > 3
  p = c + .5*log(det(C)) -.5*sum(dx.*(C*dx),1);
else
  p = c - .5*log(det(C)) -.5*sum(dx.*(C\dx),1);
end
