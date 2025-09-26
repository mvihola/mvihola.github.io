#Copyright (c) 2020,2025 Matti Vihola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np

# This is a generic 'adaptive proposal', which is instantiated
# for each 'sampling block':
class RobustAdaptiveMetropolis:
    """Simple implementation of the Robust Adaptive Metropolis sampler.

    Building blocks for the Robust Adaptive Metropolis sampler
    (doi:10.1007/s11222-011-9269-5).

    The module provides two methods, 'draw' and 'adapt', which draw
    random samples from the proposal, and adapt the proposal, respectively.
    They should be applied consecutively ('adapt' uses auxiliary variables
    stored in 'draw').
    """

    # Constructor: d is the dimension of the proposal:
    def __init__(self, d, alpha_opt=0.234, gamma=0.66):
        """Initialise the adaptive proposal distribution.

        :param d: state-space dimension
        :param alpha_opt: desired acceptance rate (default: 0.234)
        :param gamma: step size decay rate (default: 0.66)
        """

        self.d = d                 # Dimension
        self.z = np.zeros(d)       # Independent N(0,I) rv's
        self.chol = np.identity(d) # Proposal Cholesky factor
        self.alpha_opt = alpha_opt # Desired accept rate
        self.gamma = gamma         # Step size decay rate
        self.accept_rate = 0.0     # Mean acceptance rate (for diagnostics!)

    def draw(self):
        """Draw proposal increment."""

        # Draw a proposal (increment) z ~ N(0,I), output ~ N(0, chol*chol')
        # NB: It is important that 'z' is saved in the state (used in adapt)
        self.z = np.random.randn(self.d)
        return self.chol @ self.z

    # Adapt the proposal covariance:
    def adapt(self, alpha, j):
        """Adapt the proposal.

        :param alpha: acceptance probability of the last proposal.
        :param j: current iteration number.
        """

        # Step size (as in the paper):
        step = min(1, self.d*pow(j+1, -self.gamma))

        # Accumulate accept rate statistic:
        self.accept_rate = (j/(j+1))*self.accept_rate + alpha/(j+1)

        # Difference of acceptance prob vs. desired:
        dalpha = alpha - self.alpha_opt

        # Calculate normalised 'innovation', avoiding division by zero:
        sz2 = np.sqrt(np.inner(self.z, self.z))
        normalised_z = self.chol @ (self.z/sz2 if sz2 > 0 else self.z)

        # Calculate new proposal covariance:
        cov = self.chol @ self.chol.transpose()
        cov += step * dalpha * np.outer(normalised_z, normalised_z)
        # ...and its Cholesky factor:
        self.chol = np.linalg.cholesky(cov)
        # (NB: This could also be done by a rank-1 Cholesky update/downdate,
        # which saves computations if d >> 1)


def ram_sampler(log_target, x0, n, blocks = None):
    """Robust adaptive Metropolis sampler (block-wise)

    Example implementation of the Robust adaptive Metropolis sampler (doi:10.1007/s11222-011-9269-5).

    :param log_target: log-target density (function)
    :param x0: initial state (numpy vector)
    :param n: number of samples to draw
    :param blocks: list of blocks, each a list of indices (default: single block).
    :returns: (X, samplers, accrate), where 'X' is array of simulated state vectors,
    'samplers' are states of all (adaptive) samplers, and 'accrate' is the realised acceptance rate.
    """

    d = x0.size
    # If not given, update all coordinates at once
    if blocks is None:
        blocks = [list(range(d))]

    # Initialise the samplers:
    samplers = [RobustAdaptiveMetropolis(len(block)) for block in blocks]
    # Initialise the output:
    X = np.zeros([n, d])
    # We start at x0, with log target value p_x
    x = x0; p_x = log_target(x); accepted = 0

    for i in range(n):

        # Propose/accept/adapt for each block:
        for sampler, block in zip(samplers, blocks):
            # Form the proposal:
            y = np.copy(x)
            y[block] = x[block] + sampler.draw()
            # ... and calculate its log target value:
            p_y = log_target(y)
            # ... and Metropolis-Hastings acceptance rate:
            acc_prob = min(1, np.exp(p_y - p_x))
            if np.random.rand() <= acc_prob:
                # Accept:
                x = y; p_x = p_y; accepted += 1
            # Do adaptation
            sampler.adapt(acc_prob, i)

        # Store the value of the chain
        X[i] = x

    # Oberall acceptance rate (over all iterations & blocks)
    accrate = accepted/(n*len(blocks))
    return X, samplers, accrate


# Run by calling 'o = ram_demo()' 
def ram_demo():
    # Test target distribution: N(0,C) where C = [1 0.9 0; 0.9 1.81 0.9; 0 0.9 1.81]
    def log_dnorm(x):
        chol = np.matrix([[1,0,0],[0.9,1,0],[0,0.9,1]])
        z = np.linalg.solve(chol, x)
        return -0.5*np.inner(z,z)

    x0 = np.zeros(3); n = 100000

    # Run a standard, full-dimensional (adaptive) random-walk Metropolis:
    print("Full-dimensional sampler")
    X, sampler, accrate = ram_sampler(log_dnorm, x0, n)
    print(f'Acceptance rate: {accrate}')
    print("Estimated covariance: "); print(np.cov(X.transpose()))

    # Run a block-wise adaptive random-walk Metropolis-within-Gibbs)
    print("Block-wise sampler")
    # The first coordinate in first 'block', the latter two in the second:
    blocks = [[0],[1,2]]
    X_blk, samplers_blk, accrate_blk = ram_sampler(log_dnorm, x0, n, blocks)
    print(f'Acceptance rate: {accrate_blk}')
    print("Estimated covariance: "); print(np.cov(X_blk.transpose()))
    return {"X": X, "sampler": sampler, 
            "X_blk": X_blk, "samplers_blk": samplers_blk}

if __name__ == "__main__":
    ram_demo()
