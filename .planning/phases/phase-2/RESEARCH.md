# Phase 2 Research: Mathematical Stability & Estimator Refinement

## Problem 1: Small-Eigenvalue Sensitivity in Matrix-Rényi Entropy
The current implementation of Matrix-Rényi entropy for $\alpha=2$ uses:
$H_2(A) = -\log_2(Tr(A^2))$
Where $A = K / Tr(K)$.
If $Tr(A^2)$ is very small (near zero), $\log_2(Tr(A^2))$ goes to $-\infty$, and $H_2$ becomes very large.
However, $Tr(A^2) = \sum \lambda_i^2$. Since $\sum \lambda_i = 1$ and $\lambda_i \ge 0$, $Tr(A^2)$ is minimized when all $\lambda_i = 1/N$, giving $Tr(A^2) = N \cdot (1/N^2) = 1/N$.
So the minimum value of $Tr(A^2)$ is $1/N$, and the maximum entropy is $\log_2 N$.
The sensitivity occurs when $Tr(A^2)$ is calculated. In `renyi_entropy_2`, we have `tr_k_sq.clamp(min=1e-8)`.
If $N$ is very large, $1/N$ could be smaller than $1e-8$? No, that would require $N > 10^8$, which is unlikely for batch sizes.
HOWEVER, if $Tr(K)$ is very small, the normalization $A = K / Tr(K)$ could be unstable.

## Problem 2: Fractional Alpha-Entropy ($\alpha \neq 2$)
The roadmap mentions supporting $\alpha \neq 2$ (e.g., $\alpha=1$ for von Neumann).
$H_\alpha(A) = \frac{1}{1-\alpha} \log_2(Tr(A^\alpha))$
Calculating $Tr(A^\alpha)$ for non-integer $\alpha$ usually requires eigenvalue decomposition: $Tr(A^\alpha) = \sum \lambda_i^\alpha$.
This is $O(N^3)$.

## Problem 3: $O(N^2)$ Approximations
To avoid $O(N^3)$, we can use:
1. **Taylor Series Expansion**: Expand $x^\alpha$ around some point.
2. **Randomized Trace Estimators (Hutchinson)**: $Tr(f(A)) \approx \mathbb{E}[v^T f(A) v]$ where $v$ is a random vector (e.g., Rademacher).
   This requires computing $f(A)v$, which can be done efficiently if $f$ is a polynomial.
3. **Chebyshev Polynomials**: More stable than Taylor for approximating $x^\alpha$ on $[0, 1]$.

## Plan for Plan 2.1 (Audit)
1. Create a script to visualize the stability of `renyi_entropy_2` with varying batch sizes and correlation levels.
2. Check for gradient vanishing/exploding when $\lambda_i$ are small.

## Plan for Plan 2.2 (O(N^2) Approximations)
1. Implement a general `matrix_renyi_entropy(A, alpha)` function.
2. Implement Chebyshev polynomial approximation for $A^\alpha$.
3. Benchmark against exact eigenvalue method.

## Plan for Plan 2.3 (Boundary Conditions)
1. Test with $N=1$ (batch size 1).
2. Test with identical samples (perfect correlation).
3. Test with extremely high-dimensional inputs but small batch size.
