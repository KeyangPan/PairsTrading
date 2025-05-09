import numpy as np
import cvxpy as cp
from typing import Dict
from typing import Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def EW_optimizer(rets_data):
    ''' 
    rets_data: a T*N dataframe, T is the length of look-back widnow, N is the number of pairs. Each column of rets_data is the return time series of a pair's strategy
    '''
    T,N=rets_data.shape

    weights=pd.Series(np.ones(N)/N,index=rets_data.columns)

    return weights

def Sharpe_optimizer(
    rets_data: pd.DataFrame,
    rf: float = 0.0,
    solver: Optional[str] = None,
    **solver_opts
) -> pd.Series:
    """
    Max‑Sharpe (long‑only) portfolio optimisation via quadratic programming.

    Parameters
    ----------
    rets_data : pd.DataFrame
        T×N return matrix (rows = time, cols = strategy / asset).
    rf : float, default 0.0
        Risk‑free rate used to compute excess returns.
    solver : str or None
        cvxpy solver name; if None, cvxpy 会按默认顺序自动选择.
    **solver_opts
        传给 `problem.solve()` 的其他关键字参数（如 `warm_start=True` 等）。

    Returns
    -------
    pd.Series
        Optimal long‑only weights that maximise the (ex‑ante) Sharpe ratio.
    """
    if rets_data.empty:
        raise ValueError("rets_data 不能为空")

    # ————————— 数据统计量 —————————
    mu = rets_data.mean().to_numpy()              # shape = (N,)
    Sigma = rets_data.cov().to_numpy()            # shape = (N, N)
    mu_ex = mu - rf                               # excess mean return

    # 检查可行性：必须存在至少一个正的超额收益
    if (mu_ex <= 0).all():
        raise ValueError("所有资产的超额收益都 ≤ 0，无法满足 Sharpe 约束 μ̂ᵀy = 1")

    N = len(mu)
    y = cp.Variable(N, nonneg=True)               # y ≥ 0 自动加上

    # ————————— 构建 QP —————————
    objective = cp.Minimize(cp.quad_form(y, Sigma))
    constraints = [mu_ex @ y == 1]                # μ̂ᵀy = 1

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, **solver_opts)

    if prob.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"优化失败，状态: {prob.status}")

    # ————————— 归一化得到权重 —————————
    y_opt = y.value
    w = y_opt / y_opt.sum()                       # x̄ = ȳ / Σ ȳⱼ
    weights = pd.Series(w, index=rets_data.columns)

    return weights

def MRP_optimizer_scipy(
    spreads_data: pd.DataFrame,
    nu=None,
    p: int = 6,
    x0="ew",
    verbose: bool = False,
    max_iter: int = 5_000,
) -> np.ndarray:
    """
    Mean‑Reverting‑Portfolio (MRP) optimiser solved with SciPy’s
    ``trust-constr`` algorithm.

    Parameters
    ----------
    spreads_data : pd.DataFrame  (shape: T × N)
        Each column is a spread (pair‑trade) time series.
    nu : float or None, default None
        Target portfolio variance  ``wᵀ Σ₀ w = ν``.
        If *None*, ν is initialised to the equal‑weight portfolio variance.
    p : int, default 6
        Highest autocorrelation lag included in the objective.
    x0 : {"ew", ndarray}, default "ew"
        Starting weights.  "ew"  →  equal‑weight initial point.
    verbose : bool, default False
        ``True`` prints solver progress and messages.
    max_iter : int, default 5_000
        Maximum number of iterations given to the optimiser.

    Returns
    -------
    np.ndarray
        Optimal long‑only weight vector (length N).
    """
    # ---------- 1)  Estimate contemporaneous and lagged covariances ----------
    def _sym(M: np.ndarray) -> np.ndarray:
        """Numerical symmetrisation: (M + Mᵀ) / 2."""
        return 0.5 * (M + M.T)

    def _cross_cov(df: pd.DataFrame, lag: int) -> np.ndarray:
        """
        Cross‑covariance Σ_k  = E[x_t x_{t‑k}ᵀ] at a given lag.
        Fast vectorised implementation for demeaned data.
        """
        X = df.dropna()
        X = X - X.mean(axis=0)
        cur = X.iloc[lag:].to_numpy()
        lagged = X.shift(lag).iloc[lag:].to_numpy()
        return (cur.T @ lagged) / (len(cur) - 1)       # unbiased divisor

    T, N = spreads_data.shape
    cross_cov: Dict[int, np.ndarray] = {
        k: _cross_cov(spreads_data, k) for k in range(p + 1)
    }
    Sigma0 = _sym(cross_cov[0])              # contemporaneous covariance Σ₀
    Sigma_lags = [_sym(cross_cov[k]) for k in range(1, p + 1)]

    # ---------- 2)  Objective function and gradient ----------
    def _quad(w: np.ndarray, A: np.ndarray) -> float:
        """Helper: quadratic form  wᵀ A w  as a scalar float."""
        return float(w.T @ A @ w)

    def _objective(w: np.ndarray) -> float:
        """Σ_{k=1..p} (wᵀ Σ_k w)^2   — minimise cumulative autocorrelation."""
        return np.sum([_quad(w, A) ** 2 for A in Sigma_lags])

    # Gradient: ∂/∂w (wᵀAw)²  =  4 (wᵀAw) A w
    def _grad(w: np.ndarray) -> np.ndarray:
        g = np.zeros_like(w)
        for A in Sigma_lags:
            q = _quad(w, A)
            g += 4.0 * q * (A @ w)
        return g

    # ---------- 3)  Constraints ----------
    # 3.1  Non‑linear *equality* constraint  wᵀ Σ₀ w = ν
    def _var(w: np.ndarray) -> float:
        return _quad(w, Sigma0)

    def _var_jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (Sigma0 @ w)

    if nu is None:
        # Default target variance: variance of the equal‑weight portfolio
        nu = float(_quad(np.full(N, 1.0 / N), Sigma0))
        if verbose:
            print(f"nu defaulted to equal‑weight variance = {nu:.6g}")

    # NonlinearConstraint enforces  lb ≤ f(w) ≤ ub.
    # Here lb = ub = ν, hence an equality constraint.
    var_constraint = NonlinearConstraint(
        fun=_var,
        lb=nu,
        ub=nu,
        jac=_var_jac,            # analytical Jacobian accelerates convergence
    )

    # 3.2  Weight sum = 1  →  Linear equality constraint
    sum_constraint = LinearConstraint(np.ones((1, N)), 1.0, 1.0)

    # 3.3  Long‑only bounds (optional; bounds object is prepared but unused)
    # bounds = Bounds(0.0, np.inf)

    # ---------- 4)  Initial point ----------
    if isinstance(x0, str) and x0 == "ew":
        w0 = np.full(N, 1.0 / N)         # equal‑weight start
    else:
        w0 = np.asarray(x0, dtype=float)
        if w0.ndim != 1 or w0.size != N:
            raise ValueError("x0 shape mismatch.")
        w0 = np.clip(w0, 1e-6, None)     # ensure positivity
        w0 /= w0.sum()                   # renormalise to simplex

    if verbose:
            initial_obj = _objective(w0)
            print(f"Initial objective value: {initial_obj:.6g}")

    # ---------- 5)  SciPy optimiser call ----------
    res = minimize(
        fun=_objective,
        x0=w0,
        method="trust-constr",
        jac=_grad,
        constraints=[var_constraint, sum_constraint],
        # bounds=bounds,                # activate if explicit bounds are needed
        options=dict(
            verbose=3 if verbose else 1,
            maxiter=max_iter,
            xtol=1e-10,
            gtol=1e-8,
        ),
    )

    if verbose and (not res.success):
        print("Warning: local optimum not found.", res.message)

    if verbose:
            optimized_obj = _objective(res.x)
            print(f"Optimized objective value: {optimized_obj:.6g}")

    return pd.Series(res.x, index=spreads_data.columns,name='opt_w')