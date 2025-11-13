import numpy as np
import autograd.numpy as anp
from autograd import grad, hessian, jacobian
from scipy.optimize import minimize
from scipy.stats import chi2
from typing import Callable, Tuple, List, Dict, Any


def vec(A):
    return anp.reshape(A.T, (-1,))


def vech(A: anp.ndarray) -> anp.ndarray:
    n = A.shape[0]
    out = []
    for j in range(n):
        for i in range(j, n):
            out.append(A[i, j])
    return anp.array(out)


def duplication_matrix(n: int) -> anp.ndarray:
    q = n * (n + 1) // 2
    D = np.zeros((n * n, q))
    col = 0
    for j in range(n):
        for i in range(j, n):
            row1 = i + j * n
            D[row1, col] = 1.0
            if i != j:
                row2 = j + i * n
                D[row2, col] = 1.0
            col += 1
    return anp.asarray(D)


def make_selector_P(m: int, k: int, f2_indexes: List[int]) -> anp.ndarray:
    """
    P: матрица (m2*k) x (m*k), такая что vec(G^T) -> vec(G2^T).
    Порядок vec(G^T): блоки длины k по моментам i=0..m-1.
    """
    f2_indexes = list(sorted(f2_indexes))
    m2 = len(f2_indexes)
    P = np.zeros((m2 * k, m * k))
    for r, i in enumerate(f2_indexes):
        P[r*k:(r+1)*k, i*k:(i+1)*k] = np.eye(k)
    return anp.asarray(P)


def unconditional_relevance(
    data: Dict[str, np.ndarray],
    moments: List[Callable],
    f2_indexes: List[int],
    theta_init: np.ndarray,
    ridge: float = 1e-8,
    verbose: bool = False
) -> Tuple[float, float, np.array]:
    '''
    Пайплайн теста на безусловную релевантность моментов f2. Формально, проверяется равенство якобиана
    моментов G2 нулю. Если G2 = 0 статистически значимо, то в решении задачи оптимизации пул моментов
    f2 "не может сам по себе задавать направления" для локального минимума, а значит не несет информации
    для оценки параметра theta, что можно интерпретировать как "нерелевантность" моментов f2.

    Params:
    ----------
    data: dict[str, np.array]
        Словарь с данным, где ключ - название переменной, значение - массив наблюдений
    f: list[Callable]
        Список моментных условий [f1, f2, ..., fm]
    f2_indexes: list[int]
        Список индексов моментных условий, подвергающихся тесту
    theta_init: np.array
        Начальные значения параметров
    ridge: float
        Параметр регуляризация для избежания сингулярности матрицы при инвертировании
    verbose: bool
        Показывать отладку работы теста (по-умолчанию отключено)

    Return:
    ----------
    tuple[float, float, np.array]
        Результаты теста: значение статистики, p-value, оценку theta
    '''
    def vprint(*args, **kwargs):
        '''
        Для отладки
        '''
        if verbose:
            print(*args, **kwargs)

    T = len(next(iter(data.values())))
    m = len(moments)
    theta_init = anp.asarray(theta_init)
    k = int(theta_init.size)
    f2_indexes = list(sorted(f2_indexes))
    m2 = len(f2_indexes)
    m1 = m - m2
    assert m2 > 0, "Нужно указать хотя бы один момент в f2_indexes"
    assert m1 >= k, "Требуется m1 >= k (идентифицируемость на f1)"

    data_ag = {key: anp.asarray(val) for key, val in data.items()}
    def get_data_point(t: int) -> Dict[str, anp.ndarray]:
        return {k: data_ag[k][t] for k in data_ag}

    def fbar(theta: anp.ndarray) -> anp.ndarray:
        rows = []
        for t in range(T):
            dp = get_data_point(t)
            rows.append(anp.array([mom(theta, dp) for mom in moments]))
        f_mat = anp.stack(rows)
        return anp.mean(f_mat, axis=0)

    def compute_f_and_G_all(theta: anp.ndarray):
        f_rows = []
        G_rows = []
        for t in range(T):
            dp = get_data_point(t)
            f_row = []
            G_row = []
            for mom in moments:
                val = mom(theta, dp)
                f_row.append(val)
                g = grad(lambda p: mom(p, dp))(theta)
                G_row.append(g)
            f_rows.append(anp.stack(f_row))
            G_rows.append(anp.stack(G_row))        
        f_mat = anp.stack(f_rows)   
        G_tens = anp.stack(G_rows)               
        G = anp.mean(G_tens, axis=0)  
        return f_mat, G_tens, G

    def compute_Omega_from_f_mat(f_mat: anp.ndarray) -> anp.ndarray:
        f_mean = anp.mean(f_mat, axis=0)
        f_c = f_mat - f_mean
        return (f_c.T @ f_c) / (T - 1)

    # 1) старт: Omega = I
    def gmm_objective_init(theta):
        fb = fbar(anp.asarray(theta))
        return float(fb.T @ fb)

    vprint('[ESTIMATING INITIAL THETA]: ', end='')
    res1 = minimize(
        lambda th: gmm_objective_init(th),
        np.array(theta_init, dtype=np.float64),
        method='BFGS'
    )
    theta0 = anp.asarray(res1.x)
    vprint('DONE')

    # 2) оптимальная стадия: Omega != I (more complex)
    vprint('[ESTIMATING OMEGA]: ', end='')
    f_mat0, G_tens0, G0 = compute_f_and_G_all(theta0)
    Omega0 = compute_Omega_from_f_mat(f_mat0)
    vprint('DONE')

    def gmm_objective(theta):
        fb = fbar(anp.asarray(theta))
        Oinv = anp.linalg.inv(Omega0 + ridge * anp.eye(m))
        return float(fb.T @ Oinv @ fb)

    vprint('[ESTIMATING FINAL THETA]: ', end='')
    res2 = minimize(
        lambda th: gmm_objective(th),
        np.array(theta0, dtype=np.float64),
        method='BFGS'
    )
    theta_hat = anp.asarray(res2.x)
    vprint("DONE")

    # 3) оценки элементов из r (после окончательной оценки theta)
    vprint('[ESTIMATING FINAL OMEGA]: ', end='')
    f_mat, G_tens, G = compute_f_and_G_all(theta_hat)
    Omega = compute_Omega_from_f_mat(f_mat)
    Oinv = anp.linalg.inv(Omega + ridge * anp.eye(m))
    vprint("DONE")

    # 4) g2 = vec(G2^T)
    G2 = G[f2_indexes, :]
    g2 = vec(G2.T)

    # 5) H2 — блок-диагональ из Hessian(mean f2_i)
    vprint('[ESTIMATING H2]: ', end='')
    def phi_i_factory(i: int):
        def phi_i(theta):
            s = 0.0
            for t in range(T):
                s = s + moments[i](theta, get_data_point(t))
            return s / T
        return phi_i

    H2 = anp.zeros((m2 * k, k))
    for r, i in enumerate(f2_indexes):
        Hi = hessian(phi_i_factory(i))(theta_hat)
        H2 = anp.concatenate([H2[:r*k, :],
                              Hi,
                              H2[(r+1)*k:, :]], axis=0) if r < m2 else Hi
    vprint("DONE")

    # 6) B = [B_f, P]
    vprint('[ESTIMATING B]: ', end='')
    middle = G.T @ Oinv @ G
    middle_inv = anp.linalg.inv(middle + ridge * anp.eye(k))
    B_f = H2 @ middle_inv @ G.T @ Oinv
    P = make_selector_P(m=m, k=k, f2_indexes=f2_indexes)
    B = anp.concatenate([B_f, P], axis=1)
    vprint("DONE")

    vprint('[ESTIMATING SIGMA R]: ', end='')
    R_rows = []
    for t in range(T):
        ft = f_mat[t, :]
        gt = vec(G_tens[t, :, :].T)
        R_rows.append(anp.concatenate([ft, gt]))
    R = anp.stack(R_rows)

    R_mean = anp.mean(R, axis=0)
    Rc = R - R_mean
    Sigma_r = (Rc.T @ Rc) / (T - 1)
    vprint("DONE")

    Sigma_g2 = B @ Sigma_r @ B.T
    Sigma_g2 = Sigma_g2 + ridge * anp.eye(m2 * k)
    W = float(T * (g2.T @ anp.linalg.inv(Sigma_g2) @ g2))
    p_value = 1.0 - chi2.cdf(W, df=m2 * k)
    vprint("[TEST DONE]")

    return W, float(p_value), theta_hat


def conditional_relevance(
    data: Dict[str, np.ndarray],
    moments: List[Callable],
    f2_indexes: List[int],
    theta_init: np.ndarray,
    ridge: float = 1e-8,
    verbose: bool = False
) -> Tuple[float, float, np.array]:
    '''
    Пайплайн теста на условную релевантность моментов f2 поверх моментов f1. Формально, проверяется равенство якобиана
    моментов G_delta нулю. Если G_delta = 0 статистически значимо, то исключение f2 не меняет кривизну теоретического GMM-критерия и асимптотическую
    точность оценок, а значит моменты f2 могут считаться "избыточными".

    Params:
    ----------
    data: dict[str, np.array]
        Словарь с данным, где ключ - название переменной, значение - массив наблюдений
    f: list[Callable]
        Список моментных условий [f1, f2, ..., fm]
    f2_indexes: list[int]
        Список индексов моментных условий, подвергающихся тесту
    theta_init: np.array
        Начальные значения параметров
    ridge: float
        Параметр регуляризация для избежания сингулярности матрицы при инвертировании
    verbose: bool
        Показывать отладку работы теста (по-умолчанию отключено)

    Return:
    ----------
    tuple[float, float, np.array]
        Результаты теста: значение статистики, p-value, оценку theta
    '''
    def vprint(*args, **kwargs):
        '''
        Для отладки
        '''
        if verbose:
            print(*args, **kwargs)

    T = len(next(iter(data.values())))
    m = len(moments)
    theta_init = anp.asarray(theta_init)
    k = int(theta_init.size)
    f2_indexes = list(sorted(f2_indexes))
    f1_indexes = [i for i in range(len(moments)) if i not in f2_indexes]
    m2 = len(f2_indexes)
    m1 = m - m2
    assert m2 > 0, "Нужно указать хотя бы один момент в f2_indexes"
    assert m1 >= k, "Требуется m1 >= k (идентифицируемость на f1)"

    data_ag = {key: anp.asarray(val) for key, val in data.items()}
    def get_data_point(t: int) -> Dict[str, anp.ndarray]:
        return {k: data_ag[k][t] for k in data_ag}

    def fbar(theta: anp.ndarray) -> anp.ndarray:
        rows = []
        for t in range(T):
            dp = get_data_point(t)
            rows.append(anp.array([mom(theta, dp) for mom in moments]))
        f_mat = anp.stack(rows)
        return anp.mean(f_mat, axis=0)
        
    def compute_Omega_from_f_mat(f_mat: anp.ndarray) -> anp.ndarray:
        f_mean = anp.mean(f_mat, axis=0)
        f_c = f_mat - f_mean
        return (f_c.T @ f_c) / (T - 1)

    def compute_f_and_G_all(theta: anp.ndarray):
        f_rows = []
        G_rows = []
        for t in range(T):
            dp = get_data_point(t)
            f_row = []
            G_row = []
            for mom in moments:
                val = mom(theta, dp)
                f_row.append(val)
                g = grad(lambda p: mom(p, dp))(theta)
                G_row.append(g)
            f_rows.append(anp.stack(f_row))
            G_rows.append(anp.stack(G_row))        
        f_mat = anp.stack(f_rows)   
        G_tens = anp.stack(G_rows)               
        G = anp.mean(G_tens, axis=0)  
        return f_mat, G_tens, G

    # 1) старт: Omega = I
    def gmm_objective_init(theta):
        fb = fbar(anp.asarray(theta))
        return float(fb.T @ fb)

    vprint('[ESTIMATING INITIAL THETA]: ', end='')
    res1 = minimize(
        lambda th: gmm_objective_init(th),
        np.array(theta_init, dtype=np.float64),
        method='BFGS'
    )
    theta0 = anp.asarray(res1.x)
    vprint('[DONE]')

    vprint('[ESTIMATING INITIAL OMEGA]: ', end='')
    f_mat0, G_tens0, G0 = compute_f_and_G_all(theta0)
    Omega0 = compute_Omega_from_f_mat(f_mat0)
    vprint('[DONE]')

    def gmm_objective(theta):
        fb = fbar(anp.asarray(theta))
        Oinv = anp.linalg.inv(Omega0 + ridge * anp.eye(m))
        return float(fb.T @ Oinv @ fb)

    vprint('[ESTIMATING FINAL THETA]: ', end='')
    res2 = minimize(
        lambda th: gmm_objective(th),
        np.array(theta0, dtype=np.float64),
        method='BFGS'
    )
    theta_hat = anp.asarray(res2.x)
    vprint('[DONE]')

    # 2) оптимальная стадия: Omega != I (more complex)
    vprint('[ESTIMATING FINAL OMEGA]: ', end='')
    f_mat, G_tens, G = compute_f_and_G_all(theta_hat)
    Omega = compute_Omega_from_f_mat(f_mat)
    Oinv = anp.linalg.inv(Omega + ridge * anp.eye(m))
    vprint('[DONE]')


    # 3) Находим g_delta_t
    vprint('[ESTIMATING g_delta: ]', end='')
    Omega21 = Omega[f2_indexes][:,f1_indexes]
    Omega11 = Omega[f1_indexes][:,f1_indexes]
    Omega11_inv = np.linalg.inv(Omega11 + ridge * np.ones(m1))
    G2 = G[f2_indexes, :]
    G1 = G[f1_indexes, :]
    Gdelta_T = G2 - Omega21 @ Omega11_inv @ G1
    gdelta_vec = vec(Gdelta_T.T)
    vprint('[DONE]')

    # 4) Находим элементы С: C_f, C_g, C_w
    vprint('[ESTIMATING C]: ', end='')
    def phi_i_factory(i: int):
        def phi_i(theta):
            s = 0.0
            for t in range(T):
                s = s + moments[i](theta, get_data_point(t))
            return s / T
        return phi_i

    H_blocks = [hessian(phi_i_factory(i))(theta_hat) for i in range(m)]
    H_big = anp.zeros((m*k, k))
    for i, Hi in enumerate(H_blocks):
        H_big[i*k:(i+1)*k, :] = Hi

    def omega_vech(theta):
        rows = []
        for t in range(T):
            dp = get_data_point(t)
            rows.append(anp.array([moment(theta, dp) for moment in moments]))
        fm = anp.stack(rows)
        Om = compute_Omega_from_f_mat(fm)
        return vech(Om)
    Q = jacobian(omega_vech)(theta_hat)

    S1 = anp.eye(m)[f1_indexes, :]
    S2 = anp.eye(m)[f2_indexes, :]
    A_left = S2 - Omega21 @ Omega11_inv @ S1
    Ch = anp.kron(A_left, anp.eye(k))
    
    Dm = duplication_matrix(m)
    B_right = G1.T @ Omega11_inv @ S1
    A_right = Omega21 @ Omega11_inv @ S1
    C_omega = (anp.kron(A_right, B_right) - anp.kron(S2, B_right)) @ Dm

    middle = G.T @ Oinv @ G
    middle_inv = anp.linalg.inv(middle + ridge * anp.eye(k))
    Cf = (Ch @ H_big + C_omega @ Q) @ middle_inv @ G.T @ Oinv

    Cg1 = - anp.kron(Omega21 @ Omega11_inv, anp.eye(k))
    Cg2 = anp.eye(m2 * k)

    P1 = make_selector_P(m, k, f1_indexes)
    P2 = make_selector_P(m, k, f2_indexes)
    Cg_full = Cg1 @ P1 + Cg2 @ P2 

    C_total = anp.concatenate([Cf, Cg_full, C_omega], axis=1)
    vprint("[DONE]")

    # 5) Находим оценку Sigma_r
    vprint("[ESTIMATING Sigma_r]: ", end='')
    r_rows = []
    for t in range(T):
        dp = get_data_point(t)
        f_t = anp.array([moment(theta_hat, dp) for moment in moments])
        G_t = anp.stack([grad(lambda p: moment(p, dp))(theta_hat) for moment in moments])
        g_t = vec(G_t.T)
        w_t = vech(anp.outer(f_t, f_t))
        r_rows.append(anp.concatenate([f_t, g_t, w_t]))
    R = anp.stack(r_rows)
    R_mean = anp.mean(R, axis=0)
    Rc = R - R_mean
    Sigma_r = (Rc.T @ Rc) / (T-1)
    vprint("[DONE]")

    # 6) Находим оценку Sigma_g_delta
    Sigma_g_delta = C_total @ Sigma_r @ C_total.T + ridge * anp.eye(m2 * k)

    # 7) Строим финальную статистику W
    W = float(T) * gdelta_vec.T @ anp.linalg.inv(Sigma_g_delta) @ gdelta_vec
    p_value = float(1.0 - chi2.cdf(W, df=m2*k))
    vprint("[TEST DONE]")

    return W, p_value, theta_hat