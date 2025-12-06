import numpy as np
from tests_impl import unconditional_relevance, conditional_relevance
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


MOMENTS = ("Z1", "Z2", "Z3", "Z4", "Z5")

def create_observations(T, beta, gamma):
    Z1 = np.random.poisson(4, size=T)       # relevant
    Z2 = np.random.exponential(scale=50, size=T)  # relevant
    Z3 = np.random.normal(size=T)           # irrelevant
    Z4 = np.random.uniform(-10, 10, size=T) # irrelevant
    Z5 = np.random.binomial(1, 0.3, size=T) # relevant
    
    mu_c = 0.01
    rho_c = 0.9
    alpha1 = 0.02
    alpha2 = 0.01
    alpha5 = 4
    sigma_c = 0.05
    sigma_r = 0.0005
    
    C = np.zeros(T+1)
    C[0] = 1.0
    
    for t in range(T):
        eps_c = np.random.randn()
        lnC_next = (
            mu_c + rho_c * np.log(C[t]) + alpha1*Z1[t] + alpha2*Z2[t] + alpha5*Z5[t] + sigma_c*eps_c
        )
        C[t+1] = np.exp(lnC_next)
    
    
    R = (1/beta) * (C[1:]/C[:-1])**gamma + sigma_r*np.random.randn(T)
    
    data = {
        "R": R,
        "C_t": C[:-1],
        "C_t+1": C[1:],
        "Z1": Z1,
        "Z2": Z2,
        "Z3": Z3,
        "Z4": Z4,
        "Z5": Z5,
    }

    return data


def make_moment(name: str):
    def moment(theta: np.ndarray, dp: dict[str, np.ndarray]):
        beta, gamma = theta
        core = beta * (dp['C_t+1'] / dp['C_t']) ** (-gamma) * dp["R"] - 1.0
        return core * dp[name]
    return moment


def test_unconditional_relevance_prodecure(T, beta, gamma, f2_indices):
    data = create_observations(T=T, beta=beta, gamma=gamma)
    theta_init = np.array([0.0, 0.0])
    moments = [make_moment(name) for name in MOMENTS]
    return unconditional_relevance(data, moments, f2_indices, theta_init)


def test_conditional_relevance_prodecure(T, beta, gamma, f2_indices):
    data = create_observations(T=T, beta=beta, gamma=gamma)
    theta_init = np.array([0.0, 0.0])
    moments = [make_moment(name) for name in MOMENTS]
    return conditional_relevance(data, moments, f2_indices, theta_init)


def calculate_rejection_frequency(pvalues, alpha):
    return sum(p <= alpha for p in pvalues) / len(pvalues)


def calculate_alphas_to_rejection_frequency(pvalues, num_alphas):
    alphas = np.linspace(0, 1, num_alphas)
    alphas_to_rej_freq = {}
    for alpha in alphas:
        alphas_to_rej_freq[alpha] = calculate_rejection_frequency(pvalues, alpha)
    return alphas_to_rej_freq


def monte_carlo_unconditional_relevance_prodecure(T, beta, gamma, sets_of_f2_indices, B, num_alphas, n_jobs=-1):
    def run_simulation():
        results = {}
        for f2_indices in sets_of_f2_indices:
            W, pval, theta_est = test_unconditional_relevance_prodecure(T, beta, gamma, f2_indices)
            results[tuple(f2_indices)] = pval
        return results
    
    
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation)() for _ in tqdm(range(B), desc="simulating many times")
    )
    
    set_to_pval = {tuple(_set): [] for _set in sets_of_f2_indices}
    for result in all_results:
        for _set, pval in result.items():
            set_to_pval[_set].append(pval)
    
    result = {}
    for _set, pvals in set_to_pval.items():
        alphas_to_rej_freq = calculate_alphas_to_rejection_frequency(pvals, num_alphas)
        moments = " ".join(MOMENTS[i] for i in _set)
        name = f"[UNCONDITIONAL RELEVANCE]: {moments}"
        result[name] = alphas_to_rej_freq
    
    return result


def monte_carlo_conditional_relevance_prodecure(T, beta, gamma, sets_of_f2_indices, B, num_alphas, n_jobs=-1):
    def run_simulation():
        results = {}
        for f2_indices in sets_of_f2_indices:
            W, pval, theta_est = test_conditional_relevance_prodecure(T, beta, gamma, f2_indices)
            results[tuple(f2_indices)] = pval
        return results
    
    
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation)() for _ in tqdm(range(B), desc="simulating many times")
    )
    
    set_to_pval = {tuple(_set): [] for _set in sets_of_f2_indices}
    for result in all_results:
        for _set, pval in result.items():
            set_to_pval[_set].append(pval)
    
    result = {}
    for _set, pvals in set_to_pval.items():
        alphas_to_rej_freq = calculate_alphas_to_rejection_frequency(pvals, num_alphas)
        moments = " ".join(MOMENTS[i] for i in _set)
        name = f"[CONDITIONAL RELEVANCE]: {moments}"
        result[name] = alphas_to_rej_freq
    
    return result


def plot_rejection_curves(
    results: dict[str, dict[float, float]],
    save_path: str | None = None,
    figsize_scale: float = 1.0,
):
    if len(results) == 0:
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    n_tests = len(results)
    rows = int(np.ceil(np.sqrt(n_tests)))
    cols = int(np.ceil(n_tests / rows))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 4.2 * figsize_scale, rows * 3.2 * figsize_scale),
        dpi=150
    )
    axes = axes.flatten() if n_tests > 1 else [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_tests))

    for ax, (name, data), color in zip(axes, results.items(), colors):
        alphas = np.array(sorted(data.keys()))
        freqs = np.array([data[a] for a in alphas])

        ax.plot(
            alphas,
            freqs,
            linestyle="--",
            linewidth=1.6,
            color=color
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle=":",
            linewidth=1,
            color="black",
            alpha=0.2
        )

        ax.set_title(name, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Significance Level", fontsize=10)
        ax.set_ylabel("Rejection Frequency", fontsize=10)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.grid(True, alpha=0.25)
        ax.margins(x=0.05, y=0.05)

    for i in range(n_tests, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path is not None:
        if save_path.endswith(".svg"):
            plt.savefig(save_path, format="svg")
        else:
            plt.savefig(save_path, bbox_inches="tight")

    plt.show()



