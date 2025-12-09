import numpy as np
from tests_impl import partial_conditional_relevance, partial_unconditional_relevance
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import autograd.numpy as anp


MOMENTS = ('Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', "Z7", "Z8")


def create_observations(T, theta1, theta2, theta3, theta4, theta5):
    Z1 = np.random.normal(0, 1, T)
    Z2 = np.random.normal(0, 1, T)
    Z3 = np.random.normal(0, 0.3, T)
    Z4 = np.random.uniform(-1, 1, T)
    Z5 = np.random.normal(0, 1, T)
    Z6 = np.random.standard_t(5, T)
    Z7 = np.random.normal(0, 1, T)
    Z8 = np.random.normal(0, 1, T)

    def norm(z):
        return (z - z.mean()) / (z.std() + 1e-8)

    Z1n = norm(Z1)
    Z2n = norm(Z2)
    Z3n = norm(Z3)
    Z4n = norm(Z4)
    Z5n = norm(Z5)
    Z6n = norm(Z6)
    Z7n = norm(Z7)
    Z8n = norm(Z8)

    exp_val = np.exp(np.clip(theta3 * Z3n, -6, 6))
    softplus_arg = Z4n + theta5 * Z5n

    softplus_val = np.log1p(np.exp(-np.abs(softplus_arg))) + np.maximum(softplus_arg, 0)

    term1 = theta1 * np.sin(Z1n)
    term2 = theta2 * (Z2n ** 2) * exp_val
    term3 = theta4 * softplus_val

    eps = 0.05 * np.random.randn(T)
    Y = term1 + term2 + term3 + eps

    return {
        "Y": Y,
        "Z1": Z1n,
        "Z2": Z2n,
        "Z3": Z3n,
        "Z4": Z4n,
        "Z5": Z5n,
        "Z6": Z6n,
        "Z7": Z7n,
        "Z8": Z8n,
    }


def make_moment(name):
    def moment(theta, dp):
        theta1, theta2, theta3, theta4, theta5 = theta

        exp_val = anp.exp(anp.clip(theta3 * dp['Z3'], -6, 6))
        softplus_arg = dp['Z4'] + theta5 * dp['Z5']
        softplus_val = anp.log1p(anp.exp(-anp.abs(softplus_arg))) + anp.maximum(softplus_arg, 0)

        term1 = theta1 * anp.sin(dp['Z1'])
        term2 = theta2 * (dp['Z2'] ** 2) * exp_val
        term3 = theta4 * softplus_val

        eps = dp["Y"] - term1 - term2 - term3
        return eps * dp[name]
    return moment


def test_partial_unconditional_relevance_prodecure(T, thetas, f2_indices, a_indices):
    data = create_observations(T, *thetas)
    theta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    moments = [make_moment(name) for name in MOMENTS]
    return partial_unconditional_relevance(
        data=data, 
        moments=moments,
        f2_indexes=f2_indices, 
        a_indexes=a_indices,
        theta_init=theta_init
    )


def test_partial_conditional_relevance_prodecure(T, thetas, f2_indices, a_indices):
    data = create_observations(T, *thetas)
    theta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    moments = [make_moment(name) for name in MOMENTS]
    return partial_conditional_relevance(
        data=data, 
        moments=moments,
        f2_indexes=f2_indices, 
        a_indexes=a_indices,
        theta_init=theta_init
    )


def calculate_rejection_frequency(pvalues, alpha):
    return sum(p <= alpha for p in pvalues) / len(pvalues)


def calculate_alphas_to_rejection_frequency(pvalues, num_alphas):
    alphas = np.linspace(0, 1, num_alphas)
    alphas_to_rej_freq = {}
    for alpha in alphas:
        alphas_to_rej_freq[alpha] = calculate_rejection_frequency(pvalues, alpha)
    return alphas_to_rej_freq


def monte_carlo_partial_unconditional_relevance_prodecure(
        T, 
        thetas, 
        sets_of_f2_indices, 
        sets_of_a_indices, 
        B, 
        num_alphas, 
        n_jobs=-1
    ):
    def run_simulation():
        results = {}
        for f2_indices, a_indices in zip(sets_of_f2_indices, sets_of_a_indices):
            W, pval, theta_est = test_partial_unconditional_relevance_prodecure(
                T, 
                thetas, 
                f2_indices, 
                a_indices
        )
            results[(tuple(f2_indices), tuple(a_indices))] = pval
        return results
    
    
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation)() for _ in tqdm(range(B), desc="simulating many times")
    )
    
    set_to_pval = {(tuple(_set_f), tuple(_set_a)): [] for _set_f, _set_a in zip(sets_of_f2_indices, sets_of_a_indices)}
    for result in all_results:
        for _set, pval in result.items():
            set_to_pval[_set].append(pval)
    
    result = {}
    for _set, pvals in set_to_pval.items():
        alphas_to_rej_freq = calculate_alphas_to_rejection_frequency(pvals, num_alphas)
        moments = " ".join(MOMENTS[i] for i in _set[0])
        name_thetas = " ".join(map(lambda x: f"theta[{x+1}]", _set[1]))
        name = f"[PARTIAL UNCONDITIONAL RELEVANCE]: {moments} FOR {name_thetas}"
        result[name] = alphas_to_rej_freq
    
    return result


def monte_carlo_partial_conditional_relevance_prodecure(
        T, 
        thetas, 
        sets_of_f2_indices, 
        sets_of_a_indices, 
        B, 
        num_alphas, 
        n_jobs=-1
    ):
    def run_simulation():
        results = {}
        for f2_indices, a_indices in zip(sets_of_f2_indices, sets_of_a_indices):
            W, pval, theta_est = test_partial_conditional_relevance_prodecure(
                T, 
                thetas, 
                f2_indices, 
                a_indices
        )
            results[(tuple(f2_indices), tuple(a_indices))] = pval
        return results
    
    
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation)() for _ in tqdm(range(B), desc="simulating many times")
    )
    
    set_to_pval = {(tuple(_set_f), tuple(_set_a)): [] for _set_f, _set_a in zip(sets_of_f2_indices, sets_of_a_indices)}
    for result in all_results:
        for _set, pval in result.items():
            set_to_pval[_set].append(pval)
    
    result = {}
    for _set, pvals in set_to_pval.items():
        alphas_to_rej_freq = calculate_alphas_to_rejection_frequency(pvals, num_alphas)
        moments = " ".join(MOMENTS[i] for i in _set[0])
        name_thetas = " ".join(map(lambda x: f"theta[{x+1}]", _set[1]))
        name = f"[PARTIAL CONDITIONAL RELEVANCE]: {moments} FOR {name_thetas}"
        result[name] = alphas_to_rej_freq
    
    return result


def plot_rejection_curves(
    results: dict[str, dict[float, float]],
    save_path: str | None = None,
    figsize_scale: float = 1.0,
    fill_vertical_zero_line: list = None
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
        if name.split(":")[-1].strip() in fill_vertical_zero_line:
            ax.plot(
                [0.0, 0.0],
                [0.0, freqs[0]],
                linestyle='--', 
                linewidth=1.6, 
                color=color
            )

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

        max_chars_per_line = 35
        words = name.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_chars_per_line:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        wrapped_title = '\n'.join(lines)
        ax.set_title(wrapped_title, fontsize=11, fontweight="semibold", pad=12)
        
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


