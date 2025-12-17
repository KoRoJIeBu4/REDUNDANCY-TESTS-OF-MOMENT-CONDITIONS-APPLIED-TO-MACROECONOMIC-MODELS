import numpy as np
from tests_impl import partial_conditional_relevance, partial_unconditional_relevance
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import autograd.numpy as anp
import torch


MOMENTS = ("CONST", "Z1", "Z2", "Z3", "Z4", "Z5")


# Порождаем C_ratio и R
def generate_CR_process(T, mu_c=0.001, phi_c=0.3, sigma_c=0.02, mu_r=0.001, phi_r=0.2, sigma_r=0.05, rho=0.3):
    c = torch.zeros(T+1, dtype=torch.float64)
    r = torch.zeros(T, dtype=torch.float64)

    for t in range(T):
        eps_c = torch.randn((), dtype=torch.float64)
        eps_r_indep = torch.randn((), dtype=torch.float64)
        eps_r = rho * eps_c + torch.sqrt(torch.tensor(1.0 - rho**2, dtype=torch.float64)) * eps_r_indep

        c[t+1] = mu_c + phi_c * c[t] + sigma_c * eps_c
        r[t] = mu_r + phi_r * c[t] + sigma_r * eps_r

    C_ratio = torch.exp(c[1:])
    R = torch.exp(r)

    return C_ratio, R


# Базовое уравнение Эйлера
def base_Euler_moment(beta, gamma, C_ratio, R):
    return torch.mean(beta * C_ratio ** (-gamma) * R - 1)


# Для заданных beta и gamma и других параметров находим такое mu_r, 
# чтобы выполнялось базовое уравнение Эйлера в среднем по наблюдениям
def calibrate_mu_r_and_return_original_data(beta, gamma,
                T=1000,
                mu_c=0.001, phi_c=0.3, sigma_c=0.02,
                mu_r_init=0.001, phi_r=0.2, sigma_r=0.05, rho=0.3,
                lr=0.1, epochs=200, display=False):

    mu_r = torch.tensor(mu_r_init, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([mu_r], lr=lr)

    pbar = tqdm(range(epochs), desc="Calibrating mu_r") if display else range(epochs)
    for it in pbar:
        opt.zero_grad()
        C_ratio, R = generate_CR_process(T,
                                        mu_c=mu_c, phi_c=phi_c, sigma_c=sigma_c,
                                        mu_r=mu_r, phi_r=phi_r, sigma_r=sigma_r, rho=rho)
        m = base_Euler_moment(beta, gamma, C_ratio, R)
        loss = m ** 2
        loss.backward()
        opt.step()
        if display:
            pbar.set_description(f"loss: {loss.item()}")
        if loss.item() < 1e-5:
            if display:
                pbar.set_description(f"Base moment: {m:.4f}, optimal mu_r: {mu_r:.4f}")
            break

    return C_ratio.detach(), R.detach()


def generate_Z_for_CR(
        beta, 
        gamma, 
        C_ratio, 
        R, 
        T=1000
):
    eps1 = torch.randn(T)
    eps2 = torch.randn(T)
    eps3 = torch.randn(T)
    eps4 = torch.randn(T)
    eps5 = torch.randn(T)
    

    # Производная момента по beta умноженная на инструмент
    grad_beta  = C_ratio ** (-gamma) * R
    # Производная момента по gamma умноженная на инструмент
    grad_gamma = -beta * C_ratio ** (-gamma) * R * torch.log(C_ratio)

    Z1 = 2 * grad_beta + 0.5 * grad_gamma + eps1            # релевантен для beta и gamma
    Z2 = 10 * grad_gamma + eps2                             # только gamma
    Z3 = 0.5 * grad_beta + eps3                             # только beta
    Z4 = eps4                                               # нерелевантен
    Z5 = eps5                                               # нерелевантен

    return Z1, Z2, Z3, Z4, Z5


def generate_data(
        beta, 
        gamma, 
        T,
        epochs=10000, 
        lr=0.1, 
        display_extra=False
    ):
    # Гарантируем выполнение уравнения потребления Эйлера в матожидании
    C_ratio, R = calibrate_mu_r_and_return_original_data(beta, gamma, T=T, lr=lr, epochs=epochs, display=display_extra)
    
    # Собираем инструменты, откалиброванные под тесты и ожидания от них
    Z1, Z2, Z3, Z4, Z5 = generate_Z_for_CR(
        beta, gamma, C_ratio, R, T=T
    )

    # Итого
    return {
        "R" : R.detach().cpu().numpy(),
        "C_ratio" : C_ratio.detach().cpu().numpy(),
        "Z1" : Z1.detach().cpu().numpy(),
        "Z2" : Z2.detach().cpu().numpy(),
        "Z3" : Z3.detach().cpu().numpy(),
        "Z4" : Z4.detach().cpu().numpy(),
        "Z5" : Z5.detach().cpu().numpy(),
        "CONST" : np.ones(T)
    }


def make_moment(name):
    def moment(theta, dp):
        beta, gamma = theta
        return (beta * dp['C_ratio'] ** (-gamma) * dp['R'] - 1) * dp[name]
    return moment


def test_partial_unconditional_relevance_prodecure(beta, gamma, T, f2_indices, a_indices):
    data = generate_data(beta, gamma, T)
    theta_init = np.array([0.0, 0.0])
    moments = [make_moment(name) for name in MOMENTS]
    return partial_unconditional_relevance(
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
        sets_of_f2_indices,
        sets_of_a_indices,
        B,
        num_alphas,
        n_jobs=-1
    ):
    beta = 0.97
    gamma = 5
    T = 250


    def run_simulation(beta, gamma, T):
        results = {}
        for f2_indices, a_indices in zip(sets_of_f2_indices, sets_of_a_indices):
            W, pval, theta_est = test_partial_unconditional_relevance_prodecure(
                beta, gamma, T,
                f2_indices,
                a_indices
            )
            results[(tuple(f2_indices), tuple(a_indices))] = pval
        return results
    
    
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation)(beta, gamma, T) for _ in tqdm(range(B), desc="simulating many times")
    )
    
    set_to_pval = {(tuple(_set_f), tuple(_set_a)): [] for _set_f, _set_a in zip(sets_of_f2_indices, sets_of_a_indices)}
    for result in all_results:
        for _set, pval in result.items():
            set_to_pval[_set].append(pval)
    
    result = {}
    for _set, pvals in set_to_pval.items():
        alphas_to_rej_freq = calculate_alphas_to_rejection_frequency(pvals, num_alphas)
        moments = " ".join(MOMENTS[i] for i in _set[0])
        name_thetas = " ".join(map(lambda x: "beta" if x == 0 else "gamma", _set[1]))
        name = f"[PARTIAL UNCONDITIONAL RELEVANCE]: {moments} FOR {name_thetas}"
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
        if fill_vertical_zero_line is not None and name.split(":")[-1].strip() in fill_vertical_zero_line:
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
