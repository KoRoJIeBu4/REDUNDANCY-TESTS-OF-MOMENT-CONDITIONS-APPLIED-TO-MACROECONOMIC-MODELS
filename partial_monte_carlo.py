import numpy as np
from tests_impl import partial_conditional_relevance, partial_unconditional_relevance
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import autograd.numpy as anp
from multiprocessing import Pool
import torch
import os
import pickle


# Производная момента по beta умноженная на инструмент
def grad_beta_by_z(gamma, C_ratio, R, Z):
    return torch.mean(C_ratio ** (-gamma) * R * Z)


# Производная момента по gamma умноженная на инструмент
def grad_gamma_by_z(beta, gamma, C_ratio, R, Z):
    return torch.mean(-beta * C_ratio ** (-gamma) * R * Z * torch.log(C_ratio))


# Лосс, который уменьшается чем ближе аргументы/критерии к нулю
def direct_to_zero(*args):
    loss = 0.0
    for arg in args:
        loss = loss + arg ** 2
    return loss


# Лосс, который уменьшается чем дальше аргументы/критерии от нуля
def direct_from_zero(*args):
    loss = 0.0
    for arg in args:
        current_loss = torch.pi / 2 - torch.atan(torch.abs(arg))
        loss = loss + 10 * current_loss
    return loss


# Loss, минимизирующий корреляцию между инструментами
def decorrelate(*Zs, threshold=0.6):    
    Z = torch.cat(Zs, dim=1)
    Z_mean = Z.mean(dim=0)
    Z_centered = Z - Z_mean
    cov = Z_centered.T @ Z_centered / Z.shape[0]
    off_diag = cov - torch.diag(torch.diag(cov))
    off_diag = torch.where(torch.abs(off_diag) < threshold, 0, off_diag)
    return 0.1 * off_diag.mean() ** 2


# Пораждаем C_ratio и R
def generate_CR_process(T, mu_c=0.001, phi_c=0.3, sigma_c=0.02, mu_r=0.001, phi_r=0.2, sigma_r=0.05, rho=0.3):
    c = torch.zeros(T+1, dtype=torch.float64)
    r = torch.zeros(T, dtype=torch.float64)

    for t in range(T):
        eps_c = torch.randn((), dtype=torch.float64)
        eps_r_indep = torch.randn((), dtype=torch.float64)
        eps_r = rho * eps_c + torch.sqrt(torch.tensor(1.0 - rho**2, dtype=torch.float64)) * eps_r_indep

        c[t+1] = mu_c + phi_c * c[t] + sigma_c * eps_c
        r[t]   = mu_r + phi_r * c[t] + sigma_r * eps_r

    C_ratio = torch.exp(c[1:])
    R = torch.exp(r)

    return C_ratio, R


# Базовое уравнение Эйлера
def base_Euler_moment(beta, gamma, C_ratio, R):
    return torch.mean(beta * C_ratio ** (-gamma) * R - 1)


# Для заданных beta и gamma и других параметров находим такое mu_r, чтобы выполнялось базовое уравнение Эйлера в среднем по наблюдениям
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


# Генерируем инструменты, закладывая в них ожидаемые на процедуре тестирования эффекты
# Также вводим дополнительный loss за высокую корреляцию между ними
def generate_Z_for_CR(beta, gamma, C_ratio, R, T=1000, lr=0.1, epochs=1000, reduce_corr=False, display=False):
    Z1 = torch.randn(T, requires_grad=True, dtype=torch.float64) # relevant for gamma and beta
    Z2 = torch.randn(T, requires_grad=True, dtype=torch.float64) # relevant only for gamma
    Z3 = torch.randn(T, requires_grad=True, dtype=torch.float64) # relevant only for beta
    Z4 = torch.randn(T, requires_grad=True, dtype=torch.float64) # irrelevant at all
    Z5 = torch.randn(T, requires_grad=True, dtype=torch.float64) # irrelevant at all

    optimizer = torch.optim.Adam([Z1, Z2, Z3, Z4, Z5], lr=lr)
    pbar = tqdm(range(epochs), desc="Calibrating Z's") if display else range(epochs)
    for epoch in pbar:
        optimizer.zero_grad()

        # 1. Для Z1
        z1_to_beta = grad_beta_by_z(gamma, C_ratio, R, Z1)          # убегает от нуля
        z1_to_gamma = grad_gamma_by_z(beta, gamma, C_ratio, R, Z1)  # убегает от нуля
    
        # 2. Для Z2
        z2_to_beta = grad_beta_by_z(gamma, C_ratio, R, Z2)          # стремится к нулю
        z2_to_gamma = grad_gamma_by_z(beta, gamma, C_ratio, R, Z2)  # убегает от нуля
    
        # 3. Для Z3
        z3_to_beta = grad_beta_by_z(gamma, C_ratio, R, Z3)          # убегает от нуля
        z3_to_gamma = grad_gamma_by_z(beta, gamma, C_ratio, R, Z3)  # стремится к нулю
    
        # 4. Для Z4
        z4_to_beta = grad_beta_by_z(gamma, C_ratio, R, Z4)          # стремится к нулю
        z4_to_gamma = grad_gamma_by_z(beta, gamma, C_ratio, R, Z4)  # стремится к нулю
    
        # 5. Для Z5
        z5_to_beta = grad_beta_by_z(gamma, C_ratio, R, Z5)          # стремится к нулю
        z5_to_gamma = grad_gamma_by_z(beta, gamma, C_ratio, R, Z5)  # стремится к нулю

        loss_to_zero = direct_to_zero(z2_to_beta, z3_to_gamma, z4_to_beta, z4_to_gamma, z5_to_beta, z5_to_gamma)
        loss_from_zero = direct_from_zero(z1_to_beta, z1_to_gamma, z2_to_gamma, z3_to_beta)
        loss = loss_to_zero + loss_from_zero

        if reduce_corr:
            loss_of_high_correlation = decorrelate(
                Z1.view(-1, 1), Z2.view(-1, 1), Z3.view(-1, 1), Z4.view(-1, 1), Z5.view(-1, 1)
            )
            loss = loss + loss_of_high_correlation
        
        loss.backward()
        optimizer.step()

        if reduce_corr and display:
            pbar.set_description(
                f"loss: {loss.item():.4f}, " 
                f"loss_to_zero: {loss_to_zero.item():.4f}, "
                f"loss_from_zero: {loss_from_zero.item():.4f}, "
                f"loss_of_high_correlation: {loss_of_high_correlation.item():.4f}"
            )
        elif reduce_corr == False and display:
             pbar.set_description(
                f"loss: {loss.item():.4f}, " 
                f"loss_to_zero: {loss_to_zero.item():.4f}, "
                f"loss_from_zero: {loss_from_zero.item():.4f}"
            )

    return Z1, Z2, Z3, Z4, Z5, loss_to_zero.item(), loss_from_zero.item()


def generate_data(beta, gamma, T, epochs=10000, lr=0.1, display=False, display_extra=False, reduce_corr=False):
    # Гарантируем выполнение уравнения потребления Эйлера в матожидании
    C_ratio, R = calibrate_mu_r_and_return_original_data(beta, gamma, T=T, lr=lr, epochs=epochs, display=display_extra)

    # Собираем основные наблюдения
    # C_ratio, R = generate_CR_process(T, mu_r=mu_r)
    
    # Собираем инструменты
    Z1, Z2, Z3, Z4, Z5, loss_to_zero, loss_from_zero = generate_Z_for_CR(
        beta, gamma, C_ratio, R, T=T, lr=lr, epochs=epochs, reduce_corr=reduce_corr, display=display
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
    }, loss_to_zero, loss_from_zero


# ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ [B] ДАННЫХ ДЛЯ СИМУЛЯЦИИ
def prepare_data_for_simulation(B=1000, out_dir='data/partial_data_for_testing_procedures'):
    beta = 0.97
    gamma = 5
    T = 250
    os.makedirs(out_dir, exist_ok=True)
    losses_to_zero = []
    losses_from_zero = []
    losses_total = []

    for idx in tqdm(range(B), desc="Collecting data in main function"):
        data, loss_to_zero, loss_from_zero = generate_data(beta=beta, gamma=gamma, T=T, display=True)
        filename = os.path.join(out_dir, f"sim_{idx:06d}.pkl")
        with open(filename, mode="wb") as file:
            pickle.dump(data, file)
        losses_to_zero.append(loss_to_zero)
        losses_from_zero.append(loss_from_zero)
        losses_total.append(loss_to_zero + loss_from_zero)

    losses_to_zero = np.array(losses_to_zero)
    losses_from_zero = np.array(losses_from_zero)
    losses_total = np.array(losses_total)

    plt.figure(figsize=(10, 6))
    x = np.arange(B)

    plt.plot(x, losses_total, label="Total loss", color="black", linewidth=2)
    plt.plot(x, losses_to_zero, label="Loss to zero", color="tab:blue", alpha=0.8)
    plt.plot(x, losses_from_zero, label="Loss from zero", color="tab:orange", alpha=0.8)

    plt.xlabel("Simulation index")
    plt.ylabel("Loss value")
    plt.title("Losses across simulations")
    plt.legend()
    plt.grid(True, alpha=0.3)

    fig_path = os.path.join(out_dir, "losses_over_simulations.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


# def test_partial_unconditional_relevance_prodecure(T, thetas, f2_indices, a_indices):
#     data = create_observations(T, *thetas)
#     theta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
#     moments = [make_moment(name) for name in MOMENTS]
#     return partial_unconditional_relevance(
#         data=data, 
#         moments=moments,
#         f2_indexes=f2_indices, 
#         a_indexes=a_indices,
#         theta_init=theta_init
#     )


# def test_partial_conditional_relevance_prodecure(T, thetas, f2_indices, a_indices):
#     data = create_observations(T, *thetas)
#     theta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
#     moments = [make_moment(name) for name in MOMENTS]
#     return partial_conditional_relevance(
#         data=data, 
#         moments=moments,
#         f2_indexes=f2_indices, 
#         a_indexes=a_indices,
#         theta_init=theta_init
#     )


# def calculate_rejection_frequency(pvalues, alpha):
#     return sum(p <= alpha for p in pvalues) / len(pvalues)


# def calculate_alphas_to_rejection_frequency(pvalues, num_alphas):
#     alphas = np.linspace(0, 1, num_alphas)
#     alphas_to_rej_freq = {}
#     for alpha in alphas:
#         alphas_to_rej_freq[alpha] = calculate_rejection_frequency(pvalues, alpha)
#     return alphas_to_rej_freq


# def monte_carlo_partial_unconditional_relevance_prodecure(
#         T, 
#         thetas, 
#         sets_of_f2_indices, 
#         sets_of_a_indices, 
#         B, 
#         num_alphas, 
#         n_jobs=-1
#     ):
#     def run_simulation():
#         results = {}
#         for f2_indices, a_indices in zip(sets_of_f2_indices, sets_of_a_indices):
#             W, pval, theta_est = test_partial_unconditional_relevance_prodecure(
#                 T, 
#                 thetas, 
#                 f2_indices, 
#                 a_indices
#         )
#             results[(tuple(f2_indices), tuple(a_indices))] = pval
#         return results
    
    
#     all_results = Parallel(n_jobs=n_jobs)(
#         delayed(run_simulation)() for _ in tqdm(range(B), desc="simulating many times")
#     )
    
#     set_to_pval = {(tuple(_set_f), tuple(_set_a)): [] for _set_f, _set_a in zip(sets_of_f2_indices, sets_of_a_indices)}
#     for result in all_results:
#         for _set, pval in result.items():
#             set_to_pval[_set].append(pval)
    
#     result = {}
#     for _set, pvals in set_to_pval.items():
#         alphas_to_rej_freq = calculate_alphas_to_rejection_frequency(pvals, num_alphas)
#         moments = " ".join(MOMENTS[i] for i in _set[0])
#         name_thetas = " ".join(map(lambda x: f"theta[{x+1}]", _set[1]))
#         name = f"[PARTIAL UNCONDITIONAL RELEVANCE]: {moments} FOR {name_thetas}"
#         result[name] = alphas_to_rej_freq
    
#     return result


# def monte_carlo_partial_conditional_relevance_prodecure(
#         T, 
#         thetas, 
#         sets_of_f2_indices, 
#         sets_of_a_indices, 
#         B, 
#         num_alphas, 
#         n_jobs=-1
#     ):
#     def run_simulation():
#         results = {}
#         for f2_indices, a_indices in zip(sets_of_f2_indices, sets_of_a_indices):
#             W, pval, theta_est = test_partial_conditional_relevance_prodecure(
#                 T, 
#                 thetas, 
#                 f2_indices, 
#                 a_indices
#         )
#             results[(tuple(f2_indices), tuple(a_indices))] = pval
#         return results
    
    
#     all_results = Parallel(n_jobs=n_jobs)(
#         delayed(run_simulation)() for _ in tqdm(range(B), desc="simulating many times")
#     )
    
#     set_to_pval = {(tuple(_set_f), tuple(_set_a)): [] for _set_f, _set_a in zip(sets_of_f2_indices, sets_of_a_indices)}
#     for result in all_results:
#         for _set, pval in result.items():
#             set_to_pval[_set].append(pval)
    
#     result = {}
#     for _set, pvals in set_to_pval.items():
#         alphas_to_rej_freq = calculate_alphas_to_rejection_frequency(pvals, num_alphas)
#         moments = " ".join(MOMENTS[i] for i in _set[0])
#         name_thetas = " ".join(map(lambda x: f"theta[{x+1}]", _set[1]))
#         name = f"[PARTIAL CONDITIONAL RELEVANCE]: {moments} FOR {name_thetas}"
#         result[name] = alphas_to_rej_freq
    
#     return result


# def plot_rejection_curves(
#     results: dict[str, dict[float, float]],
#     save_path: str | None = None,
#     figsize_scale: float = 1.0,
#     fill_vertical_zero_line: list = None
# ):
#     if len(results) == 0:
#         return

#     plt.style.use('seaborn-v0_8-whitegrid')

#     n_tests = len(results)
#     rows = int(np.ceil(np.sqrt(n_tests)))
#     cols = int(np.ceil(n_tests / rows))

#     fig, axes = plt.subplots(
#         rows, cols,
#         figsize=(cols * 4.2 * figsize_scale, rows * 3.2 * figsize_scale),
#         dpi=150
#     )
#     axes = axes.flatten() if n_tests > 1 else [axes]

#     colors = plt.cm.tab10(np.linspace(0, 1, n_tests))

#     for ax, (name, data), color in zip(axes, results.items(), colors):
#         alphas = np.array(sorted(data.keys()))
#         freqs = np.array([data[a] for a in alphas])
#         if fill_vertical_zero_line and name.split(":")[-1].strip() in fill_vertical_zero_line:
#             ax.plot(
#                 [0.0, 0.0],
#                 [0.0, freqs[0]],
#                 linestyle='--', 
#                 linewidth=1.6, 
#                 color=color
#             )

#         ax.plot(
#             alphas,
#             freqs,
#             linestyle="--",
#             linewidth=1.6,
#             color=color
#         )

#         ax.plot(
#             [0, 1],
#             [0, 1],
#             linestyle=":",
#             linewidth=1,
#             color="black",
#             alpha=0.2
#         )

#         max_chars_per_line = 35
#         words = name.split()
#         lines = []
#         current_line = []
        
#         for word in words:
#             if len(' '.join(current_line + [word])) <= max_chars_per_line:
#                 current_line.append(word)
#             else:
#                 if current_line:
#                     lines.append(' '.join(current_line))
#                 current_line = [word]
        
#         if current_line:
#             lines.append(' '.join(current_line))
        
#         wrapped_title = '\n'.join(lines)
#         ax.set_title(wrapped_title, fontsize=11, fontweight="semibold", pad=12)
        
#         ax.set_xlabel("Significance Level", fontsize=10)
#         ax.set_ylabel("Rejection Frequency", fontsize=10)

#         ax.set_xlim(-0.02, 1.02)
#         ax.set_ylim(-0.02, 1.02)

#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#         ax.grid(True, alpha=0.25)
#         ax.margins(x=0.05, y=0.05)

#     for i in range(n_tests, len(axes)):
#         axes[i].axis("off")

#     plt.tight_layout()

#     if save_path is not None:
#         if save_path.endswith(".svg"):
#             plt.savefig(save_path, format="svg")
#         else:
#             plt.savefig(save_path, bbox_inches="tight")

#     plt.show()


