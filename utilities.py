import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import norm, chi2
import textwrap
from tqdm.auto import tqdm
from tests_impl import unconditional_relevance, conditional_relevance, partial_conditional_relevance, partial_unconditional_relevance
from statsmodels.tsa.stattools import adfuller
from itertools import combinations


def analyze_high_correlations(df, threshold=0.7):
    """
    Анализ признаков с высокой корреляцией
    
    Parameters:
    df: DataFrame с признаками
    threshold: порог корреляции (по модулю)
    """
    corr_matrix = df.corr()
    
    upper_tri_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs)
    
    total_pairs = len(high_corr_pairs)
    unique_features = set()
    for pair in high_corr_pairs:
        unique_features.add(pair['feature_1'])
        unique_features.add(pair['feature_2'])
    
    print(f"Порог корреляции: {threshold}")
    print(f"Количество пар с высокой корреляцией: {total_pairs}")
    print(f"Количество уникальных признаков в этих парах: {len(unique_features)}")
    
    if total_pairs > 0:
        print(f"\nТоп-10 самых сильных корреляций:")
        print(high_corr_df.sort_values('correlation', key=abs, ascending=False).head(10))
    
    return high_corr_df, corr_matrix


def plot_high_correlations(high_corr_df, threshold=0.7, top_n=20):
    """
    Визуализация сильных корреляций
    """
    if len(high_corr_df) == 0:
        print("Нет пар с корреляцией выше порога")
        return
    
    plot_df = high_corr_df.copy()
    plot_df['abs_corr'] = abs(plot_df['correlation'])
    plot_df = plot_df.sort_values('abs_corr', ascending=False).head(top_n)
    
    plot_df['pair'] = plot_df['feature_1'] + ' vs\n' + plot_df['feature_2']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    colors = ['red' if x < 0 else 'blue' for x in plot_df['correlation']]
    axes[0].barh(plot_df['pair'], plot_df['correlation'], color=colors)
    axes[0].set_xlabel('Корреляция')
    axes[0].set_title(f'Топ-{top_n} сильных корреляций (порог |r| > {threshold})')
    axes[0].axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=-threshold, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(high_corr_df['correlation'], bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Корреляция')
    axes[1].set_ylabel('Количество пар')
    axes[1].set_title(f'Распределение сильных корреляций (n={len(high_corr_df)})')
    axes[1].axvline(x=threshold, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(x=-threshold, color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap_with_threshold(corr_matrix, threshold=0.7):
    """
    Тепловая карта с выделением сильных корреляций
    """
    mask = np.abs(corr_matrix) < threshold
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r',
                center=0,
                vmin=-1, 
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title(f'Корреляционная матрица (показаны |r| > {threshold})', fontsize=14)
    plt.tight_layout()
    plt.show()


def select_features_by_correlation_threshold(df, threshold=0.7):
    """
    Отбирает признаки на основе корреляции. Отбирает сильно коррелирующие пары
    (их корреляция выше по модулю threshold). Затем в рамках этих пар начинает удалять признаки
    в порядке их встерчаемости. То есть мы максимизируем количество оставшихся признаков, 
    снижая эффект мультиколлинеарности

    
    Parameters:
    df: DataFrame с признаками (без целевой переменной)
    threshold: порог корреляции (по модулю)
    
    Returns:
    DataFrame с отобранными признаками
    """
    
    df_selected = df.copy()
    
    corr_matrix = df_selected.corr().abs()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if len(high_corr_pairs) == 0:
        print(f"Нет пар с корреляцией выше {threshold}")
        return df_selected
    
    print(f"Найдено {len(high_corr_pairs)} пар с корреляцией > {threshold}")
    
    remaining_pairs = high_corr_pairs.copy()
    features_to_drop = set()
    
    iteration = 1
    while remaining_pairs:
        feature_counts = {}
        for f1, f2, _ in remaining_pairs:
            feature_counts[f1] = feature_counts.get(f1, 0) + 1
            feature_counts[f2] = feature_counts.get(f2, 0) + 1
        
        if not feature_counts:
            break
        
        most_connected = max(feature_counts.items(), key=lambda x: x[1])
        feature_to_remove = most_connected[0]
        connections = most_connected[1]
        
        print(f"\nИтерация {iteration}:")
        print(f"  Удаляем '{feature_to_remove}' (участвует в {connections} сильных корреляциях)")
        
        features_to_drop.add(feature_to_remove)
        
        remaining_pairs = [(f1, f2, corr) for f1, f2, corr in remaining_pairs 
                          if f1 != feature_to_remove and f2 != feature_to_remove]
        
        print(f"  Осталось {len(remaining_pairs)} пар")
        iteration += 1
    
    df_result = df_selected.drop(columns=list(features_to_drop))
    
    print(f"\n" + "="*50)
    print(f"ИТОГ:")
    print(f"Исходное количество признаков: {df.shape[1]}")
    print(f"Удалено признаков: {len(features_to_drop)}")
    print(f"Осталось признаков: {df_result.shape[1]}")
    
    if len(features_to_drop) > 0:
        print(f"\nУдаленные признаки (по порядку удаления):")
        for i, feat in enumerate(features_to_drop):
            print(f"  {i+1}. {feat}")
    
    if len(remaining_pairs) > 0:
        print(f"\n  Внимание: осталось {len(remaining_pairs)} сильных корреляций!")
        print("Топ-3 оставшихся:")
        for f1, f2, corr in sorted(remaining_pairs, key=lambda x: x[2], reverse=True)[:3]:
            print(f"  {f1} - {f2}: {corr:.3f}")
    else:
        print(f"\n Все сильные корреляции устранены!")
    
    var_original = df.var().sum()
    var_result = df_result.var().sum()
    print(f"\nСохранено дисперсии: {var_result/var_original*100:.1f}%")
    
    return df_result


def plot_theta_estimates(
    theta_hat: np.ndarray,
    cov_theta: np.ndarray,
    alpha: float = 0.05,
    param_names: list = None,
    figsize: tuple = (12, 6)
):
    """
    Визуализация оценок параметров с нормальной аппроксимацией и доверительными интервалами.

    Parameters
    ----------
    theta_hat : np.ndarray
        Оценки параметров (k,)
    cov_theta : np.ndarray
        Ковариационная матрица оценок (k x k)
    alpha : float
        Уровень значимости (по умолчанию 0.05 → 95% ДИ)
    param_names : list[str]
        Названия параметров
    figsize : tuple
        Размер фигуры
    """

    theta_hat = np.asarray(theta_hat)
    cov_theta = np.asarray(cov_theta)

    k = theta_hat.size
    se = np.sqrt(np.diag(cov_theta))
    z = norm.ppf(1 - alpha / 2)

    if param_names is None:
        param_names = [f"$\\theta_{i}$" for i in range(k)]

    ncols = min(3, k)
    nrows = int(np.ceil(k / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i in range(k):
        ax = axes[i]

        mu = theta_hat[i]
        sigma = se[i]

        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
        y = norm.pdf(x, mu, sigma)

        ci_lower = mu - z * sigma
        ci_upper = mu + z * sigma

        ax.plot(x, y)
        ax.axvline(mu, linestyle='--')
        ax.axvspan(ci_lower, ci_upper, alpha=0.2)

        ax.set_title(param_names[i])
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")

        ax.text(
            0.02,
            0.95,
            f"Estimate = {mu:.4f}\nSE = {sigma:.4f}\nCI = [{ci_lower:.4f}, {ci_upper:.4f}]",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9
        )

    for j in range(k, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Parameter Estimates with {(1-alpha)*100:.0f}% Confidence Intervals",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


def wald_test_pairwise(thetas, covs, names=None):
    """
    Pairwise Wald tests for equality of parameter vectors.
    """

    n = len(thetas)

    if names is None:
        names = [f"model_{i+1}" for i in range(n)]

    results = []

    for i in range(n):
        for j in range(i+1, n):

            theta_i = np.array(thetas[i])
            theta_j = np.array(thetas[j])

            V = covs[i] + covs[j]

            diff = theta_i - theta_j

            W = diff.T @ np.linalg.inv(V) @ diff

            pval = 1 - chi2.cdf(W, df=2)

            results.append({
                "model_1": names[i],
                "model_2": names[j],
                "wald_stat": W,
                "p_value": pval
            })

    return pd.DataFrame(results)


def plot_gmm_estimates(thetas, covs, names=None, conf=0.95):

    n = len(thetas)

    if names is None:
        names = [f"model_{i+1}" for i in range(n)]

    names_wrapped = ["\n".join(textwrap.wrap(nm, 22)) for nm in names]

    thetas = np.array(thetas)

    betas = thetas[:,0]
    gammas = thetas[:,1]

    beta_se = np.sqrt([c[0][0] for c in covs])
    gamma_se = np.sqrt([c[1][1] for c in covs])

    z = np.sqrt(chi2.ppf(conf,1))

    colors = plt.cm.tab10.colors
    x = np.arange(n)

    plt.figure(figsize=(13,7))

    for i in range(n):

        low = betas[i] - z*beta_se[i]
        high = betas[i] + z*beta_se[i]

        c = colors[i % len(colors)]

        plt.vlines(x[i], low, high, colors=c, linewidth=2)

        plt.hlines(low, x[i]-0.25, x[i]+0.25, colors=c, linestyles="dashed")
        plt.hlines(high, x[i]-0.25, x[i]+0.25, colors=c, linestyles="dashed")

        plt.scatter(x[i], betas[i], color=c, s=90, zorder=3)

    plt.xticks(x, names_wrapped)
    plt.ylabel("beta")
    plt.title(f"Beta estimates ({int(conf*100)}% CI)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(13,7))

    for i in range(n):

        low = gammas[i] - z*gamma_se[i]
        high = gammas[i] + z*gamma_se[i]

        c = colors[i % len(colors)]

        plt.vlines(x[i], low, high, colors=c, linewidth=2)

        plt.hlines(low, x[i]-0.25, x[i]+0.25, colors=c, linestyles="dashed")
        plt.hlines(high, x[i]-0.25, x[i]+0.25, colors=c, linestyles="dashed")

        plt.scatter(x[i], gammas[i], color=c, s=90, zorder=3)

    plt.xticks(x, names_wrapped)
    plt.ylabel("gamma")
    plt.title(f"Gamma estimates ({int(conf*100)}% CI)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12,8))

    chi_val = chi2.ppf(conf,2)

    for i,(theta,cov,name) in enumerate(zip(thetas,covs,names)):

        vals, vecs = np.linalg.eigh(cov)

        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))

        width,height = 2*np.sqrt(vals*chi_val)

        c = colors[i % len(colors)]

        ell = Ellipse(
            xy=theta,
            width=width,
            height=height,
            angle=angle,
            alpha=0.25,
            color=c
        )

        plt.gca().add_patch(ell)

        plt.scatter(theta[0],theta[1],color=c,s=90)

        plt.annotate(
            name,
            (theta[0],theta[1]),
            xytext=(6,6),
            textcoords="offset points"
        )

    plt.xlabel("beta")
    plt.ylabel("gamma")
    plt.title(f"Joint confidence region ({int(conf*100)}%)")

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def filter_stationary_series(
    df: pd.DataFrame,
    alpha: float = 0.05,
    regression: str = "c",
    autolag: str = "AIC",
    verbose: bool = True
):
    """
    Фильтрация стационарных временных рядов через тест Дики-Фуллера.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица временных рядов
    alpha : float
        Уровень значимости
    regression : str
        Тип детерминированной части:
        "c"  — константа
        "ct" — константа + тренд
        "n"  — без константы
    autolag : str
        Метод выбора лагов
    verbose : bool
        Печатать отчёт

    Returns
    -------
    df_stationary : pd.DataFrame
        Датафрейм только со стационарными рядами
    report : pd.DataFrame
        Таблица результатов теста
    """

    results = []

    for col in df.columns:

        series = df[col].dropna()

        try:
            stat, pvalue, lags, nobs, *_ = adfuller(
                series,
                regression=regression,
                autolag=autolag
            )

            stationary = pvalue < alpha

        except Exception:
            stat = None
            pvalue = None
            stationary = False

        results.append({
            "variable": col,
            "ADF_stat": stat,
            "p_value": pvalue,
            "stationary": stationary
        })

    report = pd.DataFrame(results)

    stationary_cols = report.loc[
        report["stationary"], "variable"
    ].tolist()

    df_stationary = df[stationary_cols].copy()

    if verbose:

        print("\nADF Stationarity Filtering")
        print("=" * 40)
        print(f"Significance level: {alpha}")
        print()

        print("Stationary variables:")
        for v in stationary_cols:
            print("  ✓", v)

        removed = report.loc[~report["stationary"], "variable"].tolist()

        if removed:
            print("\nRemoved non-stationary variables:")
            for v in removed:
                print("  ✗", v)

    return df_stationary, report


def implement_URTA(input: pd.DataFrame, name: str, significance_level: float =0.05):
    '''
    Функция для формирования отчета по URTA
    input: pd.DataFrame
        Входной датасет, в котором обязательно есть колонки [C[t], R[t+1], C[t+1], C_ratio] и поле date является индексом
    name: str
        Название
    significance_level: float
        Уровень значимости
    '''
    assert isinstance(input.index, pd.DatetimeIndex)
    df = input.copy()

    cols = [col for col in df.columns if col not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]
    data = {
        "R[t+1]" : df['R[t+1]'],
        "C_ratio" : df['C_ratio'],
        "Const" : np.ones(len(df)),
    }
    for col in cols:
        data[col] = df[col]
    
    def make_moment(name):
        def moment(theta, dp):
            beta, gamma = theta
            m = beta * (dp['C_ratio'] ** (-gamma)) * (1 + dp['R[t+1]']) - 1
            return m * dp[name]
        return moment
    MOMENT_NAMES = [col for col in data if col not in ("R[t+1]", "C_ratio")]
    moments = [make_moment(name) for name in MOMENT_NAMES]

    results = {}
    for i, moment in tqdm(list(enumerate(MOMENT_NAMES))):
        W, pval, theta, cov = unconditional_relevance(
            data=data,
            moments=moments,
            f2_indexes=[i],
            theta_init=[0, 0],
        )
        results[moment] = {
            "W" : W,
            "logW" : np.log(W),
            "p_value" : pval
        }

    results = pd.DataFrame(results).T.sort_values(by=['p_value'])
    results['group'] = name
    results['relevance'] = results['p_value'].apply(lambda x: "🟢" if x < significance_level else "🔴")
    results = results.reset_index().rename(columns={"index" : "moment"}).sort_values(by="logW", ascending=False)

    return results


def plot_moment_relevance(df, title=None):

    df = df.sort_values("logW", ascending=True)

    colors = df["relevance"].map({
        "🟢": "#2C7BB6",
        "🔴": "#D7191C"
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        df["moment"],
        df["logW"],
        color=colors,
        alpha=0.9
    )

    for i, (logw, p) in enumerate(zip(df["logW"], df["p_value"])):
        ax.text(
            logw + 0.05,
            i,
            f"p={p:.3f}",
            va="center",
            fontsize=9
        )

    ax.set_xlabel("log(W statistic)")
    ax.set_ylabel("Moment condition")

    if title is not None:
        ax.set_title(title)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2C7BB6", label="Relevant"),
        Patch(facecolor="#D7191C", label="Not relevant")
    ]

    ax.legend(handles=legend_elements)

    sns.despine()

    plt.tight_layout()
    plt.show()


def implement_CRTA(input: pd.DataFrame,
                   URTA_report: pd.DataFrame,
                   significance_level: float = 0.05) -> pd.DataFrame:
    """
    Реализация CRTA (Conditional Relevance Test Algorithm).

    Параметры
    ----------
    input : pd.DataFrame
        Исходные данные. Должны содержать R[t+1], C_ratio и набор инструментов.

    URTA_report : pd.DataFrame
        Таблица результатов URTA со столбцами:
        ['moment', 'W', 'logW', 'p_value', 'group'].

    significance_level : float
        Уровень значимости для теста условной релевантности.

    Алгоритм
    ----------
    1. Отбираем безусловно релевантные моменты (p_value < significance_level).
    2. Для каждого момента i:
        - перебираем все комбинации остальных моментов
        - запускаем conditional_relevance
        - если p_value < significance_level — момент условно релевантен
        - среди всех таких комбинаций выбираем ту,
          где статистика W максимальна.
    3. Формируем матрицу M размера (k × k):
        строки — момент i,
        столбцы — моменты j.

        M[i,j] = 1  если момент j входит в оптимальный блок,
        при котором момент i максимально условно релевантен.

        иначе 0.

    Возвращает
    ----------
    pd.DataFrame
        Матрица включения моментов (0/1).
    """

    df = input.copy()

    cols = [c for c in df.columns if c not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]

    data = {
        "R[t+1]": df["R[t+1]"],
        "C_ratio": df["C_ratio"],
        "Const": np.ones(len(df)),
    }

    for col in cols:
        data[col] = df[col]

    def make_moment(name):
        def moment(theta, dp):
            beta, gamma = theta
            m = beta * (dp["C_ratio"] ** (-gamma)) * (1 + dp["R[t+1]"]) - 1
            return m * dp[name]
        return moment

    MOMENT_NAMES = [c for c in data if c not in ("R[t+1]", "C_ratio")]
    moments = [make_moment(name) for name in MOMENT_NAMES]

    relevant_moments = URTA_report[
        URTA_report["p_value"] < significance_level
    ]["moment"].tolist()

    result_matrix = pd.DataFrame(
        0,
        index=relevant_moments,
        columns=relevant_moments
    )

    for moment_i in tqdm(relevant_moments):

        others = [m for m in relevant_moments if m != moment_i]

        best_W = -np.inf
        best_block = None

        for r in range(len(others) + 1):

            for combo in combinations(others, r):

                block = list(combo) + [moment_i]

                f2_indexes = [MOMENT_NAMES.index(m) for m in block]

                if len(moments) - len(f2_indexes) < 2:
                    continue

                W, pval, theta, cov = conditional_relevance(
                    data=data,
                    moments=moments,
                    f2_indexes=f2_indexes,
                    theta_init=[0, 0],
                )

                if pval < significance_level:

                    if W > best_W:
                        best_W = W
                        best_block = block

        if best_block is not None:

            for m in best_block:
                if m in result_matrix.columns:
                    result_matrix.loc[moment_i, m] = 1

    return result_matrix