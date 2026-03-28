import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm, chi2
import textwrap
from tqdm.auto import tqdm
from tests_impl import unconditional_relevance, conditional_relevance, partial_conditional_relevance, partial_unconditional_relevance
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
import re
from typing import Dict, Any, Tuple, Optional, List


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


def implement_CRTA(input: pd.DataFrame, name: str, significance_level: float = 0.05):
    '''
    Функция для формирования отчета по CRTA
    input: pd.DataFrame
        Входной датасет, в котором обязательно есть колонки [C[t], R[t+1], C[t+1], C_ratio] и поле date является индексом
    name: str
        Название
    significance_level: float
        Уровень значимости
    
    Идея:
        Вот мы имеем набор инструментов. Для начала мы можем выделить все "базисные инструменты"
        "Базисные" инструменты (🟢) - это те, для которых тест на conditional_relevance дал результат, что они условно релевантны к всему исходному набору
        Для "оставшихся" интструментов хочется найти те, через которые они линейно выражаются в терминах якобианов других моментов
        Возможны такие ситуации (z - текущий инструмент из "остальных", то есть "небазовый"):
         1) z условно релевантен для базисных и условно НЕрелевантен для остальных (🟡)
         2) z условно НЕрелевантен для базисных и условно релевантен для остальных (🟠)
         3) z условно НЕрелевантен для базисных и условно НЕрелевантен для остальных (🔴)
    '''
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
    conditionally_relevant_moments = []
    conditionally_irrelevant_moments = []

    for i in tqdm(range(len(moments)), desc="[stage 1]: determine basics"):
        W, pval, theta, cov = conditional_relevance(
            data=data,
            moments=moments,
            f2_indexes=[i],
            theta_init=[0, 0],
        )
        if pval <= significance_level:
            conditionally_relevant_moments.append((MOMENT_NAMES[i], W, pval, "🟢"))
        else:
            conditionally_irrelevant_moments.append((MOMENT_NAMES[i], W, pval))

    irrelevant_due_to_basics = []
    irrelevant_due_to_others = []
    irrelevant_due_to_both = []
    
    

    for z in tqdm([z[0] for z in conditionally_irrelevant_moments], desc="[stage 2]: determine rest instruments"):
        idx = MOMENT_NAMES.index(z)

        # Прогоняем z через базисные инструменты
        basic_moments = [moments[i] for i in range(len(moments)) if MOMENT_NAMES[i] in [z[0] for z in conditionally_relevant_moments]]
        if len(basic_moments) <= 1:
            W_basic, pval_basic = None, float("inf")
        else:
            basic_moments.append(moments[idx])
            W_basic, pval_basic, _, _ = conditional_relevance(
                data=data,
                moments=basic_moments,
                f2_indexes=[len(basic_moments)-1],
                theta_init=[0, 0],
            )

        # Прогоняем z через оставшиеся инструменты
        rest_moments = [moments[i] for i in range(len(moments)) if MOMENT_NAMES[i] in [z[0] for z in conditionally_irrelevant_moments] and z != MOMENT_NAMES[i]]
        rest_moments.append(moments[idx])
        W_rest, pval_rest, _, _ = conditional_relevance(
            data=data,
            moments=rest_moments,
            f2_indexes=[len(rest_moments)-1],
            theta_init=[0, 0],
        )
        
        if pval_basic > significance_level and pval_rest < significance_level:
            if W_basic is not None:
                irrelevant_due_to_basics.append((z, W_basic, pval_basic, "🟠"))
                                             
        if pval_basic < significance_level and pval_rest > significance_level:
            irrelevant_due_to_others.append((z, W_rest, pval_rest, "🟡"))
                                             
        if pval_basic > significance_level and pval_rest > significance_level:
            irrelevant_due_to_both.append((z, W_rest, pval_rest, "🔴"))
                                             

    report = conditionally_relevant_moments
    report.extend(irrelevant_due_to_others)
    report.extend(irrelevant_due_to_basics)
    report.extend(irrelevant_due_to_both)

    report = pd.DataFrame(report, columns=["moment", "W", "p_value", "conditional_relevance"])
    report['logW'] = np.log(report['W'])
    report['group'] = name
    report = report[['moment', 'W', 'logW', 'p_value', 'group',	'conditional_relevance']]

    return report


def implement_PURTA(input: pd.DataFrame, name: str, significance_level: float =0.05):
    '''
    Функция для формирования отчета по PURTA
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
        W_beta, pval_beta, _, _ = partial_unconditional_relevance(
            data=data,
            moments=moments,
            f2_indexes=[i],
            a_indexes=[0],
            theta_init=[0, 0],
        )

        W_gamma, pval_gamma, _, _ = partial_unconditional_relevance(
            data=data,
            moments=moments,
            f2_indexes=[i],
            a_indexes=[1],
            theta_init=[0, 0],
        )

        W, pval, _, _ = partial_unconditional_relevance(
            data=data,
            moments=moments,
            f2_indexes=[i],
            a_indexes=[0, 1],
            theta_init=[0, 0],
        )
        
        results[moment] = {
            "logW_beta" : np.log(W_beta),
            "logW_gamma" : np.log(W_gamma),
            "logW" : np.log(W),
            "p_value_beta" : pval_beta,
            "p_value_gamma" : pval_gamma,
            "p_value" : pval
        }

    results = pd.DataFrame(results).T
    results['group'] = name
    results['relevance_for_beta'] = results['p_value_beta'].apply(lambda x: "🟢" if x < significance_level else "🔴")
    results['relevance_for_gamma'] = results['p_value_gamma'].apply(lambda x: "🟢" if x < significance_level else "🔴")
    results['relevance'] = results['p_value'].apply(lambda x: "🟢" if x < significance_level else "🔴")
    results = results.reset_index().rename(columns={"index" : "moment"}).sort_values(by=["logW_beta", "logW_gamma"], ascending=[False, False])

    return results


def plot_moment_relevance_by_beta_and_gamma(df, title=None, significance_level=0.05):
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    plot_df = df.copy()

    plot_df["beta_rel"] = plot_df["p_value_beta"] < significance_level
    plot_df["gamma_rel"] = plot_df["p_value_gamma"] < significance_level
    plot_df["joint_rel"] = plot_df["p_value"] < significance_level

    color_map = {
        True: "#4C72B0",
        False: "#DD8452"
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

    configs = [
        ("logW_beta", "p_value_beta", "beta_rel", "Relevance for β"),
        ("logW_gamma", "p_value_gamma", "gamma_rel", "Relevance for γ"),
    ]

    xmin = min(df["logW_beta"].min(), df["logW_gamma"].min())
    xmax = max(df["logW_beta"].max(), df["logW_gamma"].max())

    for ax, (logw_col, p_col, rel_col, subtitle) in zip(axes, configs):

        df_sorted = plot_df.sort_values(logw_col, ascending=True)
        colors = df_sorted[rel_col].map(color_map)

        bars = ax.barh(
            df_sorted["moment"],
            df_sorted[logw_col],
            color=colors,
            edgecolor="none",
            alpha=0.95
        )

        for i, row in enumerate(df_sorted.itertuples()):

            logw = getattr(row, logw_col)
            p = getattr(row, p_col)
            joint = row.joint_rel

            ax.text(
                logw + 0.02 * (xmax - xmin),
                i,
                f"{p:.3f}",
                va="center",
                fontsize=9,
                color="#333333"
            )

            if joint:
                ax.scatter(
                    logw,
                    i,
                    marker="*",
                    color="#C44E52",
                    s=55,
                    zorder=3
                )

        ax.set_title(subtitle, loc="left", pad=8)

        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.grid(axis="y", visible=False)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel("log(W statistic)")
    axes[0].set_ylabel("Moment")
    axes[1].set_ylabel("Moment")

    legend_elements = [
        Patch(facecolor=color_map[True], label="Relevant"),
        Patch(facecolor=color_map[False], label="Not relevant"),
        Line2D([0], [0], marker='*', color='w',
               label='Joint relevance',
               markerfacecolor='#C44E52',
               markersize=10)
    ]

    axes[0].legend(
        handles=legend_elements,
        loc="lower right",
        frameon=False
    )

    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    plt.tight_layout()
    plt.show()


def implement_PCRTA(input: pd.DataFrame, CRTA_results: pd.DataFrame, name: str, significance_level: float = 0.05):
    '''
    Функция для формирования отчета по PCRTA
    input: pd.DataFrame
        Входной датасет, в котором обязательно есть колонки [C[t], R[t+1], C[t+1], C_ratio] и поле date является индексом
    CRTA_results: pd.DataFrame
        Результат отработки implemet_CRTA(input) для получения условной релевантности для обоих параметров
    name: str
        Название
    significance_level: float
        Уровень значимости
    
    Идея:
        В отличии от implement_CRTA мы будем рассматривать для каждого (beta, gamma):
            Вот мы имеем набор инструментов. Для начала мы можем выделить все "базисные инструменты"
            "Базисные" инструменты (🟢) - это те, для которых тест на conditional_relevance дал результат, что они условно релевантны к всему исходному набору
            Для "оставшихся" интструментов хочется найти те, через которые они линейно выражаются в терминах якобианов других моментов
            Возможны такие ситуации (z - текущий инструмент из "остальных", то есть "небазовый"):
             1) z условно релевантен для базисных и условно НЕрелевантен для остальных (🟡)
             2) z условно НЕрелевантен для базисных и условно релевантен для остальных (🟠)
             3) z условно НЕрелевантен для базисных и условно НЕрелевантен для остальных (🔴)
        После, собираем все в одну табличку
    '''
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
    
    conditionally_relevant_moments_beta = []
    conditionally_irrelevant_moments_beta = []
    conditionally_relevant_moments_gamma = []
    conditionally_irrelevant_moments_gamma = []

    for i in tqdm(range(len(moments)), desc="[stage 1]: determine basics"):
        W_beta, pval_beta, _, _ = partial_conditional_relevance(
            data=data,
            moments=moments,
            f2_indexes=[i],
            a_indexes=[0],
            theta_init=[0, 0],
        )

        W_gamma, pval_gamma, _, _ = partial_conditional_relevance(
            data=data,
            moments=moments,
            f2_indexes=[i],
            a_indexes=[1],
            theta_init=[0, 0],
        )

        if pval_beta <= significance_level:
            conditionally_relevant_moments_beta.append((MOMENT_NAMES[i], W_beta, pval_beta, "🟢"))
        else:
            conditionally_irrelevant_moments_beta.append((MOMENT_NAMES[i], W_beta, pval_beta))

        if pval_gamma <= significance_level:
            conditionally_relevant_moments_gamma.append((MOMENT_NAMES[i], W_gamma, pval_gamma, "🟢"))
        else:
            conditionally_irrelevant_moments_gamma.append((MOMENT_NAMES[i], W_gamma, pval_gamma))

    irrelevant_due_to_basics_beta = []
    irrelevant_due_to_others_beta = []
    irrelevant_due_to_both_beta = []
    irrelevant_due_to_basics_gamma = []
    irrelevant_due_to_others_gamma = []
    irrelevant_due_to_both_gamma = []
    
    

    for z in tqdm([z[0] for z in conditionally_irrelevant_moments_beta], desc="[stage 2]: determine rest instruments for beta"):
        idx = MOMENT_NAMES.index(z)

        # Прогоняем z через базисные инструменты
        basic_moments = [moments[i] for i in range(len(moments)) if MOMENT_NAMES[i] in [z[0] for z in conditionally_relevant_moments_beta]]
        if len(basic_moments) <= 1:
            W_basic, pval_basic = None, float("inf")
        else:
            basic_moments.append(moments[idx])
            W_basic, pval_basic, _, _ = partial_conditional_relevance(
                data=data,
                moments=basic_moments,
                f2_indexes=[len(basic_moments)-1],
                a_indexes=[0],
                theta_init=[0, 0],
            )

        # Прогоняем z через оставшиеся инструменты
        rest_moments = [moments[i] for i in range(len(moments)) if MOMENT_NAMES[i] in [z[0] for z in conditionally_irrelevant_moments_beta] and z != MOMENT_NAMES[i]]
        rest_moments.append(moments[idx])
        W_rest, pval_rest, _, _ = partial_conditional_relevance(
            data=data,
            moments=rest_moments,
            f2_indexes=[len(rest_moments)-1],
            a_indexes=[0],
            theta_init=[0, 0],
        )
        
        if pval_basic > significance_level and pval_rest < significance_level:
            if W_basic is not None:
                irrelevant_due_to_basics_beta.append((z, W_basic, pval_basic, "🟠"))
                                             
        if pval_basic < significance_level and pval_rest > significance_level:
            irrelevant_due_to_others_beta.append((z, W_rest, pval_rest, "🟡"))
                                             
        if pval_basic > significance_level and pval_rest > significance_level:
            irrelevant_due_to_both_beta.append((z, W_rest, pval_rest, "🔴"))

    
    for z in tqdm([z[0] for z in conditionally_irrelevant_moments_gamma], desc="[stage 3]: determine rest instruments for gamma"):
        idx = MOMENT_NAMES.index(z)

        # Прогоняем z через базисные инструменты
        basic_moments = [moments[i] for i in range(len(moments)) if MOMENT_NAMES[i] in [z[0] for z in conditionally_relevant_moments_gamma]]
        if len(basic_moments) <= 1:
            W_basic, pval_basic = None, float("inf")
        else:
            basic_moments.append(moments[idx])
            W_basic, pval_basic, _, _ = partial_conditional_relevance(
                data=data,
                moments=basic_moments,
                f2_indexes=[len(basic_moments)-1],
                a_indexes=[1],
                theta_init=[0, 0],
            )

        # Прогоняем z через оставшиеся инструменты
        rest_moments = [moments[i] for i in range(len(moments)) if MOMENT_NAMES[i] in [z[0] for z in conditionally_irrelevant_moments_gamma] and z != MOMENT_NAMES[i]]
        rest_moments.append(moments[idx])
        W_rest, pval_rest, _, _ = partial_conditional_relevance(
            data=data,
            moments=rest_moments,
            f2_indexes=[len(rest_moments)-1],
            a_indexes=[1],
            theta_init=[0, 0],
        )
        
        if pval_basic > significance_level and pval_rest < significance_level:
            if W_basic is not None:
                irrelevant_due_to_basics_gamma.append((z, W_basic, pval_basic, "🟠"))
                                             
        if pval_basic < significance_level and pval_rest > significance_level:
            irrelevant_due_to_others_gamma.append((z, W_rest, pval_rest, "🟡"))
                                             
        if pval_basic > significance_level and pval_rest > significance_level:
            irrelevant_due_to_both_gamma.append((z, W_rest, pval_rest, "🔴"))
                                             

    report_beta = conditionally_relevant_moments_beta
    report_beta.extend(irrelevant_due_to_others_beta)
    report_beta.extend(irrelevant_due_to_basics_beta)
    report_beta.extend(irrelevant_due_to_both_beta)
    report_gamma = conditionally_relevant_moments_gamma
    report_gamma.extend(irrelevant_due_to_others_gamma)
    report_gamma.extend(irrelevant_due_to_basics_gamma)
    report_gamma.extend(irrelevant_due_to_both_gamma)

    report_beta = pd.DataFrame(report_beta, columns=["moment", "W_beta", "p_value_for_beta", "conditional_relevance_for_beta"])
    report_beta['logW_beta'] = np.log(report_beta['W_beta'])
    report_beta['group'] = name
    report_beta = report_beta[['moment', 'logW_beta', 'p_value_for_beta', 'group', 'conditional_relevance_for_beta']]
    report_gamma = pd.DataFrame(report_gamma, columns=["moment", "W_gamma", "p_value_for_gamma", "conditional_relevance_for_gamma"])
    report_gamma['logW_gamma'] = np.log(report_gamma['W_gamma'])
    report_gamma = report_gamma[['moment', 'logW_gamma', 'p_value_for_gamma', 'conditional_relevance_for_gamma']]

    report = pd.merge(
        left=report_beta,
        right=report_gamma,
        how='inner',
        on='moment'
    )

    report = report[[
        'moment',
        'logW_beta',
        'logW_gamma',
        'p_value_for_beta',
        'p_value_for_gamma',
        'conditional_relevance_for_beta',
        'conditional_relevance_for_gamma'
    ]]

    report = report.merge(
        right=CRTA_results[['moment', 'conditional_relevance']],
        on='moment',
        how='left',
    )

    # assert len(CRTA_results) == len(report), "Какой-то инструмент выпал из набора"

    return report


def build_optimal_euler_portfolio(
    tables: Dict[str, pd.DataFrame],
    alpha: float = 0.05,
    drop_const: bool = True,
    keep_shared_only: bool = False,
    min_block_support: int = 1,
) -> pd.DataFrame:
    """
    Формирует оптимальный портфель факторов (моментов) для оценки модели Эйлера
    на основе результатов тестов безусловной релевантности (URTA),
    условной релевантности (CRTA), частичной безусловной релевантности (PURTA)
    и частичной условной релевантности (PCRTA), представленных в виде
    таблиц вида *_of_feature_group_i.

    Определение оптимального портфеля
    ---------------------------------
    Под оптимальным портфелем факторов понимается такое подмножество моментов,
    которое одновременно удовлетворяет следующим формальным критериям:

    1. Безусловная информативность (URTA):
       Момент считается потенциально информативным, если в таблицах вида
       URTA_of_feature_group_i он имеет статистически значимое значение
       (p_value < alpha), что свидетельствует о наличии вклада в идентификацию
       модели в целом.

    2. Условная независимость (CRTA):
       На основании таблиц CRTA_of_feature_group_i в портфель включаются только
       те моменты, которые классифицированы как условно релевантные (🟢) либо
       не полностью редуцируемые через другие моменты (🟡, 🟠), то есть не
       принадлежат к множеству полностью условно нерелевантных инструментов (🔴).
       Данный критерий обеспечивает отсутствие линейной зависимости якобианов
       моментов и, следовательно, независимость каналов идентификации.

    3. Параметрическая релевантность (PURTA):
       Согласно таблицам PURTA_of_feature_group_i, момент включается в портфель,
       если он статистически значим (p_value_beta < alpha или p_value_gamma < alpha)
       хотя бы для одного из структурных параметров модели:
           • beta — параметр межвременного дисконтирования,
           • gamma — параметр риск-аверсии.
       Это гарантирует, что каждый выбранный фактор несёт идентификационную
       нагрузку на уровне структурных параметров.

    4. Частичная условная релевантность (PCRTA):
       На основе таблиц PCRTA_of_feature_group_i дополнительно проверяется,
       сохраняет ли момент свою параметрическую значимость после условного
       исключения влияния других моментов.
       В портфель включаются только те моменты, для которых выполняется:
           • p_value_for_beta < alpha и/или p_value_for_gamma < alpha,
       что интерпретируется как наличие независимого канала идентификации
       соответствующего параметра.

    Итоговый результат
    -----------------
    Функция возвращает единый DataFrame, содержащий оптимальный портфель факторов,
    где каждый момент:
        • прошёл отбор по URTA (общая информативность),
        • не является условно избыточным согласно CRTA,
        • значим для beta и/или gamma по PURTA,
        • сохраняет значимость после условного контроля по PCRTA,

    Тем самым итоговый портфель представляет собой строго отобранное множество
    инструментов, обеспечивающих идентификацию параметров модели Эйлера через
    независимые и экономически интерпретируемые каналы.
    """

    def parse_key(key: str) -> Tuple[str, str]:
        m = re.match(r"^(URTA|CRTA|PURTA|PCRTA)_of_(.+)$", key, flags=re.IGNORECASE)
        if not m:
            raise ValueError(
                f"Не удалось распознать ключ '{key}'. "
                f"Ожидается формат вроде 'URTA_of_feature_group_1'."
            )
        test = m.group(1).upper()
        group = m.group(2)
        return test, group

    def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def normalize_table(key: str, df: pd.DataFrame) -> pd.DataFrame:
        test, group = parse_key(key)

        if "moment" not in df.columns:
            raise ValueError(f"В таблице {key} нет колонки 'moment'.")

        out = df.copy()
        out["test"] = test
        out["group"] = group

        if test in {"URTA", "CRTA"}:
            p_col = pick_first_existing(out, ["p_value"])
            if p_col is None:
                raise ValueError(f"В таблице {key} нет колонки p_value.")
            out["p_beta"] = np.nan
            out["p_gamma"] = np.nan
            out["p_overall"] = pd.to_numeric(out[p_col], errors="coerce")

        elif test == "PURTA":
            p_beta = pick_first_existing(out, ["p_value_beta"])
            p_gamma = pick_first_existing(out, ["p_value_gamma"])
            p_overall = pick_first_existing(out, ["p_value"])
            if p_beta is None or p_gamma is None or p_overall is None:
                raise ValueError(
                    f"В таблице {key} ожидаются колонки "
                    f"p_value_beta, p_value_gamma и p_value."
                )
            out["p_beta"] = pd.to_numeric(out[p_beta], errors="coerce")
            out["p_gamma"] = pd.to_numeric(out[p_gamma], errors="coerce")
            out["p_overall"] = pd.to_numeric(out[p_overall], errors="coerce")

        elif test == "PCRTA":
            p_beta = pick_first_existing(out, ["p_value_for_beta", "p_value_beta"])
            p_gamma = pick_first_existing(out, ["p_value_for_gamma", "p_value_gamma"])
            p_overall = pick_first_existing(out, ["p_value"])
            if p_beta is None or p_gamma is None:
                raise ValueError(
                    f"В таблице {key} ожидаются колонки "
                    f"p_value_for_beta/p_value_beta и p_value_for_gamma/p_value_gamma."
                )
            out["p_beta"] = pd.to_numeric(out[p_beta], errors="coerce")
            out["p_gamma"] = pd.to_numeric(out[p_gamma], errors="coerce")
            out["p_overall"] = pd.to_numeric(out[p_overall], errors="coerce") if p_overall else np.nan

        else:
            raise ValueError(f"Неизвестный тест: {test}")

        return out[["moment", "test", "group", "p_beta", "p_gamma", "p_overall"]]

    def is_sig(x: float) -> bool:
        return pd.notna(x) and x < alpha

    # ---------- 1) Сводим все таблицы в один длинный формат ----------
    frames = []
    for key, df in tables.items():
        frames.append(normalize_table(key, df))

    all_rows = pd.concat(frames, ignore_index=True)

    if drop_const:
        all_rows = all_rows[all_rows["moment"].astype(str).str.lower() != "const"].copy()

    # ---------- 2) Агрегируем доказательства по каждому моменту ----------
    records = []

    for moment, g in all_rows.groupby("moment", sort=False):
        # Сигналы по параметрам
        beta_rows = g[pd.to_numeric(g["p_beta"], errors="coerce").notna()].copy()
        gamma_rows = g[pd.to_numeric(g["p_gamma"], errors="coerce").notna()].copy()

        beta_sig_rows = beta_rows[beta_rows["p_beta"].apply(is_sig)] if len(beta_rows) else beta_rows.iloc[0:0]
        gamma_sig_rows = gamma_rows[gamma_rows["p_gamma"].apply(is_sig)] if len(gamma_rows) else gamma_rows.iloc[0:0]

        beta_hit = len(beta_sig_rows) > 0
        gamma_hit = len(gamma_sig_rows) > 0

        # Общие p-values (URTA/CRTA и overall из partial тестов)
        overall_rows = g[pd.to_numeric(g["p_overall"], errors="coerce").notna()].copy()
        overall_sig_rows = overall_rows[overall_rows["p_overall"].apply(is_sig)] if len(overall_rows) else overall_rows.iloc[0:0]
        overall_hit = len(overall_sig_rows) > 0

        # Поддержка по блокам
        beta_blocks = sorted(set(beta_sig_rows["group"].astype(str).tolist()))
        gamma_blocks = sorted(set(gamma_sig_rows["group"].astype(str).tolist()))
        overall_blocks = sorted(set(overall_sig_rows["group"].astype(str).tolist()))

        beta_support = len(beta_blocks)
        gamma_support = len(gamma_blocks)
        overall_support = len(overall_blocks)

        # Лучшие p-values
        p_beta_best = float(beta_sig_rows["p_beta"].min()) if beta_hit else np.nan
        p_gamma_best = float(gamma_sig_rows["p_gamma"].min()) if gamma_hit else np.nan
        p_overall_best = float(overall_sig_rows["p_overall"].min()) if overall_hit else np.nan

        # Категория момента
        if beta_hit and gamma_hit:
            role = "bridge"
        elif beta_hit:
            role = "beta_core"
        elif gamma_hit:
            role = "gamma_core"
        elif keep_shared_only and overall_hit:
            role = "shared"
        else:
            role = "excluded"

        # Строгий фильтр по числу блоков
        selected = (
            (role != "excluded")
            and ((beta_support >= min_block_support) or (gamma_support >= min_block_support) or (overall_support >= min_block_support))
        )

        # Итоговый score для сортировки
        score = 0.0
        if beta_hit:
            score += 5.0
        if gamma_hit:
            score += 5.0
        if beta_hit and gamma_hit:
            score += 2.0
        score += 1.0 * overall_hit
        score += 0.25 * (beta_support + gamma_support + overall_support)

        records.append(
            {
                "moment": moment,
                "selected": selected,
                "role": role,
                "score": score,
                "p_beta_best": p_beta_best,
                "p_gamma_best": p_gamma_best,
                "p_overall_best": p_overall_best,
                "beta_hit": beta_hit,
                "gamma_hit": gamma_hit,
                "overall_hit": overall_hit,
                "beta_support_blocks": beta_support,
                "gamma_support_blocks": gamma_support,
                "overall_support_blocks": overall_support,
                "beta_blocks": ", ".join(beta_blocks),
                "gamma_blocks": ", ".join(gamma_blocks),
                "overall_blocks": ", ".join(overall_blocks),
                "tests_seen": ", ".join(sorted(set(g["test"].astype(str).tolist()))),
                "groups_seen": ", ".join(sorted(set(g["group"].astype(str).tolist()))),
            }
        )

    master = pd.DataFrame(records)

    # ---------- 3) Финальный портфель ----------
    portfolio = master[master["selected"]].copy()

    role_order = {"bridge": 0, "beta_core": 1, "gamma_core": 2, "shared": 3, "excluded": 4}
    portfolio["role_order"] = portfolio["role"].map(role_order)

    portfolio = portfolio.sort_values(
        by=["role_order", "score", "beta_support_blocks", "gamma_support_blocks", "overall_support_blocks", "moment"],
        ascending=[True, False, False, False, False, True],
    ).drop(columns=["role_order"])

    # Можно сохранить полную диагностику в attrs
    portfolio.attrs["master_diagnostics"] = master.sort_values(
        by=["selected", "score", "moment"], ascending=[False, False, True]
    ).reset_index(drop=True)

    return portfolio.reset_index(drop=True)


def plot_optimal_euler_portfolio_graph(
    input: pd.DataFrame,
    title: str = "Оптимальный портфель факторов для модели Эйлера",
    figsize: Tuple[int, int] = (18, 12),
    wrap_width: int = 22,
    savepath: Optional[str] = None,
    dpi: int = 400,
    eng2rus: dict = None
):
    portfolio = input.copy()
    if eng2rus:
        portfolio['moment'] = portfolio['moment'].map(eng2rus)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    })

    df = portfolio.copy()

    required = {"moment", "role", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В portfolio не хватает колонок: {missing}")

    df = df[df["role"].isin(["bridge", "beta_core", "gamma_core", "shared"])].copy()

    role_order = {"bridge": 0, "beta_core": 1, "gamma_core": 2, "shared": 3}
    df["role_order"] = df["role"].map(role_order).fillna(99)
    df = df.sort_values(
        by=["role_order", "score", "moment"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    groups = {
        "bridge": df[df["role"] == "bridge"].copy(),
        "beta_core": df[df["role"] == "beta_core"].copy(),
        "gamma_core": df[df["role"] == "gamma_core"].copy(),
        "shared": df[df["role"] == "shared"].copy(),
    }

    x_beta = 0.0
    x_beta_core = 2.2
    x_bridge = 5.0
    x_gamma_core = 7.8
    x_gamma = 10.0

    counts = [len(v) for v in groups.values() if len(v) > 0]
    n = max(counts) if counts else 1
    y_top = max(3, n * 1.4)

    def spaced_y(k: int) -> np.ndarray:
        if k == 0:
            return np.array([])
        if k == 1:
            return np.array([0.0])
        return np.linspace(y_top, 0, k)

    positions = []
    for role, g in groups.items():
        ys = spaced_y(len(g))

        if role == "bridge":
            x = x_bridge
        elif role == "beta_core":
            x = x_beta_core
        elif role == "gamma_core":
            x = x_gamma_core
        else:
            x = (x_beta_core + x_gamma_core) / 2.0

        for idx, (_, row) in enumerate(g.iterrows()):
            positions.append({
                "moment": row["moment"],
                "role": role,
                "score": float(row["score"]),
                "x": x,
                "y": float(ys[idx]),
                "beta_hit": bool(row.get("beta_hit", False)),
                "gamma_hit": bool(row.get("gamma_hit", False)),
                "beta_support_blocks": int(row.get("beta_support_blocks", 0)) if pd.notna(row.get("beta_support_blocks", np.nan)) else 0,
                "gamma_support_blocks": int(row.get("gamma_support_blocks", 0)) if pd.notna(row.get("gamma_support_blocks", np.nan)) else 0,
                "overall_support_blocks": int(row.get("overall_support_blocks", 0)) if pd.notna(row.get("overall_support_blocks", np.nan)) else 0,
            })

    pos_df = pd.DataFrame(positions)

    fig, ax = plt.subplots(figsize=figsize)

    model_node_size = 2200
    ax.scatter([x_beta], [y_top / 2], s=model_node_size, marker="o",
               color="#2f2f2f", edgecolor="black", linewidth=1.0, zorder=2)
    ax.scatter([x_gamma], [y_top / 2], s=model_node_size, marker="o",
               color="#2f2f2f", edgecolor="black", linewidth=1.0, zorder=2)

    ax.text(
        x_beta, y_top / 2, r"$\beta$",
        ha="center", va="center",
        fontsize=20, fontweight="bold",
        color="white", zorder=3
    )
    ax.text(
        x_gamma, y_top / 2, r"$\gamma$",
        ha="center", va="center",
        fontsize=20, fontweight="bold",
        color="white", zorder=3
    )

    guide_color = "#8a8a8a"
    for x in [x_beta, x_beta_core, x_bridge, x_gamma_core, x_gamma]:
        ax.axvline(x, color=guide_color, linestyle="--", linewidth=0.8, alpha=0.18, zorder=0)

    header_fs = 12
    ax.text(x_beta_core, y_top + 0.8, "Канал дисконтирования",
            ha="center", va="bottom", fontsize=header_fs, fontweight="bold")
    ax.text(x_bridge, y_top + 0.8, "Связующий канал",
            ha="center", va="bottom", fontsize=header_fs, fontweight="bold")
    ax.text(x_gamma_core, y_top + 0.8, "Канал риска",
            ha="center", va="bottom", fontsize=header_fs, fontweight="bold")

    color_map = {
        "bridge": "#1f77b4",
        "beta_core": "#ff7f0e",
        "gamma_core": "#2ca02c",
    }

    for _, row in pos_df.iterrows():
        role = row["role"]
        x, y = row["x"], row["y"]
        color = color_map.get(role, "#333333")

        support_sum = row["beta_support_blocks"] + row["gamma_support_blocks"] + row["overall_support_blocks"]
        size = 240 + 38 * row["score"] + 28 * support_sum

        ax.scatter(
            [x], [y],
            s=size,
            color=color,
            alpha=0.93,
            zorder=3,
            edgecolor="black",
            linewidth=0.9
        )

        if row["beta_hit"]:
            ax.annotate(
                "",
                xy=(x_beta, y_top / 2),
                xytext=(x - 0.15, y),
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.38, lw=1.4),
                zorder=1,
            )
        if row["gamma_hit"]:
            ax.annotate(
                "",
                xy=(x_gamma, y_top / 2),
                xytext=(x + 0.15, y),
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.38, lw=1.4),
                zorder=1,
            )

        wrapped = "\n".join(textwrap.wrap(str(row["moment"]), width=wrap_width))

        dx = 0.24 if role in {"beta_core", "bridge"} else -0.24
        ha = "left" if dx > 0 else "right"

        ax.text(
            x + dx, y, wrapped,
            ha=ha, va="center",
            fontsize=10.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=color,
                lw=0.8,
                alpha=0.96
            ),
            zorder=4,
        )

    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(-1.2, y_top + 1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, pad=18)

    legend_items = [
        ("bridge", r"bridge: $\beta$ and $\gamma$"),
        ("beta_core", r"$\beta$-core"),
        ("gamma_core", r"$\gamma$-core"),
    ]

    handles = []
    labels = []
    for key, lab in legend_items:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=color_map[key],
                markeredgecolor="black",
                markersize=10,
                linewidth=0
            )
        )
        labels.append(lab)

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")

    plt.show()
