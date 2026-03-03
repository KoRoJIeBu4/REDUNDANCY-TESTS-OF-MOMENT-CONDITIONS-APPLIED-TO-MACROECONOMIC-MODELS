import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


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