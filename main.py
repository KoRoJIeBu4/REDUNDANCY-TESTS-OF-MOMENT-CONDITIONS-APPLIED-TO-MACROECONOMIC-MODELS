#!/usr/bin/env python
# coding: utf-8

# ## <center> Packages <center>

# In[1]:


from tests_impl import *
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import monte_carlo_tools.non_partial_monte_carlo
import monte_carlo_tools.partial_monte_carlo
import torch
import numpy as np
from sklearn.decomposition import PCA
import utilities
import statsmodels.api as sm

# ## <center> 1.1 Relevance Test [Done] <center>
# 
# Рассмотрим разбиение моментных условий:
# $$
# f(\theta) = 
# \begin{pmatrix}
# f_1(\theta) \\
# f_2(\theta)
# \end{pmatrix},
# $$
# где $ f_2(\theta) $ — тестируемый набор условий.
# 
# **Определение:**  
# Моментные условия $ f_2 $ являются **релевантными**, если
# $$
# G_2 \neq 0, \quad \text{где } G_2 = \left. \frac{\partial \mathbb{E}[f_2(\theta)]}{\partial \theta'} \right|_{\theta = \theta_0}.
# $$
# 
# **Нулевая гипотеза:**
# $$
# H_0: G_2 = 0 \quad \text{(нерелевантность)}.
# $$
# 
# **Статистика теста:**
# $$
# W = T \, g_{2T}(\hat{\theta})' \, \hat{\Sigma}^{-1} \, g_{2T}(\hat{\theta}),
# $$
# где $ g_{2T}(\hat{\theta}) = \operatorname{vec}(G_{2T}'(\hat{\theta})) $.
# 
# **Асимптотика:**
# $$
# W \xrightarrow{d} \chi^2(m_2 k).
# $$
# 
# **Интерпретация:**  
# Тест проверяет, содержит ли набор моментных условий $f_2$ информацию, способную (потенциально) идентифицировать параметры $\theta$ самостоятельно.

# In[ ]:


# see test_impl: unconditional_relevance

# ## <center> 1.2 Simulation and sanity check [Done] <center>

# In[ ]:


sets_of_f2_indices = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [0, 1, 4],
    [2, 3],
]

unconditional_results = non_partial_monte_carlo.monte_carlo_unconditional_relevance_prodecure(
    T=200,
    beta=0.98, 
    gamma=1.5,
    sets_of_f2_indices=sets_of_f2_indices,
    B=1000,
    num_alphas=5000,
    n_jobs=-1,
)

# In[51]:


non_partial_monte_carlo.plot_rejection_curves(
    unconditional_results, 
    save_path="unconditional_results.png", 
    fill_vertical_zero_line=['Z1', 'Z2', 'Z5', 'Z1 Z2 Z5']
)

# _1 $) информацию для идентификации параметров $ \theta $.

# In[23]:


# see test_impl: conditional_relevance

# ## <center> 2.1 Conditional Relevance [Done] <center>
# 
# Рассмотрим разбиение моментных условий:
# $$
# f(\theta) = 
# \begin{pmatrix}
# f_1(\theta) \\
# f_2(\theta)
# \end{pmatrix}.
# $$
# 
# Пусть ковариационная матрица моментных условий имеет вид:
# $$
# \Omega = \mathrm{Var}\big(\sqrt{T} f_T(\theta_0)\big) =
# \begin{pmatrix}
# \Omega_{11} & \Omega_{12} \\
# \Omega_{21} & \Omega_{22}
# \end{pmatrix}.
# $$
# 
# Определим ортогонализованные моменты:
# $$
# f_{\Delta}(\theta) = f_2(\theta) - \Omega_{21}\Omega_{11}^{-1} f_1(\theta),
# $$
# и соответствующую матрицу производных:
# $$
# G_{\Delta} = \left. \frac{\partial \mathbb{E}[f_{\Delta}(\theta)]}{\partial \theta'} \right|_{\theta = \theta_0}.
# $$
# 
# **Определение:**  
# Моментные условия $ f_2 $ являются **условно релевантными** (относительно $ f_1 $), если
# $$
# G_{\Delta} \neq 0.
# $$
# 
# **Нулевая гипотеза:**
# $$
# H_0: G_{\Delta} = 0 \quad \text{(условная нерелевантность / редундантность)}.
# $$
# 
# **Статистика теста:**
# $$
# W = T \, \hat{g}_{\Delta T}(\hat{\theta})' \, \hat{\Sigma}^{-1} \, \hat{g}_{\Delta T}(\hat{\theta}),
# $$
# где
# $$
# \hat{g}_{\Delta T}(\hat{\theta}) = \operatorname{vec}\left(
# G_{2T}(\hat{\theta}) - \Omega_{21,T}(\hat{\theta}) \Omega_{11,T}^{-1}(\hat{\theta}) G_{1T}(\hat{\theta})
# \right)'.
# $$
# 
# **Асимптотика:**
# $$
# W \xrightarrow{d} \chi^2(m_2 k).
# $$
# 
# **Интерпретация:**  
# Тест проверяет, добавляют ли моментные условия $ f_2 $ новую (ортогональную к $ f_1 $) информацию для идентификации параметров $ \theta $.

# ## <center> 2.2 Simulation and sanity check [Done] <center>

# In[2]:


sets_of_f2_indices = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [0, 1, 4],
    [2, 3],
    [5]
]

conditional_results = non_partial_monte_carlo.monte_carlo_conditional_relevance_prodecure(
    T=200,
    beta=0.98, 
    gamma=1.5,
    sets_of_f2_indices=sets_of_f2_indices,
    B=1000,
    num_alphas=5000,
    n_jobs=-1,
)

# In[3]:


non_partial_monte_carlo.plot_rejection_curves(
    {k : v for k, v in conditional_results.items() if k in (
        "[CONDITIONAL RELEVANCE]: Z2",
        "[CONDITIONAL RELEVANCE]: Z3",
        "[CONDITIONAL RELEVANCE]: Z5",
        "[CONDITIONAL RELEVANCE]: Z1 Z2 Z5",
        "[CONDITIONAL RELEVANCE]: Z6"
    )},
    save_path="conditional_results.png", 
    fill_vertical_zero_line=['Z1', 'Z2', 'Z5', 'Z1 Z2 Z5']
)

# ## <center> 3.1 Partial Relevance Test [Done] <center>
# 
# Пусть параметры разбиты на два подмножества:
# $$
# \theta =
# \begin{pmatrix}
# \theta_A \\
# \theta_B
# \end{pmatrix},
# \quad \text{где } \theta_A \in \mathbb{R}^{k_A}.
# $$
# 
# Рассмотрим разбиение моментных условий:
# $$
# f(\theta) = 
# \begin{pmatrix}
# f_1(\theta) \\
# f_2(\theta)
# \end{pmatrix}.
# $$
# 
# Обозначим матрицу производных по подмножеству параметров:
# $$
# G_{2A} = \left. \frac{\partial \mathbb{E}[f_2(\theta)]}{\partial \theta_A'} \right|_{\theta = \theta_0}.
# $$
# 
# **Определение:**  
# Моментные условия $ f_2 $ являются **частично релевантными** для параметров $ \theta_A $, если
# $$
# G_{2A} \neq 0.
# $$
# 
# **Нулевая гипотеза:**
# $$
# H_0: G_{2A} = 0 \quad \text{(частичная нерелевантность)}.
# $$
# 
# **Статистика теста:**
# $$
# W = T \, g_{2A,T}(\hat{\theta})' \, \hat{\Sigma}^{-1} \, g_{2A,T}(\hat{\theta}),
# $$
# где
# $$
# g_{2A,T}(\hat{\theta}) = \operatorname{vec}\big(G_{2A,T}'(\hat{\theta})\big).
# $$
# 
# **Асимптотика:**
# $$
# W \xrightarrow{d} \chi^2(m_2 k_A).
# $$
# 
# **Интерпретация:**  
# Тест проверяет, содержат ли моментные условия $ f_2 $ информацию для идентификации конкретного подмножества параметров $ \theta_A $.

# In[4]:


# see test_impl: partial_unconditional_relevance

# ## <center> 3.2 Simulation and sanity check [Done] <center>

# In[2]:


sets_of_f2_indices = [
    [4, 5],
    [2],
    [2],
    [3],
    [3],
    [1]
]

sets_of_a_indices = [
    [0, 1],
    [1],
    [0],
    [1],
    [0],
    [0, 1]
]

results = partial_monte_carlo.monte_carlo_partial_unconditional_relevance_prodecure(
    sets_of_f2_indices=sets_of_f2_indices,
    sets_of_a_indices=sets_of_a_indices,
    B=2000,
    num_alphas=5000
)

# In[3]:


partial_monte_carlo.plot_rejection_curves(
    results=results, 
    fill_vertical_zero_line=[
        "Z3 FOR beta", "Z1 FOR beta gamma"
    ],
    save_path="partial_unconditional_results.png"
)

# ## <center> 4.1 Partial Conditional Relevance Test [Done] <center>
# 
# Пусть параметры разбиты на два подмножества:
# $$
# \theta =
# \begin{pmatrix}
# \theta_A \\
# \theta_B
# \end{pmatrix},
# \quad \text{где } \theta_A \in \mathbb{R}^{k_A}.
# $$
# 
# Рассмотрим разбиение моментных условий:
# $$
# f(\theta) = 
# \begin{pmatrix}
# f_1(\theta) \\
# f_2(\theta)
# \end{pmatrix}.
# $$
# 
# Пусть ковариационная матрица моментных условий:
# $$
# \Omega =
# \begin{pmatrix}
# \Omega_{11} & \Omega_{12} \\
# \Omega_{21} & \Omega_{22}
# \end{pmatrix}.
# $$
# 
# Определим ортогонализованные моменты:
# $$
# f_{\Delta}(\theta) = f_2(\theta) - \Omega_{21}\Omega_{11}^{-1} f_1(\theta).
# $$
# 
# Обозначим матрицу производных по подмножеству параметров:
# $$
# G_{\Delta A} = \left. \frac{\partial \mathbb{E}[f_{\Delta}(\theta)]}{\partial \theta_A'} \right|_{\theta = \theta_0}.
# $$
# 
# **Определение:**  
# Моментные условия $ f_2 $ являются **частично условно релевантными** для параметров $ \theta_A $ (относительно $ f_1 $), если
# $$
# G_{\Delta A} \neq 0.
# $$
# 
# **Нулевая гипотеза:**
# $$
# H_0: G_{\Delta A} = 0 \quad \text{(частичная условная нерелевантность)}.
# $$
# 
# **Статистика теста:**
# $$
# W = T \, g_{\Delta A,T}(\hat{\theta})' \, \hat{\Sigma}^{-1} \, g_{\Delta A,T}(\hat{\theta}),
# $$
# где
# $$
# g_{\Delta A,T}(\hat{\theta}) = \operatorname{vec}\big(G_{\Delta A,T}'(\hat{\theta})\big).
# $$
# 
# **Асимптотика:**
# $$
# W \xrightarrow{d} \chi^2(m_2 k_A).
# $$
# 
# **Интерпретация:**  
# Тест проверяет, добавляют ли моментные условия $ f_2 $ новую информацию для идентификации подмножества параметров $ \theta_A $, сверх уже содержащейся в $ f_1 $.

# In[20]:


# see test_impl: partial_conditional_relevance

# ## <center> 4.2 Simulation and sanity check [Done] <center>

# In[2]:


sets_of_f2_indices = [
    [3, 4],
    [2],
    [2],
    [1],
    [1],
    [1],
]

sets_of_a_indices = [
    [0, 1],
    [1],
    [0],
    [0, 1],
    [0],
    [1],
]

results = partial_monte_carlo.monte_carlo_partial_conditional_relevance_prodecure(
    sets_of_f2_indices=sets_of_f2_indices,
    sets_of_a_indices=sets_of_a_indices,
    B=2000,
    num_alphas=5000
)

# In[3]:


partial_monte_carlo.plot_rejection_curves(
    results=results, 
    fill_vertical_zero_line=[
        "Z3 FOR beta", "Z1 FOR beta gamma"
    ],
    save_path="partial_conditional_results_without_Z2.png"
)

# ## <center> 5 Data Collection [Done] <center> 

# ### <center> 5.1 Consumption <center>
# 
# #### Здесь и далее, индексация означает, что данные получены на конец месяца
# #### t - означает прошлый период, тогда как t+1 новый период
# 
# #### Link: https://www.sberindex.ru/ru/dashboards/consumer-spending
# #### Link: https://rosstat.gov.ru/statistics/price

# In[233]:


# Данные представлены за месяц

consumption = pd.read_excel("data/datasets/consumption.xlsx", index_col=0)

consumption.head()

# In[234]:


# Берем за базовую точку отчета: Декабрь 2018 - и приводим все к ценам базового периода
# Также под агрегированным потреблением будем понимать сумму продовольсвтенного, потребления услуг и общественное питание

consumption_indices = pd.read_excel('data/datasets/cpi.xlsx', sheet_name='Final')
consumption['P_foods'] = consumption_indices['P food (base 2018)'].values
consumption['P_nonfoods'] = consumption_indices['P nonfood (base 2018)'].values
consumption['P_services'] = consumption_indices['P services (base 2018)'].values

consumption['real_foods'] = consumption['foods'] / consumption['P_foods'] * 100
consumption['real_services'] = consumption['services'] / consumption['P_services'] * 100
consumption['real_caterings'] = consumption['caterings'] / consumption['P_foods'] * 100

# consumption['C'] = consumption['real_foods'] + consumption['real_services'] + consumption['real_caterings']

# consumption = consumption.rename(columns={"C" : "C[t]"})

consumption.head()

# In[235]:


consumption.to_excel('data/datasets/consumption.xlsx')
sns.lineplot(data=consumption, x='date', y='C[t]')

# ### <center> 5.2 Return <center>
# #### Link: https://www.cbr.ru/statistics/bank_sector/int_rat
# #### Link: https://rosstat.gov.ru/statistics/price
# #### Link: https://www.cbr.ru/analytics/dkp/inflationary_expectations/

# In[263]:


# Из данных ЦБ РФ по ожидаемой и наблюдаемой инфляции на годовой горизонт можно выцепить (в предположении одинаковых месячных темпов роста инфляции)
# ежемесячную ожидаемую инфляцию
expected_and_observed_inflation = pd.read_excel("data/datasets/expected_and_observed_inflation.xlsx", sheet_name="Final")
expected_and_observed_inflation.head()

# In[264]:


# Номинальная ставка - ежемесячная доходность ставки по депозитам со сроком привлечения от 181 дня до 1 года

real_return = pd.read_excel('data/datasets/return.xlsx', sheet_name='Final')

# Данные ожидаемой инфляции оказались очень шумными, поэтому их не стоит использовать, чтобы обеспечить устойчивость оценок при сходимости
# Поэтому мы полагаем, что агент очень точно угадывает ожидаемую инфляцию между периодами

# real_return['expected_inflation'] = expected_and_observed_inflation['expected_month_inflation'].shift()
real_return['pi[t+1]'] = real_return['inflation_registered'].shift(-1)

real_return['i_monthly'] = (np.power(1 + real_return['i'], 1/12) - 1).shift(-1)
real_return['R'] = (1 + real_return['i_monthly']) / (1 + real_return['pi[t+1]']) - 1

real_return = real_return.rename(columns={"R" : "R[t+1]"})

real_return

# In[265]:


real_return.to_excel('data/datasets/real_return.xlsx')
sns.lineplot(data=real_return, x='date', y='R[t+1]')

# ### <center> 5.3 Feature Selection [Done] <center>

# #### <center> 5.3.1 Монетарно-финансовый блок [Done] <center>
# ##### Link: https://www.cbr.ru/hd_base/keyrate/
# ##### Link: https://www.cbr.ru/statistics/bank_sector/int_rat
# ##### Link: https://www.cbr.ru/analytics/dkp/inflationary_expectations/

# In[266]:


# Берем лаги прошлых значений доходности
feature_group_1 = pd.DataFrame({
    "date" : real_return['date'],
    "R[t]" : real_return['R[t+1]'].shift(1),
    "R[t-1]" : real_return['R[t+1]'].shift(2)
})

# In[267]:


# Данные по ключевой ставки нужно сагрегировать к месячной
interest_rate = pd.read_excel('data/datasets/interest_rate.xlsx')

interest_rate['date'] = pd.to_datetime(interest_rate['date'])
interest_rate.set_index('date', inplace=True)

def weighted_monthly_rate(group):
    counts = group.value_counts()
    wavg = np.sum(counts.index * counts.values) / counts.values.sum()
    return wavg

mean_rate = interest_rate['interest_rate'].resample('ME').mean()
median_rate = interest_rate['interest_rate'].resample('ME').median()
last_rate = interest_rate['interest_rate'].resample('ME').last()
min_rate = interest_rate['interest_rate'].resample('ME').min()
max_rate = interest_rate['interest_rate'].resample('ME').max()
range_rate = max_rate - min_rate
std_rate = interest_rate['interest_rate'].resample('ME').std()
wavg_rate = interest_rate['interest_rate'].resample('ME').apply(weighted_monthly_rate)

to_add = {
    'mean_rate[t]': mean_rate,
    # 'median_rate[t]': median_rate,
    # 'last_rate[t]': last_rate,
    # 'min_rate[t]': min_rate,
    # 'max_rate[t]': max_rate,
    # 'range_rate[t]': range_rate,
    'std_rate[t]': std_rate,
    # 'wavg_rate[t]': wavg_rate
}

for k, v in to_add.items():
    _df = pd.DataFrame(v).reset_index()
    _df.columns = ['date', k]
    feature_group_1 = pd.merge(feature_group_1, _df, how='left', on='date')

feature_group_1['mean_rate[t-1]'] = feature_group_1['mean_rate[t]'].shift()
feature_group_1['std_rate[t-1]'] = feature_group_1['std_rate[t]'].shift()

# In[268]:


# Также рассмотрим спред между средневзвешанными процентными ставками по кредитам и депозитам
credits_and_deposits = pd.read_excel('data/datasets/credits_and_deposits.xlsx')

credits_and_deposits['i_spread_30[t]'] = credits_and_deposits['i_cred_30'] - credits_and_deposits['i_dep_30']
credits_and_deposits['i_spread_31_90[t]'] = credits_and_deposits['i_cred_31_90'] - credits_and_deposits['i_dep_31_90']
credits_and_deposits['i_spread_181_year[t]'] = credits_and_deposits['i_cred_181_year'] - credits_and_deposits['i_dep_181_year']

feature_group_1 = pd.merge(feature_group_1, credits_and_deposits[['date', 'i_spread_30[t]']], how='left', on='date')
feature_group_1 = pd.merge(feature_group_1, credits_and_deposits[['date', 'i_spread_31_90[t]']], how='left', on='date')
feature_group_1 = pd.merge(feature_group_1, credits_and_deposits[['date', 'i_spread_181_year[t]']], how='left', on='date')


# Добавляем наблюдаемую инфляцию за месяц (из данных опроса ЦБ РФ)
feature_group_1 = pd.merge(feature_group_1, expected_and_observed_inflation[['date', 'observed_month_inflation']], how='left', on='date')
feature_group_1 = feature_group_1.rename(columns={"observed_month_inflation" : "observed_inflation[t]"})

# In[269]:


# Также добавим агрегированные статистики по индексу MOEX (имеем date, close, volume)
moex = pd.read_excel('data/datasets/moex.xlsx')
moex = moex.set_index('date')

moex_monthly = pd.DataFrame(moex['close'].resample("ME").mean()).reset_index().rename(columns={"close" : "mean_close[t]"})
moex_monthly['std_close[t]'] = moex['close'].resample("ME").std().values
moex_monthly['return_moex[t]'] = (moex['close'].resample("ME").last() / moex['close'].resample("ME").first() - 1).values
moex_monthly['std_return_moex[t]'] = moex['close'].pct_change().resample("ME").std().values

feature_group_1 = pd.merge(feature_group_1, moex_monthly[['date', 'mean_close[t]']], how='left', on='date')
feature_group_1 = pd.merge(feature_group_1, moex_monthly[['date', 'std_close[t]']], how='left', on='date')
feature_group_1 = pd.merge(feature_group_1, moex_monthly[['date', 'return_moex[t]']], how='left', on='date')
feature_group_1 = pd.merge(feature_group_1, moex_monthly[['date', 'std_return_moex[t]']], how='left', on='date')

# In[270]:


# Собранный блок фичией 1
feature_group_1.to_excel("data/datasets/features/feature_group_1.xlsx")
feature_group_1.head()

# #### <center> 5.3.2 Реальный макроэкономический блок [Done] <center>
# ##### Link: https://rosstat.gov.ru/enterprise_industrial
# ##### Link: https://rosstat.gov.ru/statistics/roznichnayatorgovlya
# ##### Link: https://rosstat.gov.ru/labor_market_employment_salaries
# ##### Link: https://ru.tradingview.com/symbols/ECONOMICS-RUUR/reports-history/
# ##### Link: https://www.cbr.ru/statistics/bank_sector/sors/
# ##### Link: https://wciom.ru/ratings/indeks-potrebitelskogo-doverija

# In[271]:


# Берем данные по ключевым промышленным дивизионам и считаем темпы роста (в логарифмах для симметрии)
production = pd.read_excel('data/datasets/production_index.xlsx', sheet_name="Final")

production['g_mining'] = np.log(production['mining']) - np.log(production['mining'].shift())
production['g_manufacturing'] = np.log(production['manufacturing']) - np.log(production['manufacturing'].shift())
production['g_utilities'] = np.log(production['utilities']) - np.log(production['utilities'].shift())
production['g_water_waste'] = np.log(production['water_waste']) - np.log(production['water_waste'].shift())
production = production.dropna()

# Применяем PCA, т.к. темпы роста сильно скоррелированы и мы хотим выделить один основной сигнал из показателей четырех
# Можно еще построить график
pca = PCA(n_components=1)
production['pca_g_production_index[t]'] = pca.fit_transform(production[['g_mining', 'g_manufacturing', 'g_utilities', 'g_water_waste']].dropna())
feature_group_2 = production[['date', 'pca_g_production_index[t]']].copy()
feature_group_2['pca_g_production_index[t-1]'] = feature_group_2['pca_g_production_index[t]'].shift()
feature_group_2['mean_g_production_index[t]'] = 0.25 * (
    production['g_mining'] + production['g_manufacturing'] + production['g_utilities'] + production['g_water_waste']
)
feature_group_2['mean_g_production_index[t-1]'] = feature_group_2['mean_g_production_index[t]'].shift()

# In[272]:


# Добавляем данные из оборота розничной торговли - берем темпы прироста
retail_turnover = pd.read_excel('data/datasets/retail_turnover.xlsx', sheet_name="Final")
retail_turnover['g_retail_food[t]'] = retail_turnover['retail_food'].pct_change()
retail_turnover['g_retail_nonfood[t]'] = retail_turnover['retail_nonfood'].pct_change()

feature_group_2 = pd.merge(feature_group_2, retail_turnover[['date', 'g_retail_food[t]', 'g_retail_nonfood[t]']], how='left', on='date')
feature_group_2['g_retail_food[t-1]'] = feature_group_2['g_retail_food[t]'].shift()
feature_group_2['g_retail_nonfood[t-1]'] = feature_group_2['g_retail_nonfood[t]'].shift()

# In[273]:


# Добавляем номинальную среднемесячную зарплату - точнее прирост реальной заработной платы, чтобы обеспечить стационарность инструмента
nominal_salary = pd.read_excel('data/datasets/nominal_salary.xlsx')
nominal_salary['real_salary'] = nominal_salary['nominal_salary'] / nominal_salary['CPI (base = 31.12.2018)']
nominal_salary['g_real_salary[t]'] = nominal_salary['real_salary'].pct_change(fill_method=None)

feature_group_2 = pd.merge(feature_group_2, nominal_salary[['date', 'g_real_salary[t]']], how='left', on='date')
feature_group_2['g_real_salary[t-1]'] = feature_group_2['g_real_salary[t]'].shift()

# In[274]:


# Добавляем уровень безработицы
unemployment_rate = pd.read_excel('data/datasets/unemployment.xlsx')
unemployment_rate = unemployment_rate.rename(columns={"unemployment_rate" : "unemployment_rate[t]"})
unemployment_rate = unemployment_rate.sort_values(by='date')

unemployment_rate['unemployment_rate_diff[t]'] = unemployment_rate['unemployment_rate[t]'].diff()

feature_group_2 = pd.merge(feature_group_2, unemployment_rate, on='date', how='left')

# In[275]:


# Добавляем темп прироста общего объема кредитов, выданных физлицам
household_loans = pd.read_excel('data/datasets/household_loans.xlsx')
household_loans['g_household_loans[t]'] = household_loans['household_loans'].pct_change()
feature_group_2 = pd.merge(feature_group_2, household_loans[['date', 'g_household_loans[t]']], on='date', how='left')
feature_group_2['g_household_loans[t-1]'] = feature_group_2['g_household_loans[t]'].shift()

# In[276]:


# Добавим индекс потребительского доверия от ВЦИОМа
consumer_confidence = pd.read_excel('data/datasets/consumer_confidence.xlsx')
consumer_confidence = consumer_confidence.rename(columns={"consumer_confidence" : "consumer_confidence[t]"})

feature_group_2 = pd.merge(feature_group_2, consumer_confidence, on='date', how='left')
feature_group_2['consumer_confidence[t-1]'] = feature_group_2['consumer_confidence[t]'].shift(1)
feature_group_2['consumer_confidence[t-2]'] = feature_group_2['consumer_confidence[t]'].shift(2)

# In[277]:


feature_group_2.to_excel("data/datasets/features/feature_group_2.xlsx")
feature_group_2.head()

# #### <center> 5.3.3 Внешний блок [Done] <center>
# ##### Link: https://www.finam.ru/quote/commodities6/bz/export/
# ##### Link: https://www.finam.ru/quote/forex/usdrub/
# ##### Link: https://fred.stlouisfed.org/series/VIXCLS
# ##### Link: https://fred.stlouisfed.org/series/FEDFUNDS

# In[278]:


# Добавляем цены на нефть марки brent, как эталона (ежемесячное изменение цены, волатильность цены)
brent_oil_price = pd.read_excel('data/datasets/brent_oik_price.xlsx', sheet_name='Final').set_index('date')
feature_group_3 = feature_group_1[['date']].copy()
brent_oil_price['daily_log_return'] = np.log(brent_oil_price['price']) - np.log(brent_oil_price['price'].shift())
feature_group_3 = feature_group_3.merge(
    brent_oil_price['daily_log_return'].resample("ME").sum().reset_index().rename(columns={'daily_log_return' : 'total_return_brent_price[t]'}), 
    how='left', on='date'
)
feature_group_3 = feature_group_3.merge(
    brent_oil_price['daily_log_return'].resample("ME").std().reset_index().rename(columns={'daily_log_return' : 'std_return_brent_price[t]'}), 
    how='left', on='date'
)

# In[279]:


# Аналогично, но для курса доллара к рублю
usd_rub_currency = pd.read_excel("data/datasets/usd_rub_currency.xlsx").set_index("date")
usd_rub_currency['daily_log_return'] = np.log(usd_rub_currency['price']) - np.log(usd_rub_currency['price'].shift())
feature_group_3 = feature_group_3.merge(
    usd_rub_currency['daily_log_return'].resample("ME").sum().reset_index().rename(columns={'daily_log_return' : 'total_return_usd_rub_price[t]'}), 
    how='left', on='date'
)
feature_group_3 = feature_group_3.merge(
    usd_rub_currency['daily_log_return'].resample("ME").std().reset_index().rename(columns={'daily_log_return' : 'std_return_usd_rub_price[t]'}), 
    how='left', on='date'
)

# In[280]:


# VIX (CBOE Volatility Index) — "индекс страха"
# Показывает ожидаемую рынком волатильность индекса S&P 500 на основе цен опционов.
cboe_vix = pd.read_excel('data/datasets/cboe_vix.xlsx').set_index('date')
feature_group_3 = feature_group_3.merge(
    cboe_vix['VIXCLS'].resample("ME").mean().reset_index().rename(columns={"VIXCLS" : "mean_vix[t]"})
)
feature_group_3 = feature_group_3.merge(
    cboe_vix['VIXCLS'].resample("ME").std().reset_index().rename(columns={"VIXCLS" : "std_vix[t]"})
)

# In[281]:


# Берем ставку ФРС США. Из данных доступно средне-месячное значение этого показателя
federal_funds_rate = pd.read_excel("data/datasets/federal_funds_rate.xlsx").rename(columns={"federal_funds_rate" : "federal_funds_rate[t]"})
feature_group_3 = feature_group_3.merge(
    federal_funds_rate, how='left', on='date'
)

# In[282]:


# Собираем лаги

feature_group_3['total_return_brent_price[t-1]'] = feature_group_3['total_return_brent_price[t]'].shift(1)
feature_group_3['std_return_brent_price[t-1]'] = feature_group_3['std_return_brent_price[t]'].shift(1)
feature_group_3['total_return_usd_rub_price[t-1]'] = feature_group_3['total_return_usd_rub_price[t]'].shift(1)
feature_group_3['std_return_usd_rub_price[t-1]'] = feature_group_3['std_return_usd_rub_price[t]'].shift(1)
feature_group_3['mean_vix[t-1]'] = feature_group_3['mean_vix[t]'].shift(1)
feature_group_3['std_vix[t-1]'] = feature_group_3['std_vix[t]'].shift(1)
feature_group_3['federal_funds_rate[t-1]'] = feature_group_3['federal_funds_rate[t]'].shift(1)

# In[283]:


feature_group_3.to_excel("data/datasets/features/feature_group_3.xlsx")
feature_group_3.head()

# ### <center> 5.4 Feature Selection [Убираем сильно скоррелированные и нестационарные признаки] <center>
# #### Соглсано экономической теории большинство признаков стационарны - стоит привести в работе это обоснование

# In[19]:


feature_group_1 = pd.read_excel("data/datasets/features/feature_group_1.xlsx", index_col=0)
feature_group_2 = pd.read_excel("data/datasets/features/feature_group_2.xlsx", index_col=0)
feature_group_3 = pd.read_excel("data/datasets/features/feature_group_3.xlsx", index_col=0)
groups = [feature_group_1, feature_group_2, feature_group_3]

# Порог корреляции для определения мультиколлинеарности
THRESHOLD = 0.8

pd.DataFrame({
    "Группа признаков" : ["Монетарно-финансовый блок", "Блок фундаментальных макроэкономических условий", "Внешний блок"],
    "Число наблюдений" : [len(group) for group in groups],
    "Число полных наблюдений" : [len(group.dropna()) for group in groups],
    "Число признаков" : [len(group.columns) - 1 for group in groups]
})

# #### <center> 5.4.1 Feature group 1 - Mitigate multicolliniarity <center>

# In[3]:


feature_group_1, report = utilities.filter_stationary_series(
    feature_group_1.set_index('date'),
    alpha=1,
    regression="ct",
    verbose=True
)
print(f"Процент сохранившихся признаков: {report['stationary'].mean() * 100:.2f}%")

# In[11]:


high_corr_df, corr_matrix = utilities.analyze_high_correlations(feature_group_1, threshold=THRESHOLD)
utilities.plot_high_correlations(high_corr_df, THRESHOLD)
utilities.plot_correlation_heatmap_with_threshold(
    corr_matrix, 
    THRESHOLD, 
    title=f'Корреляционная матрица для монетарно-финансового блока (|corr| > {THRESHOLD})',
    savename="feature_1_corr"
)

# In[378]:


cleaned_feature_group_1 = utilities.select_features_by_correlation_threshold(
    feature_group_1.drop(['mean_rate[t-1]', 'mean_close[t]'], axis=1), 
    threshold=THRESHOLD
).reset_index()
cleaned_feature_group_1.to_excel('data/datasets/features/cleaned_feature_group_1.xlsx')

# #### <center> 5.4.2 Feature group 2 - Mitigate multicolliniarity <center>

# In[9]:


high_corr_df, corr_matrix = utilities.analyze_high_correlations(feature_group_2.drop('date', axis=1), threshold=THRESHOLD)
utilities.plot_high_correlations(high_corr_df, THRESHOLD)
utilities.plot_correlation_heatmap_with_threshold(
    corr_matrix, 
    THRESHOLD, 
    title=f'Корреляционная матрица для блока фундаментальных макроэкономических условий (|corr| > {THRESHOLD})',
    savename="feature_2_corr"
)

# In[380]:


feature_group_2, report = utilities.filter_stationary_series(
    feature_group_2.set_index('date'),
    alpha=1,
    regression="ct",
    verbose=False
)
print(f"Процент сохранившихся признаков: {report['stationary'].mean() * 100:.2f}%")

# In[381]:


cleaned_feature_group_2 = utilities.select_features_by_correlation_threshold(
    feature_group_2, 
    threshold=THRESHOLD
).reset_index()
cleaned_feature_group_2.to_excel('data/datasets/features/cleaned_feature_group_2.xlsx')

# #### <center> 5.4.3 Feature group 3 - Mitigate multicolliniarity <center>

# In[382]:


feature_group_3, report = utilities.filter_stationary_series(
    feature_group_3.set_index('date'),
    alpha=1,
    regression="ct",
    verbose=False
)
print(f"Процент сохранившихся признаков: {report['stationary'].mean() * 100:.2f}%")

# In[12]:


high_corr_df, corr_matrix = utilities.analyze_high_correlations(feature_group_3, threshold=THRESHOLD)
utilities.plot_high_correlations(high_corr_df, THRESHOLD)
utilities.plot_correlation_heatmap_with_threshold(
    corr_matrix, 
    THRESHOLD, 
    title=f'Корреляционная матрица для внешнего блока (|corr| > {THRESHOLD})',
    savename="feature_3_corr"
)

# In[384]:


cleaned_feature_group_3 = utilities.select_features_by_correlation_threshold(
    feature_group_3.drop(['federal_funds_rate[t-1]'], axis=1),
    threshold=THRESHOLD
).reset_index()
cleaned_feature_group_3.to_excel('data/datasets/features/cleaned_feature_group_3.xlsx')

# ### <center> 5.5 Feature Concatenation <center>

# In[385]:


features = cleaned_feature_group_1.merge(cleaned_feature_group_2, how='inner', on='date').merge(cleaned_feature_group_3, how='inner', on='date')
features.head()

# ## <center> 6 Model Pipeline <center>

# In[23]:


# Перед оценкой коэффициентов на всем наборе данных, попробуем это сделать на отдельных блоках и посмотреть на распределение оценок
cleaned_feature_group_1 = pd.read_excel('data/datasets/features/cleaned_feature_group_1.xlsx', index_col=0)
cleaned_feature_group_2 = pd.read_excel('data/datasets/features/cleaned_feature_group_2.xlsx', index_col=0)
cleaned_feature_group_3 = pd.read_excel('data/datasets/features/cleaned_feature_group_3.xlsx', index_col=0)
feature_group_1 = pd.read_excel('data/datasets/features/feature_group_1.xlsx', index_col=0)
feature_group_2 = pd.read_excel('data/datasets/features/feature_group_2.xlsx', index_col=0)
feature_group_3 = pd.read_excel('data/datasets/features/feature_group_3.xlsx', index_col=0)
consumption = pd.read_excel('data/datasets/consumption.xlsx', index_col=0)
real_return = pd.read_excel('data/datasets/real_return.xlsx', index_col=0)


initial_dataset = consumption.copy()[['date', 'C[t]']]
initial_dataset = pd.merge(initial_dataset, real_return[['date', 'R[t+1]']], how='left', on='date')
initial_dataset['C[t+1]'] = initial_dataset['C[t]'].shift(-1)
initial_dataset['C_ratio'] = initial_dataset['C[t+1]'] / initial_dataset['C[t]']

# ### <center> 6.1 Model Pipeline: feature_group_1 <center>

# In[4]:


test_feature_group_1 = initial_dataset.copy().merge(cleaned_feature_group_1, on='date', how='left').dropna().set_index('date')
assert test_feature_group_1.isna().sum().sum() == 0

cols = [col for col in test_feature_group_1.columns if col not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]

data = {
    "R[t+1]" : test_feature_group_1['R[t+1]'],
    "C_ratio" : test_feature_group_1['C_ratio'],
    "Const" : np.ones(len(test_feature_group_1)),
}

for col in cols:
    data[col] = test_feature_group_1[col]

def make_moment(name):
    def moment(theta, dp):
        beta, gamma = theta
        m = beta * (dp['C_ratio'] ** (-gamma)) * (1 + dp['R[t+1]']) - 1
        return m * dp[name]
    return moment

MOMENT_NAMES = [col for col in data if col not in ("R[t+1]", "C_ratio")]

test_feature_group_1.head()

# In[5]:


moments = [make_moment(name) for name in MOMENT_NAMES]
f2_indices = [3]
a_indexes = [1]
print(f"[CHECK] {' + '.join([MOMENT_NAMES[i] for i in range(len(MOMENT_NAMES)) if i in f2_indices])}")
W_1, pval_1, theta_1, cov_1 = unconditional_relevance(
    data=data,
    moments=moments,
    f2_indexes=f2_indices,
    # a_indexes=a_indexes,
    theta_init=[0, 0],
    # verbose=True
)
print(f"{W_1=}")
print(f"{pval_1=}")
print(f"{theta_1=}")

# In[5]:


utilities.plot_theta_estimates(
    theta_hat=theta_1,
    cov_theta=cov_1,
    alpha=0.05,
    param_names=['beta', 'gamma']
)

# ### <center> 6.2 Model Pipeline: feature_group_2 <center>

# In[6]:


test_feature_group_2 = initial_dataset.copy().merge(cleaned_feature_group_2, on='date', how='left').dropna().set_index('date')
assert test_feature_group_2.isna().sum().sum() == 0

cols = [col for col in test_feature_group_2.columns if col not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]

data = {
    "R[t+1]" : test_feature_group_2['R[t+1]'],
    "C_ratio" : test_feature_group_2['C_ratio'],
    "Const" : np.ones(len(test_feature_group_2)),
}

for col in cols:
    data[col] = test_feature_group_2[col]

def make_moment(name):
    def moment(theta, dp):
        beta, gamma = theta
        m = beta * (dp['C_ratio'] ** (-gamma)) * (1 + dp['R[t+1]']) - 1
        return m * dp[name]
    return moment

MOMENT_NAMES = [col for col in data if col not in ("R[t+1]", "C_ratio")]

test_feature_group_2.head()

# In[7]:


moments = [make_moment(name) for name in MOMENT_NAMES]
f2_indices = [7]
a_indexes = [1]
print(f"[CHECK] {' + '.join([MOMENT_NAMES[i] for i in range(len(MOMENT_NAMES)) if i in f2_indices])}")
W_2, pval_2, theta_2, cov_2 = unconditional_relevance(
    data=data,
    moments=moments,
    f2_indexes=f2_indices,
    # a_indexes=a_indexes,
    theta_init=[0, 0],
    # verbose=True
)
print(f"{W_2=}")
print(f"{pval_2=}")
print(f"{theta_2=}")

# In[8]:


utilities.plot_theta_estimates(
    theta_hat=theta_2,
    cov_theta=cov_2,
    alpha=0.05,
    param_names=['beta', 'gamma']
)

# ### <center> 6.3 Model Pipeline: feature_group_3 <center>

# In[9]:


test_feature_group_3 = initial_dataset.copy().merge(cleaned_feature_group_3, on='date', how='left').dropna().set_index('date')
assert test_feature_group_3.isna().sum().sum() == 0

cols = [col for col in test_feature_group_3.columns if col not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]

data = {
    "R[t+1]" : test_feature_group_3['R[t+1]'],
    "C_ratio" : test_feature_group_3['C_ratio'],
    "Const" : np.ones(len(test_feature_group_3)),
}

for col in cols:
    data[col] = test_feature_group_3[col]

def make_moment(name):
    def moment(theta, dp):
        beta, gamma = theta
        m = beta * (dp['C_ratio'] ** (-gamma)) * (1 + dp['R[t+1]']) - 1
        return m * dp[name]
    return moment

MOMENT_NAMES = [col for col in data if col not in ("R[t+1]", "C_ratio")]

test_feature_group_3.head()

# In[10]:


moments = [make_moment(name) for name in MOMENT_NAMES]
f2_indices = [10]
a_indexes = [1]
print(f"[CHECK] {' + '.join([MOMENT_NAMES[i] for i in range(len(MOMENT_NAMES)) if i in f2_indices])}")
W_3, pval_3, theta_3, cov_3 = unconditional_relevance(
    data=data,
    moments=moments,
    f2_indexes=f2_indices,
    # a_indexes=a_indexes,
    theta_init=[0, 0],
    # verbose=True
)
print(f"{W_3=}")
print(f"{pval_3=}")
print(f"{theta_3=}")

# In[11]:


utilities.plot_theta_estimates(
    theta_hat=theta_3,
    cov_theta=cov_3,
    alpha=0.05,
    param_names=['beta', 'gamma']
)

# ### <center> 6.4 Plot three estimations <center>

# In[12]:


utilities.plot_gmm_estimates(
    thetas=[theta_1, theta_2, theta_3],
    covs=[cov_1, cov_2, cov_3],
    names=['Монетарно-финансовый блок', 'Блок фундаментальных макроэкономических условий', 'Внешний блок'],
    conf=0.95
)

# In[13]:


# Параметр beta одинаков во всех трех блоках
# Параметр gamma стат значим от нуля только во внешнем блоке. Это интересно
# Получается, что на восприятие риска влиют больше макроэкономические шоки, которые выражаются через колебания курса, цен на нефть, ставки ФРС США..

utilities.wald_test_pairwise(
    thetas=[theta_1, theta_2, theta_3],
    covs=[cov_1, cov_2, cov_3],
    names=['Монетарно-финансовый блок', 'Блок фундаментальных макроэкономических условий', 'Внешний блок']
)

# ### <center> 6.5 Combination of feature blocks and reducing multicollinearity <center>

# In[14]:


blocks = (
    feature_group_1
    .merge(feature_group_2, how='left', on='date')
    .merge(feature_group_3, how='left', on='date')
).set_index('date')

# In[15]:


THRESHOLD = 0.86

high_corr_df, corr_matrix = utilities.analyze_high_correlations(blocks, threshold=THRESHOLD)
utilities.plot_high_correlations(high_corr_df, THRESHOLD)
utilities.plot_correlation_heatmap_with_threshold(corr_matrix, THRESHOLD)

# In[16]:


cleaned_blocks = utilities.select_features_by_correlation_threshold(
    blocks.drop(['mean_rate[t-1]', 'federal_funds_rate[t-1]'], axis=1),
    threshold=THRESHOLD
).reset_index()
main_dataset = pd.merge(initial_dataset, cleaned_blocks, how='left', on='date').set_index('date').dropna()

# In[17]:


cols = [col for col in main_dataset.columns if col not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]

data = {
    "R[t+1]" : main_dataset['R[t+1]'],
    "C_ratio" : main_dataset['C_ratio'],
    "Const" : np.ones(len(main_dataset)),
}

for col in cols:
    data[col] = main_dataset[col]

def make_moment(name):
    def moment(theta, dp):
        beta, gamma = theta
        m = beta * (dp['C_ratio'] ** (-gamma)) * (1 + dp['R[t+1]']) - 1
        return m * dp[name]
    return moment

MOMENT_NAMES = [col for col in data if col not in ("R[t+1]", "C_ratio")]
print(f"[SHAPE: n, k] = ({len(main_dataset)}, {len(main_dataset.columns)})")
main_dataset.head()

# In[18]:


moments = [make_moment(name) for name in MOMENT_NAMES]
f2_indices = [19]
a_indexes = [0]
print(f"[CHECK] {' + '.join([MOMENT_NAMES[i] for i in range(len(MOMENT_NAMES)) if i in f2_indices])}")
W, pval, theta, cov = unconditional_relevance(
    data=data,
    moments=moments,
    f2_indexes=f2_indices,
    # a_indexes=a_indexes,
    theta_init=[0, 0],
    # verbose=True
)
print(f"{W=}")
print(f"{pval=}")
print(f"{theta=}")

# In[19]:


utilities.plot_theta_estimates(
    theta_hat=theta,
    cov_theta=cov,
    alpha=0.05,
    param_names=['beta', 'gamma']
)

# In[20]:


utilities.plot_gmm_estimates(
    thetas=[theta],
    covs=[cov],
    names=['Общее поле'],
    conf=0.95
)

# In[21]:


# Получается, что риск аверсия нулевая, что в целом согласуется с результатами Roberta Halla?

# In[22]:


test_feature_group_1.to_excel('data/datasets/prepared_datasets/test_feature_group_1.xlsx')
test_feature_group_2.to_excel('data/datasets/prepared_datasets/test_feature_group_2.xlsx')
test_feature_group_3.to_excel('data/datasets/prepared_datasets/test_feature_group_3.xlsx')
main_dataset.to_excel('data/datasets/prepared_datasets/main_dataset.xlsx')

# ### <center> 6.6 Unconditional Relevance Test Application [URTA] <center>

# In[2]:


# Для теста на безусловную релевантность бОльший интерес представляет возможность понимать,
# выбранный инструмент в прицнипе способен идентифицировать оба параметра (beta и gamma)
# Предлагается прогонять его не в группах, а по одному
# Получать pvalue
# Формировать те, что
#     1) значимы
#     2) не значимы
# Со второй группой будем работать отдельно уже в рамках partial тестов

test_feature_group_1 = pd.read_excel('data/datasets/prepared_datasets/test_feature_group_1.xlsx', index_col='date')
test_feature_group_2 = pd.read_excel('data/datasets/prepared_datasets/test_feature_group_2.xlsx', index_col='date')
test_feature_group_3 = pd.read_excel('data/datasets/prepared_datasets/test_feature_group_3.xlsx', index_col='date')
main_dataset = pd.read_excel('data/datasets/prepared_datasets/main_dataset.xlsx', index_col='date')

# In[3]:


# URTA_of_feature_group_1 = utilities.implement_URTA(test_feature_group_1, 'Монетарно-финансовый блок')
# URTA_of_feature_group_1.to_excel('data/datasets/test_reports/URTA/URTA_of_feature_group_1.xlsx')
URTA_of_feature_group_1 = pd.read_excel('data/datasets/test_reports/URTA/URTA_of_feature_group_1.xlsx', index_col=0)
URTA_of_feature_group_1

# In[4]:


# URTA_of_feature_group_2 = utilities.implement_URTA(test_feature_group_2, 'Блок фундаментальных макроэкономических условий')
# URTA_of_feature_group_2.to_excel('data/datasets/test_reports/URTA/URTA_of_feature_group_2.xlsx')
URTA_of_feature_group_2 = pd.read_excel('data/datasets/test_reports/URTA/URTA_of_feature_group_2.xlsx', index_col=0)
URTA_of_feature_group_2

# In[5]:


# URTA_of_feature_group_3 = utilities.implement_URTA(test_feature_group_3, 'Внешний блок')
# URTA_of_feature_group_3.to_excel('data/datasets/test_reports/URTA/URTA_of_feature_group_3.xlsx')
URTA_of_feature_group_3 = pd.read_excel('data/datasets/test_reports/URTA/URTA_of_feature_group_3.xlsx', index_col=0)
URTA_of_feature_group_3

# In[6]:


# URTA_of_main_dataset = utilities.implement_URTA(main_dataset, 'Общее поле')
# URTA_of_main_dataset.to_excel('data/datasets/test_reports/URTA/URTA_of_main_dataset.xlsx')
URTA_of_main_dataset = pd.read_excel('data/datasets/test_reports/URTA/URTA_of_main_dataset.xlsx', index_col=0)
URTA_of_main_dataset

# In[7]:


# utilities.plot_moment_relevance(URTA_of_main_dataset, title='URTA: Информационное поле агента')
utilities.plot_moment_relevance(URTA_of_main_dataset, title='Informational field of Agent')

# ### <center> 6.7 Conditional Relevance Test Application [CRTA] <center>

# In[8]:


# CRTA_of_feature_group_1 = utilities.implement_CRTA(test_feature_group_1, name='Монетарно-финансовый блок', significance_level=0.05)
# CRTA_of_feature_group_1.to_excel('data/datasets/test_reports/CRTA/CRTA_of_feature_group_1.xlsx')
CRTA_of_feature_group_1 = pd.read_excel('data/datasets/test_reports/CRTA/CRTA_of_feature_group_1.xlsx', index_col=0)
CRTA_of_feature_group_1


[const 🟢] и [[i_spread_31_90[t] 🟡], [], []]

# In[9]:


# CRTA_of_feature_group_2 = utilities.implement_CRTA(test_feature_group_2, name='Блок фундаментальных макроэкономических условий', significance_level=0.05)
# CRTA_of_feature_group_2.to_excel('data/datasets/test_reports/CRTA/CRTA_of_feature_group_2.xlsx')
CRTA_of_feature_group_2 = pd.read_excel('data/datasets/test_reports/CRTA/CRTA_of_feature_group_2.xlsx', index_col=0)
CRTA_of_feature_group_2

# In[10]:


# Тут только один или нисколько условно релевантны?
# CRTA_of_feature_group_3 = utilities.implement_CRTA(test_feature_group_3, name='Внешний блок', significance_level=0.05)
# CRTA_of_feature_group_3.to_excel('data/datasets/test_reports/CRTA/CRTA_of_feature_group_3.xlsx')
CRTA_of_feature_group_3 = pd.read_excel('data/datasets/test_reports/CRTA/CRTA_of_feature_group_3.xlsx', index_col=0)
CRTA_of_feature_group_3

# In[11]:


# CRTA_of_main_dataset = utilities.implement_CRTA(main_dataset, name='Общее поле', significance_level=0.05)
# CRTA_of_main_dataset.to_excel('data/datasets/test_reports/CRTA/CRTA_of_main_dataset.xlsx')
CRTA_of_main_dataset = pd.read_excel('data/datasets/test_reports/CRTA/CRTA_of_main_dataset.xlsx', index_col=0)
CRTA_of_main_dataset

# In[12]:


# Исключаем ставку ЦБ за месяц и ожидаем увидеть несколько осей идентификации
# CRTA_of_main_dataset_without_mean_rate = utilities.implement_CRTA(
#     main_dataset.drop(['mean_rate[t]'], axis=1), 
#     name='Общее поле', 
#     significance_level=0.05
# )
# CRTA_of_main_dataset_without_mean_rate.to_excel('data/datasets/test_reports/CRTA/CRTA_of_main_dataset_without_mean_rate.xlsx')
CRTA_of_main_dataset_without_mean_rate = pd.read_excel('data/datasets/test_reports/CRTA/CRTA_of_main_dataset_without_mean_rate.xlsx', index_col=0)
CRTA_of_main_dataset_without_mean_rate

# ### <center> 6.8 Partial Unconditional Relevance Test Application [PURTA] <center>

# In[13]:


# PURTA_of_feature_group_1 = utilities.implement_PURTA(test_feature_group_1, name='Монетарно-финансовый блок')
# PURTA_of_feature_group_1.to_excel("data/datasets/test_reports/PURTA/PURTA_of_feature_group_1.xlsx")
PURTA_of_feature_group_1 = pd.read_excel('data/datasets/test_reports/PURTA/PURTA_of_feature_group_1.xlsx', index_col=0)
PURTA_of_feature_group_1

# In[14]:


# PURTA_of_feature_group_2 = utilities.implement_PURTA(test_feature_group_2, name='Блок фундаментальных макроэкономических условий')
# PURTA_of_feature_group_2.to_excel("data/datasets/test_reports/PURTA/PURTA_of_feature_group_2.xlsx")
PURTA_of_feature_group_2 = pd.read_excel('data/datasets/test_reports/PURTA/PURTA_of_feature_group_2.xlsx', index_col=0)
PURTA_of_feature_group_2

# In[15]:


# PURTA_of_feature_group_3 = utilities.implement_PURTA(test_feature_group_3, name='Внешний блок')
# PURTA_of_feature_group_3.to_excel("data/datasets/test_reports/PURTA/PURTA_of_feature_group_3.xlsx")
PURTA_of_feature_group_3 = pd.read_excel('data/datasets/test_reports/PURTA/PURTA_of_feature_group_3.xlsx', index_col=0)
PURTA_of_feature_group_3

# In[16]:


# PURTA_of_main_dataset = utilities.implement_PURTA(main_dataset, name='Общее поле')
# PURTA_of_main_dataset.to_excel("data/datasets/test_reports/PURTA/PURTA_of_main_dataset.xlsx")
PURTA_of_main_dataset = pd.read_excel('data/datasets/test_reports/PURTA/PURTA_of_main_dataset.xlsx', index_col=0)
PURTA_of_main_dataset

# In[17]:


utilities.plot_moment_relevance_by_beta_and_gamma(PURTA_of_main_dataset, title='Informational Field')

# ### <center> 6.9 Partial Conditional Relevance Test Application [PCRTA] <center>

# In[18]:


# PCRTA_of_feature_group_1 = utilities.implement_PCRTA(
#     input=test_feature_group_1,
#     CRTA_results=CRTA_of_feature_group_1,
#     name='Монетарно-финансовый блок', 
#     significance_level=0.05
# )
# PCRTA_of_feature_group_1.to_excel('data/datasets/test_reports/PCRTA/PCRTA_of_feature_group_1.xlsx')
PCRTA_of_feature_group_1 = pd.read_excel('data/datasets/test_reports/PCRTA/PCRTA_of_feature_group_1.xlsx', index_col=0)
PCRTA_of_feature_group_1

# In[19]:


# PCRTA_of_feature_group_2 = utilities.implement_PCRTA(
#     input=test_feature_group_2,
#     CRTA_results=CRTA_of_feature_group_2,
#     name='Блок фундаментальных макроэкономических условий', 
#     significance_level=0.05
# )
# PCRTA_of_feature_group_2.to_excel('data/datasets/test_reports/PCRTA/PCRTA_of_feature_group_2.xlsx')
PCRTA_of_feature_group_2 = pd.read_excel('data/datasets/test_reports/PCRTA/PCRTA_of_feature_group_2.xlsx', index_col=0)
PCRTA_of_feature_group_2

# In[20]:


# PCRTA_of_feature_group_3 = utilities.implement_PCRTA(
#     input=test_feature_group_3, 
#     CRTA_results=CRTA_of_feature_group_3,
#     name='Внешний блок', 
#     significance_level=0.05
# )
# PCRTA_of_feature_group_3.to_excel('data/datasets/test_reports/PCRTA/PCRTA_of_feature_group_3.xlsx')
PCRTA_of_feature_group_3 = pd.read_excel('data/datasets/test_reports/PCRTA/PCRTA_of_feature_group_3.xlsx', index_col=0)
PCRTA_of_feature_group_3

# In[21]:


# PCRTA_of_main_dataset = utilities.implement_PCRTA(
#     input=main_dataset,
#     CRTA_results=CRTA_of_main_dataset,
#     name='Общее поле',
#     significance_level=0.05
# )
# PCRTA_of_main_dataset.to_excel('data/datasets/test_reports/PCRTA/PCRTA_of_main_dataset.xlsx')
PCRTA_of_main_dataset = pd.read_excel('data/datasets/test_reports/PCRTA/PCRTA_of_main_dataset.xlsx', index_col=0)
PCRTA_of_main_dataset

# ### <center> 6.10 Optimal Portfolio of Factors <center>

# In[22]:


# Собираем применения всех 4ех тестов на 3ех блоках

tables = {
    "URTA_of_feature_group_1" : URTA_of_feature_group_1,
    "URTA_of_feature_group_2" : URTA_of_feature_group_1,
    "URTA_of_feature_group_3" : URTA_of_feature_group_3,

    "CRTA_of_feature_group_1" : CRTA_of_feature_group_1,
    "CRTA_of_feature_group_2" : CRTA_of_feature_group_2,
    "CRTA_of_feature_group_3" : CRTA_of_feature_group_3,

    "PURTA_of_feature_group_1" : PURTA_of_feature_group_1,
    "PURTA_of_feature_group_2" : PURTA_of_feature_group_2,
    "PURTA_of_feature_group_3" : PURTA_of_feature_group_3,

    "PCRTA_of_feature_group_1" : PCRTA_of_feature_group_1.merge(CRTA_of_feature_group_1[['moment', 'p_value']], how='left', on='moment'),
    "PCRTA_of_feature_group_2" : PCRTA_of_feature_group_2.merge(CRTA_of_feature_group_2[['moment', 'p_value']], how='left', on='moment'),
    "PCRTA_of_feature_group_3" : PCRTA_of_feature_group_3.merge(CRTA_of_feature_group_3[['moment', 'p_value']], how='left', on='moment')
}

# In[23]:


optimal_portfolio = utilities.build_optimal_euler_portfolio(
    tables=tables,
    alpha=0.05,
    drop_const=True,
    keep_shared_only=False,
    min_block_support=1
)

# In[24]:


# Далее сравним оценки при общем поле и при оптимальном поле

optimal_dataset = main_dataset[list(optimal_portfolio['moment'].values) + ["R[t+1]", "C_ratio"]]

# In[91]:


# Получаем оценки для main_dataset

cols = [col for col in main_dataset.columns if col not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]

data = {
    "R[t+1]" : main_dataset['R[t+1]'],
    "C_ratio" : main_dataset['C_ratio'],
    "Const" : np.ones(len(main_dataset)),
}

for col in cols:
    data[col] = main_dataset[col]

def make_moment(name):
    def moment(theta, dp):
        beta, gamma = theta
        m = beta * (dp['C_ratio'] ** (-gamma)) * (1 + dp['R[t+1]']) - 1
        return m * dp[name]
    return moment

MOMENT_NAMES_total = [col for col in data if col not in ("R[t+1]", "C_ratio")]
moments = [make_moment(name) for name in MOMENT_NAMES_total]
f2_indices = [0]
a_indexes = [0]
_, _, theta_total, cov_total = unconditional_relevance(
    data=data,
    moments=moments,
    f2_indexes=f2_indices,
    theta_init=[0, 0],
)


# Получаем оценки для optimal_dataset

cols = [col for col in optimal_dataset.columns if col not in ("C[t]", "R[t+1]", "C[t+1]", "C_ratio")]

data = {
    "R[t+1]" : optimal_dataset['R[t+1]'],
    "C_ratio" : optimal_dataset['C_ratio'],
    "Const" : np.ones(len(optimal_dataset)),
}

for col in cols:
    data[col] = optimal_dataset[col]

def make_moment(name):
    def moment(theta, dp):
        beta, gamma = theta
        m = beta * (dp['C_ratio'] ** (-gamma)) * (1 + dp['R[t+1]']) - 1
        return m * dp[name]
    return moment

MOMENT_NAMES_optimal = [col for col in data if col not in ("R[t+1]", "C_ratio")]
moments = [make_moment(name) for name in MOMENT_NAMES_optimal]
f2_indices = [0]
a_indexes = [0]
_, _, theta_optimal, cov_optimal = unconditional_relevance(
    data=data,
    moments=moments,
    f2_indexes=f2_indices,
    theta_init=[0, 0],
)

# In[104]:


print(f"Признаков в просто объединенном на основе снижения мультиколлинеарности датасете: {len(MOMENT_NAMES_total)}")
print(f"Признаков в оптимальном датасете: {len(MOMENT_NAMES_optimal)}")

# In[93]:


utilities.plot_gmm_estimates(
    thetas=[theta_total, theta_optimal],
    covs=[cov_total, cov_optimal],
    names=['Общее поле', 'Оптимальное поле'],
    conf=0.95
)

# In[94]:


utilities.wald_test_pairwise(
    thetas=[theta_total, theta_optimal],
    covs=[cov_total, cov_optimal],
    names=['Общее поле', "Оптимальное поле"]
)

# In[95]:


# Заметим, что теперь мы можем сказать, что gamma не ноль!

utilities.plot_theta_estimates(
    theta_hat=theta_optimal,
    cov_theta=cov_optimal,
    alpha=0.05,
    param_names=['beta', 'gamma']
)

# In[96]:


eng2rus = {
    "i_spread_30[t]": "Спрэд ставок (≤30 дн.)",
    "i_spread_31_90[t]": "Спрэд ставок (31–90 дн.)",
    "i_spread_181_year[t]": "Спрэд ставок (181 дн.–1 г.)",

    "mean_rate[t]": "Ключевая ставка",
    "std_rate[t]": "Волатильность ставки",
    "std_rate[t-1]": "Волатильность ставки (t−1)",

    "observed_inflation[t]": "Наблюдаемая инфляция",

    "std_return_moex[t]": "Волатильность MOEX",

    "consumer_confidence[t]": "Индекс потребительского доверия",
    "consumer_confidence[t-1]": "Индекс потребительского доверия (t−1)",
    "consumer_confidence[t-2]": "Индекс потребительского доверия (t−2)",

    "federal_funds_rate[t]": "Ставка ФРС",

    "mean_vix[t]": "VIX (среднее)",
    "mean_vix[t-1]": "VIX (среднее, t−1)",

    "std_vix[t]": "Волатильность VIX",
    "std_vix[t-1]": "Волатильность VIX (t−1)",

    "std_return_brent_price[t]": "Волатильность Brent",
    "std_return_brent_price[t-1]": "Волатильность Brent (t−1)",

    "std_return_usd_rub_price[t]": "Волатильность USD/RUB",
    "std_return_usd_rub_price[t-1]": "Волатильность USD/RUB (t−1)",

    "unemployment_rate[t]": "Уровень безработицы",

    "g_real_salary[t]": "Рост реальной зарплаты",
    "g_real_salary[t-1]": "Рост реальной зарплаты (t−1)",

    "mean_g_production_index[t]": "Рост промпроизводства",
    "mean_g_production_index[t-1]": "Рост промпроизводства (t−1)",

    "g_retail_nonfood[t]": "Рост непродовольственной розницы",
}

# In[97]:


utilities.plot_optimal_euler_portfolio_graph(
    input=optimal_portfolio,
    title='Оптимальный портфель факторов для модели межвременного потребления Эйлера',
    eng2rus=eng2rus,
    figsize=(18, 12),
    dpi=500,
    wrap_width=30,
    savepath='data/images/optimal_portfolio.png'
)

# ## <center> 7 Practical Implementaion [Economics + Managment] <center>

# In[ ]:


# Нужно обосновать чем это может быть полезно с точки зрения менеджмента и экономики
