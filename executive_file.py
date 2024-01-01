import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels as sm
from sklearn.linear_model import LinearRegression
import itertools
import statsmodels.tsa as tsa
import warnings
warnings.filterwarnings('ignore')
import statsmodels.tsa.api as tsa
import datetime
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error

df = pd.read_csv("./train.csv")
sample = pd.read_csv("./sample_submission.csv")

plt.rc('font', family='Malgun Gothic')

idx_list = []
del_list = []

for i in range(len(df)):
    if ('12-31' not in df.loc[i, '일시']) and ('-02-29' not in df.loc[i, '일시']):
        del_list.append(i)
        
df2 = df.loc[del_list, :].reset_index(drop=True)
for i in range(len(df2)):
    if (i+1)%7 == 0:
        idx_list.append(i)
        
# 원래는 적합 과정이 필요하지만, 미리 적합해 놓은 parameter가 있으므로
df3 = df2.loc[idx_list, :].reset_index(drop=True)
df4 = df3.set_index('일시')
arima_setup = pd.read_csv("arima_parameter.csv")

best_mode = tsa.statespace.SARIMAX(df4['평균기온'][:3278-54], order=(1, 0, 1),
                                  seasonal_order = (0, 1, 1, 52),
                                  enforc_stationarity=False,
                                  enforce_invertibility=False)
best_results = best_mode.fit()
forecast = best_results.forecast(53)

forecast = pd.DataFrame(forecast).reset_index(drop=True)

temp_list = []
for i in range(62):
    temp_list.append(df.rolling(window=365).mean()['평균기온'][i*365])
    
temp_list = pd.DataFrame(temp_list).dropna()

start_ = datetime(2022, 12, 30)
current_ = start_
days = []

for i in range(370):
    days.append(current_)
    current_ += timedelta(days=1)
    
final = pd.DataFrame(index = days, columns = ['temp'])
final.iloc[0, 0] = -3.9

for i in range(len(forecast)):
    final.iloc[1+i*7, 0] = forecast['predicted_mean'][i]
    
final = final.astype(float).interpolate()
final2 = final[2:360]

# abs(temp_list.diff())[:-15].mean()
final2['temp'] = final2['temp']+0.563927

final2

#월별로 만들어놓은게 있는데, 아까워서 올려는 놓음
monthly = pd.read_csv("./monthly.csv")

final3 = final2.copy()

df_linear = pd.DataFrame()
for i in range(1, 13):
    if i < 10:
        num = '0'+str(i)
    else:
        num = str(i)
    df_monthly = df[df['일시'].str.contains(f'-{num}-')].reset_index(drop=True)
    
    lr_monthly = LinearRegression()
    lr_monthly.fit(np.array(df_monthly.index).reshape(-1, 1), df_monthly[['평균기온']])
    
    df_linear.loc[i, 'month'] = i
    df_linear.loc[i, 'coef'] = lr_monthly.coef_[0]
    df_linear.loc[i, 'intercept'] = lr_monthly.intercept_[0]

for i in range(1, 13):
    months = f'-{i}-'
    for j in final3.index:
        if months in str(j):
            final3.loc[j, 'temp'] += df_linear.loc[i, 'coef']

sample['평균기온'] = final3['temp']

sample.to_csv("final_.csv")

sample['평균기온'] = np.array(final3['temp'])

sample

sample.to_csv("final_.csv")

sample2 = sample.copy()

# sample2['평균기온'] -= 0.360556

sample2['평균기온'] = round(sample2['평균기온'], 1)

# sample2.columns = test2.columns

# sample2 = sample2.set_index('일시')

# round 처리 된 것
sample2.to_csv("final_3.csv")
# round 처리 안 된 것
sample.to_csv("final_sample.csv")