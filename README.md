# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 11/05/26



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv("C:/Users/admin/OneDrive/Desktop/index_1.csv")

data['datetime'] = pd.to_datetime(
    data['date'] + ' ' + data['datetime'],
    errors='coerce'
)

data = data.dropna(subset=['datetime'])

data = data.sort_values('datetime')

X = data['money']

plt.rcParams['figure.figsize'] = [12, 6]

plt.plot(X)

plt.title('Original Coffee Sales Data')
plt.xlabel('Transactions')
plt.ylabel('Money')

plt.show()

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plot_acf(X, lags=40, ax=plt.gca())

plt.title('Original Data ACF')

plt.subplot(2, 1, 2)

plot_pacf(X, lags=40, ax=plt.gca())

plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

print("\nARMA(1,1) Model Summary")
print(arma11_model.summary())

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

N = 1000

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.figure(figsize=(12, 6))

plt.plot(ARMA_1)

plt.title('Simulated ARMA(1,1) Coffee Sales')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.xlim([0, 500])

plt.show()

plot_acf(ARMA_1)

plt.title('ACF of ARMA(1,1)')

plt.show()

plot_pacf(ARMA_1)

plt.title('PACF of ARMA(1,1)')

plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

print("\nARMA(2,2) Model Summary")
print(arma22_model.summary())

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

plt.figure(figsize=(12, 6))

plt.plot(ARMA_2)

plt.title('Simulated ARMA(2,2) Coffee Sales')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.xlim([0, 500])

plt.show()

plot_acf(ARMA_2)

plt.title('ACF of ARMA(2,2)')

plt.show()

plot_pacf(ARMA_2)

plt.title('PACF of ARMA(2,2)')

plt.show()
```

OUTPUT:
ORIGINAL COFFEE SALES DATA

<img width="607" height="324" alt="image" src="https://github.com/user-attachments/assets/446af1c0-b6de-492d-98b0-e2350a2e880b" />

ORIGINAL DATA ACF
<img width="602" height="201" alt="image" src="https://github.com/user-attachments/assets/fcaf4e41-9555-424f-834b-cbaac2c32d19" />

ORIGINAL DATA PACF
<img width="601" height="205" alt="image" src="https://github.com/user-attachments/assets/0275b8eb-3add-48b4-9d41-b659869bc5b1" />

SIMULATED ARMA(1,1) PROCESS:

AUTOCORRELATION
<img width="601" height="315" alt="image" src="https://github.com/user-attachments/assets/06fd14a2-189f-43c8-960d-7bbc00374aa3" />
PARTIAL AUTOCORRELATION
<img width="608" height="320" alt="image" src="https://github.com/user-attachments/assets/6e46e891-6a42-437d-9610-923b69bf57e7" />

SIMULATED ARMA(1,1) COFFEE SALES
<img width="607" height="326" alt="image" src="https://github.com/user-attachments/assets/84b26e37-a168-4eae-8107-c7b7883aca0c" />


SIMULATED ARMA(2,2) PROCESS:

AUTOCORRELATION
<img width="602" height="320" alt="image" src="https://github.com/user-attachments/assets/6319fb85-e458-4c3a-8cea-e2888f1bb329" />
PARTIAL AUTOCORRELATION
<img width="607" height="305" alt="image" src="https://github.com/user-attachments/assets/792e0147-c4b6-4f31-9735-e1c23d87573e" />

SIMULATED ARMA(2,2) COFFEE SALES
<img width="598" height="309" alt="image" src="https://github.com/user-attachments/assets/72e31767-45d2-4710-bbe4-e72c2a7dac4d" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
