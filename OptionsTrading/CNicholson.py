'''
Fisrt of all, what is the Crank Nicholson Method? Basically, the method helps us figure out the past values of the an option contract based on its current value
or values at expiration.

Let's say we have a stock currently priced at $100, and we want to price a call option with a strike price of $100 that expires in 3 months.
This would be how we lay out our matrix:

Time (months)   0    1    2    3
Stock Price
$120           ?    ?    ?    20
$110           ?    ?    ?    10
$100           ?    ?    ?    0
$90            ?    ?    ?    0
$80            ?    ?    ?    0

Knowing the current values of the option based on the underlying stock price, using the Crank Nicholson method, we can figure out the past values, 
in this case the "?" values, of the option.

=> In this code, we will be figuring out this matrix based on a series of assumptions and plotting them.
'''
#-------------------------------------
#importing libs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
plt.style.use('ggplot')

def crank_nicholson(type, expiration, sigma, r, strike, NAS, NTS):
    S_min = strike/3
    S_max = strike*2
    dS = (S_max-S_min)/NAS
    dt = expiration/NTS
    S = np.arange(0, NAS+1)* dS +S_min
    V = np.zeros((NAS + 1, NTS + 1))
    payoff = np.maximum((strike-S), 0)
    V[:, -1] = payoff
    V[-1, :] = 0
    V[0, :] = np.maximum(strike - S_min, 0) * np.exp(-r * np.linspace(0, expiration, NTS + 1)[::-1])
    I = np.arange(0,NAS+1)
    alpha = 0.25 * dt * ((sigma**2) * (I**2) - r*I)
    beta = -dt * 0.5 * (sigma**2 * (I**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (I**2) + r * I)
    #creating the matrix
    ML = sparse.diags([-alpha[2:], 1-beta[1:], -gamma[1:]], [-1,0,1], shape=(NAS-1, NAS-1)).tocsc()
    MR = sparse.diags([alpha[2:], 1+beta[1:], gamma[1:]], [-1,0,1], shape=(NAS-1, NAS-1)).tocsc()

    for t in range(NTS - 1, -1, -1):
        #solving the actual matrix
        boundary_t = np.zeros(NAS - 1)
        boundary_t[0] = alpha[1] * (V[0, t] + V[0, t + 1]) -alpha[0] * V[0, t + 1]
        boundary_t[-1] = gamma[NAS - 1] * (V[NAS, t] + V[NAS, t + 1])
        b = MR.dot(V[1:NAS, t + 1]) + boundary_t
        V[1:NAS, t] = spsolve(ML, b)
        #V[0, t] = 2 * V[1, t] - V[2, t]
       asset_range = np.arange(0, NAS + 1) * dS  + S_min  # Asset price range
    time_steps = np.arange(0, NTS + 1) * dt
    rounded_time_steps = np.round(time_steps, decimals=3)
    df = pd.DataFrame(V, index=asset_range, columns=rounded_time_steps).round(3)
    return df

#controlables 
K =100
sigma = 0.2
r = 0.1
q = 0
expiration = 1
NAS = 200
NTS = 300
type = "call"

option_df = crank_nicholson(type = type, strike = K, sigma = sigma, r = r, expiration = expiration, NAS = NAS, NTS = NTS)

#plotting the options value
plt.figure(figsize=(10, 6))
sns.heatmap(option_df, cmap='YlGnBu', fmt=".3f")
plt.title('Option Values Heatmap')
plt.xlabel('Time Steps')
plt.ylabel('Asset Steps')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plot_df = option_df
X, Y = np.meshgrid(plot_df.columns, plot_df.index)
Z = plot_df.values
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('Time')
ax.set_ylabel('Asset Price')
ax.set_zlabel('Option Value')
ax.set_title('Option Value Surface Plot')
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(10, 196+50)
plt.show()

