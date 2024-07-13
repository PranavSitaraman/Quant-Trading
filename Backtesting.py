import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def stats(portfolio, signals, start_date, end_date, capital):

    stats = pd.DataFrame([[portfolio['total'].iloc[-1]/capital - 1]], columns=['portfolio return'])
    growth_rate = (portfolio['total'].iloc[-1]/capital) ** (1/len(signals)) - 1
    std = np.sqrt((((portfolio['return']-growth_rate)**2).sum())/len(signals))

    # Use S&P500 as benchmark
    benchmark = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    benchmark_return = benchmark['Close'].iloc[-1]/benchmark['Open'].iloc[0] - 1.0
    benchmark_rate = (benchmark_return+1) ** (1/len(signals)) - 1

    stats['benchmark return'] = benchmark_return
    stats['growth rate'] = growth_rate
    stats['sharpe ratio'] = (growth_rate-benchmark_rate)/std
    stats['maximum drawdown'] = 0.0
    for i in range(1, len(portfolio['total'])):
        dd = portfolio['total'].iloc[i]/portfolio['total'].iloc[:i].max() - 1
        stats.at[stats.index[0], 'maximum drawdown'] = min(stats['maximum drawdown'].iloc[0], dd)
    stats['calmar ratio'] = growth_rate/stats['maximum drawdown']
    trade_num = signals['signals'].loc[signals['signals'] != 0].count()
    stats['avg trade length'] = signals['signals'].loc[signals['positions'] != 0].count()/trade_num
    stats['profit per trade'] = (portfolio['total'].iloc[-1] - capital)/trade_num

    return stats

def portfolio(signals, capital, title):
    portfolio = pd.DataFrame()
    portfolio['holdings'] = signals['positions'] * signals['asset']

    if min(portfolio['holdings']) < 0:
        # Position size to maintain 50% short position margin requirement
        positions = -2 * capital/min(portfolio['holdings'])
    else:
        # Position size to avoid buying long on margin
        positions = capital/max(portfolio['holdings'])

    portfolio['holdings'] *= positions
    portfolio['cash'] = capital - (signals['signals'] * signals['asset'] * positions).cumsum()
    portfolio['total'] = portfolio['holdings'] + portfolio['cash']
    portfolio['return'] = portfolio['total'].pct_change()
    
    plt.figure(figsize=(11,7))
    plt.plot(portfolio['total'], alpha=0.8, label="Total Assets")
    plt.xlabel('Date', labelpad=15)
    plt.ylabel('Value ($)', labelpad=15)
    plt.legend(loc="upper left")
    plt.title(title, pad=15)
    plt.grid(True)
    plt.show()

    return portfolio