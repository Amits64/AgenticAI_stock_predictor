class StrategyBacktester:
    def __init__(self, df, initial_capital=10000, position_size=1, slippage=0.0001):
        """
        Initialize the backtester with the necessary parameters.
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.slippage = slippage


def backtest_strategy(df):
    backtester = StrategyBacktester(df)
    results, sharpe_ratio, max_drawdown = backtester.backtest()
    plot_url = backtester.plot_results()
    return results, sharpe_ratio, max_drawdown, plot_url
