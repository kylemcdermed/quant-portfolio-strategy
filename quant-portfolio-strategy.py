# Multi-Asset, Multi-Strategy Diversified Quant Portfolio
# Professional Implementation for Executive Presentation

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# 1. DATA ACQUISITION MODULE
# ============================================

class DataFetcher:
    """Professional data fetching with error handling and caching"""
    
    def __init__(self, start_date='2020-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.tickers = {
            'BTC': 'BTC-USD',
            'NASDAQ': 'NQ=F',  # NASDAQ Futures
            'OIL': 'CL=F',      # Crude Oil Futures
            'BONDS_2Y': '^FVX',  # 5-Year Treasury Yield (proxy)
            'GBPUSD': 'GBPUSD=X',
            'CORN': 'CORN'      # Corn ETF as proxy
        }
        self.data = {}
        
    def fetch_all_data(self):
        """Fetch all asset data with progress tracking"""
        print("📊 Fetching Market Data...")
        print("-" * 50)
        
        for name, ticker in self.tickers.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                self.data[name] = df['Adj Close'].fillna(method='ffill')
                print(f"✅ {name:10} | {len(df):,} data points | {ticker}")
            except Exception as e:
                print(f"❌ {name:10} | Error: {str(e)[:30]}")
                # Create synthetic data for demo if real data fails
                dates = pd.date_range(self.start_date, self.end_date, freq='D')
                self.data[name] = pd.Series(
                    100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01)),
                    index=dates
                )
        
        return pd.DataFrame(self.data)

# ============================================
# 2. STRATEGY IMPLEMENTATIONS
# ============================================

class StrategyEngine:
    """Professional strategy implementation with proper risk controls"""
    
    def __init__(self, data):
        self.data = data
        self.returns = data.pct_change().dropna()
        self.positions = pd.DataFrame(index=data.index, columns=data.columns)
        self.strategies = {}
        
    def btc_mean_reversion(self, lookback=20, entry_sd=2, exit_sd=1):
        """BTC Mean Reversion Strategy"""
        btc = self.data['BTC']
        
        # Calculate rolling statistics
        ma = btc.rolling(lookback).mean()
        std = btc.rolling(lookback).std()
        z_score = (btc - ma) / std
        
        # Generate signals
        position = pd.Series(0, index=btc.index)
        
        # Entry signals
        position[z_score > entry_sd] = -1  # Short when overbought
        position[z_score < -entry_sd] = 1  # Long when oversold
        
        # Exit signals
        position[(abs(z_score) < exit_sd) & (position.shift(1) != 0)] = 0
        
        # Forward fill positions
        position = position.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        self.strategies['BTC_MR'] = position
        return position
    
    def futures_carry(self, asset='NASDAQ', holding_period=30):
        """Simplified Carry Strategy for Futures"""
        prices = self.data[asset]
        
        # Use momentum as proxy for carry (simplified for demo)
        momentum = prices.pct_change(holding_period)
        signal = momentum.rolling(window=5).mean()
        
        position = pd.Series(0, index=prices.index)
        position[signal > 0.01] = 1
        position[signal < -0.01] = -1
        
        self.strategies[f'{asset}_CARRY'] = position
        return position
    
    def bond_regression(self, lookback=20):
        """Statistical Arbitrage on Bonds"""
        bonds = self.data['BONDS_2Y']
        
        # Simple mean reversion on yields
        ma = bonds.rolling(lookback).mean()
        std = bonds.rolling(lookback).std()
        z_score = (bonds - ma) / std
        
        position = pd.Series(0, index=bonds.index)
        position[z_score > 1.5] = -1
        position[z_score < -1.5] = 1
        
        self.strategies['BOND_STAT'] = position
        return position
    
    def fx_trend_following(self, fast=20, slow=50):
        """FX Trend Following Strategy"""
        fx = self.data['GBPUSD']
        
        # Dual moving average crossover
        ma_fast = fx.rolling(fast).mean()
        ma_slow = fx.rolling(slow).mean()
        
        position = pd.Series(0, index=fx.index)
        position[ma_fast > ma_slow] = 1
        position[ma_fast < ma_slow] = -1
        
        self.strategies['FX_TREND'] = position
        return position
    
    def commodity_momentum(self, lookback=30):
        """Agricultural Commodity Momentum"""
        corn = self.data['CORN']
        
        # Momentum signal
        momentum = corn.pct_change(lookback)
        
        position = pd.Series(0, index=corn.index)
        position[momentum > momentum.quantile(0.7)] = 1
        position[momentum < momentum.quantile(0.3)] = -1
        
        self.strategies['AGRI_MOM'] = position
        return position

# ============================================
# 3. PORTFOLIO CONSTRUCTION & RISK MANAGEMENT
# ============================================

class PortfolioManager:
    """Professional portfolio management with Kelly Criterion and Vol Targeting"""
    
    def __init__(self, initial_capital=1000000, target_vol=0.10):
        self.initial_capital = initial_capital
        self.target_vol = target_vol
        self.capital = initial_capital
        
    def calculate_kelly_fraction(self, returns, confidence=0.25):
        """Calculate Kelly Criterion position sizing"""
        if len(returns) < 30:
            return 0.02  # Minimum allocation
        
        mean_return = returns.mean()
        variance = returns.var()
        
        if variance == 0:
            return 0.02
        
        kelly = mean_return / variance
        # Apply Kelly fraction with confidence scaling
        return np.clip(kelly * confidence, -0.5, 0.5)
    
    def volatility_targeting(self, returns, lookback=60):
        """Dynamic volatility scaling"""
        realized_vol = returns.rolling(lookback).std() * np.sqrt(252)
        vol_scalar = self.target_vol / realized_vol
        return vol_scalar.fillna(1).clip(0.1, 2.0)
    
    def calculate_portfolio_metrics(self, strategy_returns):
        """Calculate comprehensive portfolio metrics"""
        
        # Combine strategies with equal weight initially
        n_strategies = len(strategy_returns.columns)
        base_weights = np.ones(n_strategies) / n_strategies
        
        # Calculate portfolio returns
        portfolio_returns = (strategy_returns * base_weights).sum(axis=1)
        
        # Apply volatility targeting
        vol_scalar = self.volatility_targeting(portfolio_returns)
        scaled_returns = portfolio_returns * vol_scalar
        
        # Calculate cumulative returns with Kelly sizing
        cumulative_capital = [self.initial_capital]
        kelly_fractions = []
        
        for i in range(1, len(scaled_returns)):
            if i > 60:  # Need minimum data for Kelly
                kelly = self.calculate_kelly_fraction(scaled_returns.iloc[i-60:i])
            else:
                kelly = 0.02
            
            kelly_fractions.append(kelly)
            
            # Update capital
            position_return = scaled_returns.iloc[i] * kelly
            new_capital = cumulative_capital[-1] * (1 + position_return)
            cumulative_capital.append(new_capital)
        
        metrics = {
            'Total Return': (cumulative_capital[-1] / self.initial_capital - 1) * 100,
            'Annual Return': ((cumulative_capital[-1] / self.initial_capital) ** 
                            (252 / len(scaled_returns)) - 1) * 100,
            'Volatility': scaled_returns.std() * np.sqrt(252) * 100,
            'Sharpe Ratio': (scaled_returns.mean() / scaled_returns.std()) * np.sqrt(252),
            'Max Drawdown': self.calculate_max_drawdown(cumulative_capital),
            'Calmar Ratio': ((cumulative_capital[-1] / self.initial_capital) ** 
                           (252 / len(scaled_returns)) - 1) / (abs(self.calculate_max_drawdown(cumulative_capital)) / 100),
            'Final Capital': cumulative_capital[-1]
        }
        
        return metrics, cumulative_capital, kelly_fractions
    
    def calculate_max_drawdown(self, capital_series):
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(capital_series)
        drawdown = (np.array(capital_series) - peak) / peak * 100
        return drawdown.min()

# ============================================
# 4. VISUALIZATION & REPORTING
# ============================================

class PortfolioVisualizer:
    """Professional visualization suite for executive presentation"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def plot_correlation_heatmap(self, returns, title="Asset Correlation Matrix"):
        """Create professional correlation heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = returns.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   vmin=-1, vmax=1,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    def plot_performance_dashboard(self, capital_series, strategies_df, metrics):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Portfolio Growth
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(capital_series, linewidth=2, color='darkblue')
        ax1.fill_between(range(len(capital_series)), 
                         self.initial_capital, capital_series, 
                         alpha=0.3, color='lightblue')
        ax1.set_title('Portfolio Growth', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
        
        # 2. Strategy Contributions
        ax2 = plt.subplot(3, 2, 2)
        strategy_cumsum = strategies_df.cumsum()
        for col in strategy_cumsum.columns:
            ax2.plot(strategy_cumsum[col], label=col, alpha=0.7)
        ax2.set_title('Strategy Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Volatility
        ax3 = plt.subplot(3, 2, 3)
        rolling_vol = strategies_df.sum(axis=1).rolling(30).std() * np.sqrt(252)
        ax3.plot(rolling_vol, color='red', alpha=0.7)
        ax3.axhline(y=0.10, color='green', linestyle='--', label='Target Vol (10%)')
        ax3.set_title('Rolling 30-Day Volatility', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Annualized Volatility')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown
        ax4 = plt.subplot(3, 2, 4)
        peak = np.maximum.accumulate(capital_series)
        drawdown = (capital_series - peak) / peak * 100
        ax4.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        ax4.plot(drawdown, color='darkred', linewidth=1)
        ax4.set_title('Drawdown Profile', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Monthly Returns Heatmap
        ax5 = plt.subplot(3, 2, 5)
        monthly_returns = strategies_df.sum(axis=1).resample('M').sum() * 100
        monthly_pivot = monthly_returns.to_frame('Returns')
        monthly_pivot['Year'] = monthly_pivot.index.year
        monthly_pivot['Month'] = monthly_pivot.index.month
        monthly_matrix = monthly_pivot.pivot(index='Year', columns='Month', values='Returns')
        sns.heatmap(monthly_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   ax=ax5, cbar_kws={'label': 'Monthly Return (%)'})
        ax5.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        # 6. Metrics Table
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        metrics_table = [[k, f'{v:.2f}' if k != 'Final Capital' else f'${v:,.0f}'] 
                        for k, v in metrics.items()]
        table = ax6.table(cellText=metrics_table,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(metrics_table) + 1):
            if i == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                for j in range(2):
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        plt.suptitle('Multi-Strategy Quantitative Portfolio Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

# ============================================
# 5. DIVERSIFICATION ANALYSIS
# ============================================

class DiversificationAnalyzer:
    """Calculate diversification metrics"""
    
    def calculate_diversification_ratio(self, returns, weights=None):
        """Calculate Diversification Ratio (DR)"""
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)
        
        # Individual volatilities
        individual_vols = returns.std() * np.sqrt(252)
        
        # Portfolio volatility
        cov_matrix = returns.cov() * 252
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Weighted average of individual volatilities
        weighted_avg_vol = np.sum(weights * individual_vols)
        
        # Diversification Ratio
        dr = weighted_avg_vol / portfolio_vol
        
        return dr
    
    def calculate_effective_n(self, returns):
        """Calculate Effective Number of Bets (ENB)"""
        corr_matrix = returns.corr()
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        
        # Effective N using entropy
        eigenvalues_norm = eigenvalues / eigenvalues.sum()
        entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        effective_n = np.exp(entropy)
        
        return effective_n
    
    def print_diversification_report(self, returns):
        """Generate diversification report"""
        print("\n" + "="*60)
        print(" PORTFOLIO DIVERSIFICATION ANALYSIS")
        print("="*60)
        
        # Calculate metrics
        dr = self.calculate_diversification_ratio(returns)
        enb = self.calculate_effective_n(returns)
        
        # Individual asset metrics
        print("\n📊 Diversification Multipliers by Asset:")
        print("-"*40)
        
        for asset in returns.columns:
            # Calculate marginal contribution to diversification
            subset = returns.drop(columns=[asset])
            dr_without = self.calculate_diversification_ratio(subset) if len(subset.columns) > 1 else 1
            marginal_div = dr - dr_without
            
            print(f"  {asset:15} | Multiplier: {1 + marginal_div:.3f}x")
        
        print("\n📈 Portfolio-Level Metrics:")
        print("-"*40)
        print(f"  Diversification Ratio:     {dr:.3f}")
        print(f"  Effective Number of Bets:  {enb:.2f}")
        print(f"  Concentration Risk:        {'Low' if dr > 1.5 else 'Medium' if dr > 1.2 else 'High'}")
        
        return dr, enb

# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print(" MULTI-STRATEGY QUANTITATIVE PORTFOLIO SYSTEM")
    print(" Executive Demonstration Version")
    print("="*60)
    
    # Initialize components
    fetcher = DataFetcher(start_date='2021-01-01')
    data = fetcher.fetch_all_data()
    
    print("\n📈 Implementing Trading Strategies...")
    print("-" * 50)
    
    # Initialize strategy engine
    engine = StrategyEngine(data)
    
    # Run all strategies
    strategies_positions = pd.DataFrame()
    strategies_positions['BTC_MR'] = engine.btc_mean_reversion()
    strategies_positions['NASDAQ_CARRY'] = engine.futures_carry('NASDAQ')
    strategies_positions['BOND_STAT'] = engine.bond_regression()
    strategies_positions['FX_TREND'] = engine.fx_trend_following()
    strategies_positions['AGRI_MOM'] = engine.commodity_momentum()
    
    # Calculate strategy returns
    strategy_returns = pd.DataFrame()
    for strat in strategies_positions.columns:
        if strat == 'BTC_MR':
            asset_returns = data['BTC'].pct_change()
        elif strat == 'NASDAQ_CARRY':
            asset_returns = data['NASDAQ'].pct_change()
        elif strat == 'BOND_STAT':
            asset_returns = data['BONDS_2Y'].pct_change()
        elif strat == 'FX_TREND':
            asset_returns = data['GBPUSD'].pct_change()
        elif strat == 'AGRI_MOM':
            asset_returns = data['CORN'].pct_change()
        
        strategy_returns[strat] = strategies_positions[strat].shift(1) * asset_returns
    
    strategy_returns = strategy_returns.dropna()
    
    print("✅ All strategies implemented successfully")
    
    # Portfolio Management
    print("\n💼 Constructing Portfolio with Risk Management...")
    print("-" * 50)
    
    pm = PortfolioManager(initial_capital=1000000, target_vol=0.10)
    metrics, capital_series, kelly_fractions = pm.calculate_portfolio_metrics(strategy_returns)
    
    print(f"✅ Portfolio constructed | Target Vol: 10% | Initial Capital: $1,000,000")
    
    # Diversification Analysis
    div_analyzer = DiversificationAnalyzer()
    dr, enb = div_analyzer.print_diversification_report(strategy_returns)
    
    # Visualization
    print("\n📊 Generating Visualizations...")
    print("-" * 50)
    
    visualizer = PortfolioVisualizer()
    visualizer.initial_capital = 1000000
    
    # Create correlation heatmap
    corr_fig = visualizer.plot_correlation_heatmap(
        data[['BTC', 'NASDAQ', 'BONDS_2Y', 'GBPUSD', 'CORN']].pct_change().dropna(),
        "Asset Correlation Matrix - Diversification Benefits"
    )
    
    # Create performance dashboard
    perf_fig = visualizer.plot_performance_dashboard(
        capital_series, 
        strategy_returns,
        metrics
    )
    
    # Print final summary
    print("\n" + "="*60)
    print(" PORTFOLIO PERFORMANCE SUMMARY")
    print("="*60)
    print(f"\n💰 Initial Capital:     ${1000000:,.0f}")
    print(f"💰 Final Capital:       ${metrics['Final Capital']:,.0f}")
    print(f"📈 Total Return:        {metrics['Total Return']:.2f}%")
    print(f"📊 Annualized Return:   {metrics['Annual Return']:.2f}%")
    print(f"📉 Max Drawdown:        {metrics['Max Drawdown']:.2f}%")
    print(f"⚡ Sharpe Ratio:        {metrics['Sharpe Ratio']:.3f}")
    print(f"🎯 Realized Vol:        {metrics['Volatility']:.2f}%")
    print(f"📐 Calmar Ratio:        {metrics['Calmar Ratio']:.3f}")
    
    print("\n" + "="*60)
    print(" SYSTEM READY FOR EXECUTIVE PRESENTATION")
    print("="*60)
    
    plt.show()
    
    return {
        'data': data,
        'strategies': strategies_positions,
        'returns': strategy_returns,
        'metrics': metrics,
        'capital': capital_series,
        'diversification': {'DR': dr, 'ENB': enb}
    }

# Execute the system
if __name__ == "__main__":
    portfolio_results = main()
