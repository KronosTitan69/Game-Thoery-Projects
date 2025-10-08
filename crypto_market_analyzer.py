"""
Cryptocurrency Market Analyzer
===============================

A comprehensive tool for fetching real-time cryptocurrency data, analyzing price trends,
visualizing market movements, and predicting undervalued currencies for Bitcoin buying.

Features:
1. Real-time data fetching from CoinGecko API
2. Price trend analysis and volatility calculations
3. Live updating visualization dashboard
4. Correlation analysis across multiple time periods (1h, 24h, 7d, 30d)
5. Undervalued currency detection for Bitcoin buying opportunities
6. Interactive plots with matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import json
warnings.filterwarnings('ignore')

try:
    import requests
except ImportError:
    print("⚠️  requests library not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'requests'])
    import requests

# Configure visualization style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)


class CryptoDataFetcher:
    """Handles API integration for fetching real-time cryptocurrency data"""
    
    def __init__(self, base_url: str = "https://api.coingecko.com/api/v3"):
        self.base_url = base_url
        self.session = requests.Session()
        self.rate_limit_delay = 1.5  # seconds between API calls
        
    def fetch_top_cryptocurrencies(self, limit: int = 50) -> pd.DataFrame:
        """
        Fetch top cryptocurrencies by market cap
        
        Args:
            limit: Number of top cryptocurrencies to fetch
            
        Returns:
            DataFrame with cryptocurrency data
        """
        print(f"📊 Fetching top {limit} cryptocurrencies...")
        
        endpoint = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': limit,
            'page': 1,
            'sparkline': False,
            'price_change_percentage': '1h,24h,7d,30d'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data)
            
            # Extract relevant columns
            columns_to_keep = [
                'id', 'symbol', 'name', 'current_price', 'market_cap', 
                'market_cap_rank', 'total_volume', 'high_24h', 'low_24h',
                'price_change_percentage_1h_in_currency',
                'price_change_percentage_24h_in_currency',
                'price_change_percentage_7d_in_currency',
                'price_change_percentage_30d_in_currency',
                'circulating_supply', 'total_supply', 'ath', 'atl'
            ]
            
            available_cols = [col for col in columns_to_keep if col in df.columns]
            df = df[available_cols]
            
            # Rename percentage columns for clarity
            df = df.rename(columns={
                'price_change_percentage_1h_in_currency': 'change_1h',
                'price_change_percentage_24h_in_currency': 'change_24h',
                'price_change_percentage_7d_in_currency': 'change_7d',
                'price_change_percentage_30d_in_currency': 'change_30d'
            })
            
            print(f"✓ Fetched {len(df)} cryptocurrencies")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    def fetch_bitcoin_price(self) -> float:
        """
        Fetch current Bitcoin price in USD
        
        Returns:
            Current BTC price
        """
        endpoint = f"{self.base_url}/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data['bitcoin']['usd']
        except Exception as e:
            print(f"❌ Error fetching Bitcoin price: {e}")
            return 0.0
    
    def fetch_historical_data(self, coin_id: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch historical price data for a cryptocurrency
        
        Args:
            coin_id: CoinGecko coin identifier
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical price data
        """
        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily' if days > 1 else 'hourly'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            time.sleep(self.rate_limit_delay)  # Rate limiting
            return df
            
        except Exception as e:
            print(f"❌ Error fetching historical data for {coin_id}: {e}")
            return pd.DataFrame()


class CryptoAnalyzer:
    """Analyzes cryptocurrency data for trends, volatility, and correlations"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.analysis_results = {}
        
    def calculate_volatility(self) -> pd.Series:
        """
        Calculate price volatility using 24h high/low range
        
        Returns:
            Series with volatility percentages
        """
        volatility = ((self.data['high_24h'] - self.data['low_24h']) / 
                     self.data['current_price'] * 100)
        return volatility.fillna(0)
    
    def calculate_momentum_score(self) -> pd.Series:
        """
        Calculate momentum score based on price changes across time periods
        
        Returns:
            Series with momentum scores
        """
        # Weight recent changes more heavily
        weights = {'change_1h': 0.1, 'change_24h': 0.3, 'change_7d': 0.4, 'change_30d': 0.2}
        
        momentum = pd.Series(0.0, index=self.data.index)
        for col, weight in weights.items():
            if col in self.data.columns:
                momentum += self.data[col].fillna(0) * weight
                
        return momentum
    
    def calculate_value_score(self) -> pd.Series:
        """
        Calculate value score to identify undervalued cryptocurrencies
        
        Considers:
        - Distance from all-time high (discount)
        - Recent negative momentum (potential bounce)
        - Volume to market cap ratio (liquidity)
        
        Returns:
            Series with value scores (higher = more undervalued)
        """
        value_scores = pd.Series(0.0, index=self.data.index)
        
        # Distance from ATH (discount indicator)
        if 'ath' in self.data.columns and 'current_price' in self.data.columns:
            ath_distance = (self.data['ath'] - self.data['current_price']) / self.data['ath'] * 100
            value_scores += ath_distance.fillna(0) * 0.4
        
        # Recent negative momentum (oversold indicator)
        if 'change_7d' in self.data.columns:
            negative_momentum = -self.data['change_7d'].clip(upper=0).fillna(0) * 0.3
            value_scores += negative_momentum
        
        # Volume/Market Cap ratio (liquidity indicator)
        if 'total_volume' in self.data.columns and 'market_cap' in self.data.columns:
            volume_ratio = (self.data['total_volume'] / self.data['market_cap'] * 100).fillna(0)
            value_scores += volume_ratio * 0.3
            
        return value_scores
    
    def identify_undervalued_coins(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify potentially undervalued cryptocurrencies for Bitcoin buying
        
        Args:
            top_n: Number of top undervalued coins to return
            
        Returns:
            DataFrame with undervalued coins and analysis
        """
        # Calculate all scores
        self.data['volatility'] = self.calculate_volatility()
        self.data['momentum_score'] = self.calculate_momentum_score()
        self.data['value_score'] = self.calculate_value_score()
        
        # Filter: exclude stablecoins and very low volume coins
        filtered = self.data[
            (self.data['total_volume'] > self.data['market_cap'] * 0.01) &  # Min 1% volume
            (self.data['market_cap_rank'] <= 200)  # Top 200 by market cap
        ].copy()
        
        # Sort by value score
        undervalued = filtered.nlargest(top_n, 'value_score')
        
        # Select relevant columns for display
        result_cols = ['symbol', 'name', 'current_price', 'market_cap_rank', 
                      'change_7d', 'change_30d', 'volatility', 'value_score']
        
        return undervalued[[col for col in result_cols if col in undervalued.columns]]
    
    def calculate_correlation_matrix(self, time_period: str = '7d') -> pd.DataFrame:
        """
        Calculate correlation matrix for cryptocurrencies based on price changes
        
        Args:
            time_period: Time period column to use for correlation
            
        Returns:
            Correlation matrix
        """
        col_name = f'change_{time_period}'
        if col_name not in self.data.columns:
            print(f"⚠️  Column {col_name} not available")
            return pd.DataFrame()
        
        # Create a pivot table of price changes
        price_changes = self.data[['symbol', col_name]].dropna()
        
        return price_changes
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyze correlations across multiple time periods
        
        Returns:
            Dictionary with correlation analyses for different time periods
        """
        time_periods = ['1h', '24h', '7d', '30d']
        results = {}
        
        for period in time_periods:
            col_name = f'change_{period}'
            if col_name in self.data.columns:
                changes = self.data[col_name].dropna()
                
                results[period] = {
                    'mean_change': changes.mean(),
                    'std_change': changes.std(),
                    'positive_count': (changes > 0).sum(),
                    'negative_count': (changes < 0).sum(),
                    'max_gain': changes.max(),
                    'max_loss': changes.min(),
                    'median_change': changes.median()
                }
        
        return results
    
    def generate_analysis_report(self) -> str:
        """
        Generate comprehensive analysis report
        
        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 70)
        report.append("📊 CRYPTOCURRENCY MARKET ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"Total Cryptocurrencies Analyzed: {len(self.data)}")
        report.append("")
        
        # Market overview
        report.append("🌍 MARKET OVERVIEW")
        report.append("-" * 70)
        if 'market_cap' in self.data.columns:
            total_market_cap = self.data['market_cap'].sum()
            report.append(f"Total Market Cap: ${total_market_cap:,.0f}")
        
        if 'total_volume' in self.data.columns:
            total_volume = self.data['total_volume'].sum()
            report.append(f"24h Total Volume: ${total_volume:,.0f}")
        report.append("")
        
        # Correlation analysis
        correlation_results = self.analyze_correlations()
        report.append("📈 PRICE CHANGE ANALYSIS BY TIME PERIOD")
        report.append("-" * 70)
        
        for period, stats in correlation_results.items():
            report.append(f"\n{period.upper()} Period:")
            report.append(f"  Mean Change: {stats['mean_change']:.2f}%")
            report.append(f"  Std Deviation: {stats['std_change']:.2f}%")
            report.append(f"  Median Change: {stats['median_change']:.2f}%")
            report.append(f"  Gainers: {stats['positive_count']} | Losers: {stats['negative_count']}")
            report.append(f"  Max Gain: {stats['max_gain']:.2f}% | Max Loss: {stats['max_loss']:.2f}%")
        
        report.append("")
        
        # Undervalued coins
        report.append("💎 TOP UNDERVALUED CRYPTOCURRENCIES FOR BTC BUYING")
        report.append("-" * 70)
        undervalued = self.identify_undervalued_coins(top_n=10)
        
        for idx, row in undervalued.iterrows():
            report.append(f"\n{row['symbol'].upper()} - {row['name']}")
            report.append(f"  Current Price: ${row['current_price']:.6f}")
            report.append(f"  Market Cap Rank: #{int(row['market_cap_rank'])}")
            if 'change_7d' in row:
                report.append(f"  7d Change: {row['change_7d']:.2f}%")
            if 'change_30d' in row:
                report.append(f"  30d Change: {row['change_30d']:.2f}%")
            report.append(f"  Value Score: {row['value_score']:.2f}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


class CryptoVisualizer:
    """Creates visualizations for cryptocurrency market data"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.fig = None
        
    def create_comprehensive_dashboard(self, analyzer: CryptoAnalyzer):
        """
        Create comprehensive visualization dashboard
        
        Args:
            analyzer: CryptoAnalyzer instance with analyzed data
        """
        self.fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        self.fig.suptitle('Cryptocurrency Market Analysis Dashboard', 
                         fontsize=16, fontweight='bold')
        
        # 1. Top 10 by Market Cap
        self._plot_top_by_market_cap(axes[0, 0])
        
        # 2. Price Changes Across Time Periods
        self._plot_price_changes_heatmap(axes[0, 1])
        
        # 3. Volatility Analysis
        self._plot_volatility_distribution(axes[1, 0])
        
        # 4. Volume vs Market Cap
        self._plot_volume_vs_market_cap(axes[1, 1])
        
        # 5. Undervalued Coins
        self._plot_undervalued_coins(axes[2, 0], analyzer)
        
        # 6. Correlation Analysis
        self._plot_correlation_analysis(axes[2, 1])
        
        plt.tight_layout()
        return self.fig
    
    def _plot_top_by_market_cap(self, ax):
        """Plot top 10 cryptocurrencies by market cap"""
        top10 = self.data.nlargest(10, 'market_cap')
        
        colors = ['#f7931a' if s.lower() == 'btc' else '#627eea' if s.lower() == 'eth' 
                 else '#00b300' for s in top10['symbol']]
        
        ax.barh(range(len(top10)), top10['market_cap'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10['symbol'].str.upper())
        ax.set_xlabel('Market Cap (USD)', fontweight='bold')
        ax.set_title('Top 10 Cryptocurrencies by Market Cap', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
    def _plot_price_changes_heatmap(self, ax):
        """Plot heatmap of price changes across time periods"""
        top15 = self.data.nlargest(15, 'market_cap')
        
        change_cols = ['change_1h', 'change_24h', 'change_7d', 'change_30d']
        available_cols = [col for col in change_cols if col in top15.columns]
        
        if available_cols:
            change_data = top15[available_cols].fillna(0)
            change_data.index = top15['symbol'].str.upper()
            
            sns.heatmap(change_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, ax=ax, cbar_kws={'label': 'Change %'})
            ax.set_title('Price Changes Across Time Periods (Top 15)', fontweight='bold')
            ax.set_xlabel('Time Period', fontweight='bold')
            ax.set_ylabel('Cryptocurrency', fontweight='bold')
    
    def _plot_volatility_distribution(self, ax):
        """Plot volatility distribution"""
        if 'volatility' not in self.data.columns:
            self.data['volatility'] = ((self.data['high_24h'] - self.data['low_24h']) / 
                                       self.data['current_price'] * 100).fillna(0)
        
        volatility = self.data['volatility'].dropna()
        
        ax.hist(volatility, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(volatility.mean(), color='red', linestyle='--', 
                  label=f'Mean: {volatility.mean():.2f}%', linewidth=2)
        ax.axvline(volatility.median(), color='blue', linestyle='--', 
                  label=f'Median: {volatility.median():.2f}%', linewidth=2)
        ax.set_xlabel('Volatility (%)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('24h Price Volatility Distribution', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _plot_volume_vs_market_cap(self, ax):
        """Plot volume vs market cap relationship"""
        if 'total_volume' in self.data.columns and 'market_cap' in self.data.columns:
            # Filter out extreme outliers for better visualization
            data_filtered = self.data[
                (self.data['market_cap'] > 0) & 
                (self.data['total_volume'] > 0)
            ].copy()
            
            # Color by market cap rank
            scatter = ax.scatter(data_filtered['market_cap'], 
                               data_filtered['total_volume'],
                               c=data_filtered['market_cap_rank'],
                               cmap='viridis_r', alpha=0.6, s=50)
            
            ax.set_xlabel('Market Cap (USD)', fontweight='bold')
            ax.set_ylabel('24h Volume (USD)', fontweight='bold')
            ax.set_title('Volume vs Market Cap', fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Market Cap Rank', fontweight='bold')
            
            # Annotate Bitcoin and Ethereum
            for idx, row in data_filtered.iterrows():
                if row['symbol'].lower() in ['btc', 'eth']:
                    ax.annotate(row['symbol'].upper(), 
                              (row['market_cap'], row['total_volume']),
                              xytext=(10, 5), textcoords='offset points',
                              fontweight='bold', fontsize=10)
    
    def _plot_undervalued_coins(self, ax, analyzer: CryptoAnalyzer):
        """Plot top undervalued coins"""
        undervalued = analyzer.identify_undervalued_coins(top_n=10)
        
        if not undervalued.empty:
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(undervalued)))
            
            ax.barh(range(len(undervalued)), undervalued['value_score'], 
                   color=colors, alpha=0.8)
            ax.set_yticks(range(len(undervalued)))
            ax.set_yticklabels([f"{row['symbol'].upper()}\n${row['current_price']:.4f}" 
                               for _, row in undervalued.iterrows()],
                              fontsize=9)
            ax.set_xlabel('Value Score', fontweight='bold')
            ax.set_title('Top 10 Undervalued Cryptocurrencies', fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
    
    def _plot_correlation_analysis(self, ax):
        """Plot correlation across time periods"""
        time_periods = ['1h', '24h', '7d', '30d']
        available_periods = []
        mean_changes = []
        
        for period in time_periods:
            col_name = f'change_{period}'
            if col_name in self.data.columns:
                available_periods.append(period.upper())
                mean_changes.append(self.data[col_name].mean())
        
        if available_periods:
            colors = ['green' if x > 0 else 'red' for x in mean_changes]
            
            ax.bar(available_periods, mean_changes, color=colors, alpha=0.7)
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xlabel('Time Period', fontweight='bold')
            ax.set_ylabel('Mean Price Change (%)', fontweight='bold')
            ax.set_title('Average Price Changes Across Time Periods', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (period, value) in enumerate(zip(available_periods, mean_changes)):
                ax.text(i, value, f'{value:.2f}%', ha='center', 
                       va='bottom' if value > 0 else 'top', fontweight='bold')
    
    def create_live_updating_plot(self, fetcher: CryptoDataFetcher, 
                                 coin_id: str = 'bitcoin', 
                                 update_interval: int = 60,
                                 duration_minutes: int = 5):
        """
        Create a live updating price plot
        
        Args:
            fetcher: CryptoDataFetcher instance
            coin_id: Coin identifier to track
            update_interval: Seconds between updates
            duration_minutes: How long to run the live plot
        """
        print(f"📈 Starting live price tracker for {coin_id.upper()}...")
        print(f"   Updates every {update_interval}s for {duration_minutes} minutes")
        print("   Close the plot window to stop tracking.")
        
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(12, 6))
        
        timestamps = []
        prices = []
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # Fetch current price
                current_price = fetcher.fetch_bitcoin_price()
                current_time = datetime.now()
                
                if current_price > 0:
                    timestamps.append(current_time)
                    prices.append(current_price)
                    
                    # Clear and redraw
                    ax.clear()
                    ax.plot(timestamps, prices, 'o-', linewidth=2, markersize=6, 
                           color='#f7931a', label=f'{coin_id.upper()} Price')
                    
                    ax.set_xlabel('Time', fontweight='bold')
                    ax.set_ylabel('Price (USD)', fontweight='bold')
                    ax.set_title(f'{coin_id.upper()} Live Price Tracker', 
                               fontweight='bold', fontsize=14)
                    ax.grid(alpha=0.3)
                    ax.legend()
                    
                    # Format x-axis
                    plt.xticks(rotation=45)
                    fig.tight_layout()
                    
                    # Display current price
                    if len(prices) > 1:
                        change = prices[-1] - prices[0]
                        change_pct = (change / prices[0]) * 100
                        ax.text(0.02, 0.98, 
                               f'Current: ${current_price:,.2f}\n'
                               f'Change: ${change:+,.2f} ({change_pct:+.2f}%)',
                               transform=ax.transAxes, fontsize=11,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    plt.draw()
                    plt.pause(0.1)
                
                # Wait for next update
                time.sleep(update_interval)
                
            except KeyboardInterrupt:
                print("\n⏹️  Live tracking stopped by user")
                break
            except Exception as e:
                print(f"⚠️  Error in live update: {e}")
                time.sleep(update_interval)
        
        plt.ioff()
        print("✓ Live tracking complete")


def print_header(text: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main():
    """Main execution function with comprehensive cryptocurrency analysis"""
    
    print_header("🚀 CRYPTOCURRENCY MARKET ANALYZER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Initialize data fetcher
    print_header("📡 INITIALIZING DATA FETCHER")
    fetcher = CryptoDataFetcher()
    print("✓ Data fetcher initialized")
    print("✓ Using CoinGecko API (free tier)")
    
    # Step 2: Fetch cryptocurrency data
    print_header("📊 FETCHING REAL-TIME CRYPTOCURRENCY DATA")
    crypto_data = fetcher.fetch_top_cryptocurrencies(limit=50)
    
    if crypto_data.empty:
        print("❌ Failed to fetch cryptocurrency data. Exiting.")
        return
    
    print(f"✓ Successfully fetched data for {len(crypto_data)} cryptocurrencies")
    
    # Step 3: Analyze data
    print_header("🔍 ANALYZING CRYPTOCURRENCY MARKET")
    analyzer = CryptoAnalyzer(crypto_data)
    
    # Generate and display comprehensive report
    report = analyzer.generate_analysis_report()
    print(report)
    
    # Step 4: Create visualizations
    print_header("📈 GENERATING VISUALIZATIONS")
    visualizer = CryptoVisualizer(crypto_data)
    
    print("Creating comprehensive dashboard...")
    fig = visualizer.create_comprehensive_dashboard(analyzer)
    
    # Save the dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'crypto_analysis_dashboard_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Dashboard saved as: {filename}")
    
    # Display the plot
    plt.show(block=False)
    plt.pause(2)
    
    # Step 5: Correlation tests
    print_header("📊 CORRELATION ANALYSIS ACROSS TIME PERIODS")
    correlation_results = analyzer.analyze_correlations()
    
    print("\nDetailed Correlation Statistics:")
    print("-" * 70)
    for period, stats in correlation_results.items():
        print(f"\n{period.upper()} Period Analysis:")
        print(f"  • Mean Change: {stats['mean_change']:.2f}%")
        print(f"  • Standard Deviation: {stats['std_change']:.2f}%")
        print(f"  • Median Change: {stats['median_change']:.2f}%")
        print(f"  • Gainers vs Losers: {stats['positive_count']} vs {stats['negative_count']}")
        print(f"  • Max Gain: {stats['max_gain']:.2f}%")
        print(f"  • Max Loss: {stats['max_loss']:.2f}%")
    
    # Step 6: Bitcoin buying recommendations
    print_header("💰 BITCOIN BUYING RECOMMENDATIONS")
    print("\nTop Undervalued Cryptocurrencies for BTC Purchase:")
    print("-" * 70)
    
    undervalued = analyzer.identify_undervalued_coins(top_n=10)
    
    btc_price = fetcher.fetch_bitcoin_price()
    if btc_price > 0:
        print(f"\n🪙 Current Bitcoin Price: ${btc_price:,.2f}")
        print(f"\nRecommended Strategy: Consider diversifying BTC holdings into:")
        print()
        
        for rank, (idx, row) in enumerate(undervalued.iterrows(), 1):
            print(f"{rank}. {row['symbol'].upper()} - {row['name']}")
            print(f"   Price: ${row['current_price']:.6f} | Rank: #{int(row['market_cap_rank'])}")
            print(f"   7d Change: {row.get('change_7d', 0):.2f}% | "
                  f"Value Score: {row['value_score']:.2f}")
            
            # Calculate how much BTC could buy
            if row['current_price'] > 0:
                units_per_btc = btc_price / row['current_price']
                print(f"   → 1 BTC = {units_per_btc:,.2f} {row['symbol'].upper()}")
            print()
    
    # Step 7: Offer live tracking
    print_header("📺 LIVE PRICE TRACKING")
    print("\nOptions:")
    print("  1. Skip live tracking")
    print("  2. Start 5-minute live Bitcoin price tracker")
    print("  3. Start 10-minute live Bitcoin price tracker")
    
    choice = input("\nEnter your choice (1-3) [default: 1]: ").strip()
    
    if choice == '2':
        visualizer.create_live_updating_plot(fetcher, 'bitcoin', 
                                            update_interval=30, duration_minutes=5)
    elif choice == '3':
        visualizer.create_live_updating_plot(fetcher, 'bitcoin', 
                                            update_interval=30, duration_minutes=10)
    else:
        print("Skipping live tracking...")
    
    # Final summary
    print_header("✅ ANALYSIS COMPLETE")
    print("Summary:")
    print(f"  • Analyzed {len(crypto_data)} cryptocurrencies")
    print(f"  • Generated comprehensive visualizations")
    print(f"  • Identified {len(undervalued)} undervalued opportunities")
    print(f"  • Performed correlation analysis across 4 time periods")
    print(f"\n📁 Dashboard saved as: {filename}")
    print("\n🔄 To run again, execute: python crypto_market_analyzer.py")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
