# Cryptocurrency Market Analyzer

A comprehensive Python tool for fetching real-time cryptocurrency data, performing market analysis, visualizing trends, and identifying undervalued cryptocurrencies for Bitcoin buying opportunities.

## 🎯 Features

### 1. **Real-Time Data Fetching**
- Integration with CoinGecko API (free, no authentication required)
- Fetches top 50+ cryptocurrencies by market cap
- Real-time price updates
- Historical data retrieval

### 2. **Market Analysis**
- **Volatility Calculation**: Analyze price volatility using 24h high/low ranges
- **Momentum Scoring**: Calculate momentum across multiple time periods
- **Value Scoring**: Identify undervalued cryptocurrencies based on:
  - Distance from all-time high (discount indicator)
  - Recent negative momentum (oversold indicator)
  - Volume-to-market-cap ratio (liquidity indicator)

### 3. **Correlation Analysis**
Comprehensive correlation tests across multiple time periods:
- **1 Hour**: Short-term price movements
- **24 Hours**: Daily trends
- **7 Days**: Weekly patterns
- **30 Days**: Monthly trends

Each period includes:
- Mean and median price changes
- Standard deviation
- Number of gainers vs losers
- Maximum gains and losses

### 4. **Visualization Dashboard**
Six comprehensive plots:
1. Top 10 cryptocurrencies by market cap
2. Price changes heatmap across time periods
3. Volatility distribution
4. Volume vs market cap relationship
5. Top undervalued coins
6. Average price changes across time periods

### 5. **Live Price Tracking**
- Real-time price updates with configurable intervals
- Interactive matplotlib plots
- Session-based tracking (5 or 10 minutes)
- Visual change indicators

### 6. **Bitcoin Buying Recommendations**
- Identifies top 10 undervalued cryptocurrencies
- Calculates how many units of each coin 1 BTC can buy
- Provides comprehensive metrics for decision-making

## 🚀 Installation

### Prerequisites
```bash
# Python 3.7 or higher required
python --version

# Install required packages
pip install numpy pandas matplotlib seaborn requests
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage
Simply run the script:
```bash
python crypto_market_analyzer.py
```

### Advanced Usage

#### Import as a Module
```python
from crypto_market_analyzer import CryptoDataFetcher, CryptoAnalyzer, CryptoVisualizer

# Fetch data
fetcher = CryptoDataFetcher()
crypto_data = fetcher.fetch_top_cryptocurrencies(limit=100)

# Analyze
analyzer = CryptoAnalyzer(crypto_data)
undervalued = analyzer.identify_undervalued_coins(top_n=15)
correlation_results = analyzer.analyze_correlations()

# Visualize
visualizer = CryptoVisualizer(crypto_data)
fig = visualizer.create_comprehensive_dashboard(analyzer)
```

#### Fetch Specific Coin Historical Data
```python
fetcher = CryptoDataFetcher()
historical_data = fetcher.fetch_historical_data('bitcoin', days=30)
print(historical_data)
```

#### Generate Analysis Report
```python
analyzer = CryptoAnalyzer(crypto_data)
report = analyzer.generate_analysis_report()
print(report)
```

#### Custom Live Tracking
```python
visualizer = CryptoVisualizer(crypto_data)
visualizer.create_live_updating_plot(
    fetcher=fetcher,
    coin_id='ethereum',
    update_interval=60,  # Update every 60 seconds
    duration_minutes=15  # Track for 15 minutes
)
```

## 📊 Output Examples

### Console Output
```
======================================================================
  🚀 CRYPTOCURRENCY MARKET ANALYZER
======================================================================
Started at: 2024-01-15 14:30:45

======================================================================
  📡 INITIALIZING DATA FETCHER
======================================================================
✓ Data fetcher initialized
✓ Using CoinGecko API (free tier)

======================================================================
  📊 FETCHING REAL-TIME CRYPTOCURRENCY DATA
======================================================================
📊 Fetching top 50 cryptocurrencies...
✓ Fetched 50 cryptocurrencies
✓ Successfully fetched data for 50 cryptocurrencies

======================================================================
  🔍 ANALYZING CRYPTOCURRENCY MARKET
======================================================================
======================================================================
📊 CRYPTOCURRENCY MARKET ANALYSIS REPORT
======================================================================
Analysis Time: 2024-01-15 14:30:50 UTC
Total Cryptocurrencies Analyzed: 50

🌍 MARKET OVERVIEW
----------------------------------------------------------------------
Total Market Cap: $1,850,234,567,890
24h Total Volume: $89,456,123,456

📈 PRICE CHANGE ANALYSIS BY TIME PERIOD
----------------------------------------------------------------------

1H Period:
  Mean Change: 0.15%
  Std Deviation: 1.23%
  Median Change: 0.08%
  Gainers: 28 | Losers: 22
  Max Gain: 4.56% | Max Loss: -3.21%
...
```

### Analysis Report
The tool generates a comprehensive report including:
- Market overview (total market cap, volume)
- Price change analysis for all time periods
- Top 10 undervalued cryptocurrencies with detailed metrics
- Bitcoin buying recommendations

### Visual Dashboard
The dashboard includes:
- Market cap rankings (bar chart)
- Price change heatmap (color-coded by performance)
- Volatility distribution (histogram)
- Volume vs market cap scatter plot
- Undervalued coins ranking
- Time-period correlation chart

## 🔍 Understanding the Analysis

### Value Score Calculation
The value score identifies potentially undervalued cryptocurrencies using:

1. **ATH Distance (40% weight)**: How far the price is from its all-time high
   - Higher distance = bigger discount = higher score

2. **Negative Momentum (30% weight)**: Recent price drops
   - Identifies oversold conditions
   - Potential bounce opportunities

3. **Volume Ratio (30% weight)**: Trading volume relative to market cap
   - Higher ratio = better liquidity
   - Easier to enter/exit positions

### Correlation Analysis
For each time period (1h, 24h, 7d, 30d), the tool calculates:
- **Mean Change**: Average price movement
- **Standard Deviation**: Market volatility
- **Median Change**: Middle value (less affected by outliers)
- **Gainers/Losers**: Number of coins up vs down
- **Max Gain/Loss**: Extreme performers

## 📈 Best Practices

### When to Run
- **Market Open**: US trading hours (9:30 AM - 4:00 PM EST)
- **High Volatility**: During major announcements or events
- **Regular Intervals**: Daily for consistent tracking

### Interpreting Results
1. **Undervalued Coins**: 
   - Focus on top 3-5 recommendations
   - Check market cap rank (prefer top 200)
   - Review 7d and 30d trends

2. **Volatility**:
   - High volatility = higher risk/reward
   - Low volatility = more stable, less opportunity

3. **Volume**:
   - Higher volume = better liquidity
   - Volume > 1% of market cap is healthy

### Risk Management
⚠️ **Important**: This tool provides analysis, not financial advice
- Always do additional research
- Never invest more than you can afford to lose
- Diversify your portfolio
- Consider market conditions and trends

## 🔧 Configuration

### API Rate Limiting
The tool includes built-in rate limiting (1.5s between calls) to respect CoinGecko's free tier limits.

To adjust:
```python
fetcher = CryptoDataFetcher()
fetcher.rate_limit_delay = 2.0  # 2 seconds between calls
```

### Customizing Analysis
```python
# Fetch more/fewer coins
crypto_data = fetcher.fetch_top_cryptocurrencies(limit=100)

# Change undervalued coin count
undervalued = analyzer.identify_undervalued_coins(top_n=20)

# Adjust historical data period
historical = fetcher.fetch_historical_data('bitcoin', days=90)
```

## 🐛 Troubleshooting

### "Failed to fetch data"
- Check internet connection
- Verify CoinGecko API is accessible
- Wait a few minutes and retry (rate limit)

### "ModuleNotFoundError"
```bash
pip install --upgrade numpy pandas matplotlib seaborn requests
```

### Plots not displaying
- Ensure matplotlib backend is properly configured
- Try: `export MPLBACKEND=TkAgg` (Linux/Mac)
- Or install: `pip install pyqt5`

### API Timeout
- Increase timeout in fetcher:
```python
response = self.session.get(endpoint, params=params, timeout=30)
```

## 📚 API Documentation

### CoinGecko API
- **Free Tier**: 50 calls/minute
- **No Authentication**: Required
- **Documentation**: https://www.coingecko.com/en/api/documentation

### Endpoints Used
1. `/coins/markets` - Market data for multiple coins
2. `/simple/price` - Current price for specific coins
3. `/coins/{id}/market_chart` - Historical price data

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional data sources (Binance, Coinbase APIs)
- Machine learning price predictions
- Sentiment analysis integration
- Portfolio tracking features
- Backtesting capabilities

## 📄 License

This project is part of the Game Theory Projects repository and follows the same license.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Cryptocurrency investments are highly volatile and risky. Always conduct your own research and consult with financial advisors before making investment decisions.

## 🔗 Related Projects

Other projects in this repository:
- Tamil Nadu Election Forecasting Model
- Evolutionary Social Dynamics
- Networked Games Control

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Author**: Game Theory Projects Team
