"""
Cryptocurrency Market Analyzer - Usage Examples
================================================

This file demonstrates various ways to use the crypto market analyzer.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for demo
import matplotlib.pyplot as plt

from crypto_market_analyzer import CryptoDataFetcher, CryptoAnalyzer, CryptoVisualizer


def example_1_basic_analysis():
    """Example 1: Basic market analysis with real API"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Market Analysis")
    print("=" * 70)
    
    # Initialize fetcher
    fetcher = CryptoDataFetcher()
    
    # Fetch top 30 cryptocurrencies
    print("Fetching data...")
    crypto_data = fetcher.fetch_top_cryptocurrencies(limit=30)
    
    if crypto_data.empty:
        print("⚠️  API unavailable. This example requires internet connection.")
        return
    
    # Create analyzer
    analyzer = CryptoAnalyzer(crypto_data)
    
    # Get top 5 undervalued coins
    undervalued = analyzer.identify_undervalued_coins(top_n=5)
    
    print("\nTop 5 Undervalued Cryptocurrencies:")
    print("-" * 70)
    for idx, (_, row) in enumerate(undervalued.iterrows(), 1):
        print(f"{idx}. {row['symbol'].upper()} ({row['name']})")
        print(f"   Price: ${row['current_price']:.6f}")
        print(f"   Value Score: {row['value_score']:.2f}")
        print()


def example_2_correlation_analysis():
    """Example 2: Detailed correlation analysis"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Correlation Analysis Across Time Periods")
    print("=" * 70)
    
    fetcher = CryptoDataFetcher()
    crypto_data = fetcher.fetch_top_cryptocurrencies(limit=50)
    
    if crypto_data.empty:
        print("⚠️  API unavailable. This example requires internet connection.")
        return
    
    analyzer = CryptoAnalyzer(crypto_data)
    correlation_results = analyzer.analyze_correlations()
    
    print("\nCorrelation Statistics by Time Period:")
    print("-" * 70)
    
    for period in ['1h', '24h', '7d', '30d']:
        if period in correlation_results:
            stats = correlation_results[period]
            print(f"\n{period.upper()} Period:")
            print(f"  Mean Change: {stats['mean_change']:+.2f}%")
            print(f"  Volatility (Std): {stats['std_change']:.2f}%")
            print(f"  Median Change: {stats['median_change']:+.2f}%")
            print(f"  Market Sentiment: {stats['positive_count']} up, "
                  f"{stats['negative_count']} down")
            print(f"  Range: {stats['max_loss']:.2f}% to {stats['max_gain']:.2f}%")


def example_3_bitcoin_comparison():
    """Example 3: Bitcoin vs top altcoins comparison"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Bitcoin Buying Power Analysis")
    print("=" * 70)
    
    fetcher = CryptoDataFetcher()
    
    # Get Bitcoin price
    btc_price = fetcher.fetch_bitcoin_price()
    if btc_price == 0:
        print("⚠️  API unavailable. This example requires internet connection.")
        return
    
    print(f"\nCurrent Bitcoin Price: ${btc_price:,.2f}")
    
    # Fetch data
    crypto_data = fetcher.fetch_top_cryptocurrencies(limit=20)
    
    if crypto_data.empty:
        return
    
    analyzer = CryptoAnalyzer(crypto_data)
    undervalued = analyzer.identify_undervalued_coins(top_n=5)
    
    print("\nWhat 1 BTC Can Buy (Top 5 Undervalued):")
    print("-" * 70)
    
    for idx, (_, row) in enumerate(undervalued.iterrows(), 1):
        units = btc_price / row['current_price']
        print(f"{idx}. {row['symbol'].upper()}: {units:,.2f} units")
        print(f"   Current Price: ${row['current_price']:.6f}")
        print(f"   Rank: #{int(row['market_cap_rank'])}")
        print()


def example_4_custom_visualization():
    """Example 4: Create custom visualization"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Visualization")
    print("=" * 70)
    
    fetcher = CryptoDataFetcher()
    crypto_data = fetcher.fetch_top_cryptocurrencies(limit=40)
    
    if crypto_data.empty:
        print("⚠️  API unavailable. This example requires internet connection.")
        return
    
    analyzer = CryptoAnalyzer(crypto_data)
    visualizer = CryptoVisualizer(crypto_data)
    
    print("Creating comprehensive dashboard...")
    fig = visualizer.create_comprehensive_dashboard(analyzer)
    
    # Save
    filename = 'example_crypto_dashboard.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Dashboard saved as: {filename}")


def example_5_full_report():
    """Example 5: Generate full analysis report"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Complete Analysis Report")
    print("=" * 70)
    
    fetcher = CryptoDataFetcher()
    crypto_data = fetcher.fetch_top_cryptocurrencies(limit=50)
    
    if crypto_data.empty:
        print("⚠️  API unavailable. This example requires internet connection.")
        return
    
    analyzer = CryptoAnalyzer(crypto_data)
    report = analyzer.generate_analysis_report()
    
    # Print the report
    print(report)
    
    # Save to file
    filename = 'crypto_analysis_report.txt'
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Full report saved as: {filename}")


def main():
    """Run all examples"""
    print("=" * 70)
    print("  CRYPTOCURRENCY MARKET ANALYZER - USAGE EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate various ways to use the analyzer.")
    print("Note: Internet connection required for real API calls.")
    print()
    
    examples = [
        ("Basic Analysis", example_1_basic_analysis),
        ("Correlation Analysis", example_2_correlation_analysis),
        ("Bitcoin Buying Power", example_3_bitcoin_comparison),
        ("Custom Visualization", example_4_custom_visualization),
        ("Full Report", example_5_full_report)
    ]
    
    print("Available examples:")
    for idx, (name, _) in enumerate(examples, 1):
        print(f"  {idx}. {name}")
    
    print("\nRunning all examples...")
    print("-" * 70)
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"⚠️  Example '{name}' failed: {e}")
    
    print("\n" + "=" * 70)
    print("  EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
