# main.py - Complete YouTube Subscriber Growth Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Set style for better visuals
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("YOUTUBE SUBSCRIBER & VIEW GROWTH ANALYSIS")
print("="*60)

# ============================================
# STEP 1: GENERATE SAMPLE DATA
# ============================================
print("\n📊 STEP 1: Generating sample data...")

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Create growth patterns
days = np.arange(len(dates))
base_subs = 1000
weekly_pattern = 50 * np.sin(2 * np.pi * days / 7)
growth_trend = 25 * days / 365
viral_spikes = np.random.choice([0, 0, 0, 0, 100, 200], size=len(dates), p=[0.95, 0.02, 0.01, 0.01, 0.005, 0.005])

# Generate daily new subscribers
new_subs = np.maximum(0, 15 + weekly_pattern + growth_trend + viral_spikes + 
                       np.random.normal(0, 8, len(dates)))
new_subs = new_subs.astype(int)

# Calculate cumulative subscribers
total_subs = base_subs + np.cumsum(new_subs)

# Generate view data
views_per_sub = 0.5 + 0.1 * np.sin(2 * np.pi * days / 30)
total_views = (total_subs * views_per_sub + 
               np.random.normal(0, total_subs * 0.1, len(dates))).astype(int)
total_views = np.maximum(0, total_views)

# Generate engagement metrics
likes = (total_views * 0.04 + np.random.normal(0, total_views * 0.005, len(dates))).astype(int)
comments = (total_views * 0.005 + np.random.normal(0, total_views * 0.001, len(dates))).astype(int)
shares = (total_views * 0.002 + np.random.normal(0, total_views * 0.0005, len(dates))).astype(int)

# Content uploads (2-3 videos per week)
upload_days = np.random.choice([0, 1], size=len(dates), p=[0.6, 0.4])

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'new_subscribers': new_subs,
    'total_subscribers': total_subs,
    'total_views': total_views,
    'likes': likes,
    'comments': comments,
    'shares': shares,
    'video_uploaded': upload_days
})

# Add derived metrics
df['views_per_subscriber'] = df['total_views'] / df['total_subscribers']
df['engagement_rate'] = (df['likes'] + df['comments'] + df['shares']) / df['total_views']
df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month_name()
df['week'] = df['date'].dt.isocalendar().week

print(f"✅ Data generated: {len(df)} days of data")
print(f"   Period: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"   Subscribers: {df['total_subscribers'].min():,} → {df['total_subscribers'].max():,}")

# Save to CSV
df.to_csv('youtube_data.csv', index=False)
print("✅ Data saved to 'youtube_data.csv'")

# ============================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================
print("\n📈 STEP 2: Exploratory Data Analysis...")

# Basic statistics
print("\nBasic Statistics:")
print(df[['new_subscribers', 'total_subscribers', 'total_views', 'engagement_rate']].describe())

# Growth summary
start_subs = df.iloc[0]['total_subscribers']
end_subs = df.iloc[-1]['total_subscribers']
growth_rate = ((end_subs - start_subs) / start_subs) * 100
print(f"\nOverall Growth: {start_subs:,} → {end_subs:,} ({growth_rate:.1f}% increase)")

# Best and worst days
best_day = df.loc[df['new_subscribers'].idxmax()]
worst_day = df.loc[df['new_subscribers'].idxmin()]
print(f"⭐ Best day: {best_day['date'].date()} - {best_day['new_subscribers']} new subscribers")
print(f"💀 Worst day: {worst_day['date'].date()} - {worst_day['new_subscribers']} new subscribers")

# Create EDA plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].plot(df['date'], df['total_subscribers'], linewidth=2, color='#2E86AB')
axes[0,0].set_title('Total Subscribers Growth Over Time', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Total Subscribers')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(df['date'], df['new_subscribers'], linewidth=1, color='#A23B72', alpha=0.7)
axes[0,1].set_title('Daily New Subscribers', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Date')
axes[0,1].set_ylabel('New Subscribers')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].scatter(df['total_subscribers'], df['total_views'], alpha=0.5, s=10, color='#F18F01')
axes[1,0].set_title('Views vs Subscribers Correlation', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Total Subscribers')
axes[1,0].set_ylabel('Total Views')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].hist(df['engagement_rate'], bins=30, edgecolor='black', alpha=0.7, color='#C73E1D')
axes[1,1].set_title('Engagement Rate Distribution', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Engagement Rate')
axes[1,1].set_ylabel('Frequency')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('1_eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plots saved as '1_eda_plots.png'")

# ============================================
# STEP 3: GROWTH PATTERN ANALYSIS
# ============================================
print("\n📅 STEP 3: Growth Pattern Analysis...")

# Day of week analysis
dow_analysis = df.groupby('day_of_week').agg({
    'new_subscribers': ['mean', 'median', 'std'],
    'total_views': 'mean',
    'engagement_rate': 'mean'
}).round(2)

weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_analysis = dow_analysis.reindex(weekdays)

print("\nAverage New Subscribers by Day of Week:")
print(dow_analysis[('new_subscribers', 'mean')])

# Monthly analysis
monthly = df.groupby('month').agg({
    'new_subscribers': 'sum',
    'total_views': 'sum',
    'video_uploaded': 'sum'
}).round(2)

months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
monthly = monthly.reindex(months_order)

print("\nMonthly Performance:")
print(monthly)

# Upload day effect
upload_effect = df.groupby('video_uploaded').agg({
    'new_subscribers': 'mean',
    'total_views': 'mean',
    'engagement_rate': 'mean'
})
upload_effect.index = ['No Upload', 'Upload Day']
print("\nEffect of Uploading Videos:")
print(upload_effect)

# Growth patterns visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

dow_analysis[('new_subscribers', 'mean')].plot(kind='bar', ax=axes[0,0], color='skyblue', edgecolor='black')
axes[0,0].set_title('Average New Subscribers by Day of Week', fontsize=12, fontweight='bold')
axes[0,0].set_xlabel('Day')
axes[0,0].set_ylabel('Avg New Subscribers')
axes[0,0].tick_params(axis='x', rotation=45)

monthly['new_subscribers'].plot(kind='bar', ax=axes[0,1], color='lightcoral', edgecolor='black')
axes[0,1].set_title('Total New Subscribers by Month', fontsize=12, fontweight='bold')
axes[0,1].set_xlabel('Month')
axes[0,1].set_ylabel('Total New Subscribers')
axes[0,1].tick_params(axis='x', rotation=45)

upload_effect['new_subscribers'].plot(kind='bar', ax=axes[1,0], color=['gray', 'green'], edgecolor='black')
axes[1,0].set_title('Upload Days vs Non-Upload Days', fontsize=12, fontweight='bold')
axes[1,0].set_xlabel('Day Type')
axes[1,0].set_ylabel('Avg New Subscribers')

df['rolling_avg_growth'] = df['new_subscribers'].rolling(window=30).mean()
df['rolling_views'] = df['total_views'].rolling(window=30).mean()
axes[1,1].plot(df['date'], df['rolling_avg_growth'], label='30-day avg new subs', linewidth=2)
axes[1,1].plot(df['date'], df['rolling_views'] / 1000, label='30-day avg views (k)', linewidth=2)
axes[1,1].set_title('Rolling Averages (30-day window)', fontsize=12, fontweight='bold')
axes[1,1].set_xlabel('Date')
axes[1,1].set_ylabel('Value')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2_growth_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Growth patterns saved as '2_growth_patterns.png'")

# ============================================
# STEP 4: CORRELATION ANALYSIS
# ============================================
print("\n🔗 STEP 4: Correlation & Driver Analysis...")

correlation_cols = ['new_subscribers', 'total_views', 'likes', 'comments', 
                    'shares', 'engagement_rate', 'views_per_subscriber', 'video_uploaded']
correlation_matrix = df[correlation_cols].corr()

print("\nCorrelation with new_subscribers:")
print(correlation_matrix['new_subscribers'].sort_values(ascending=False))

# High vs low growth days
threshold = df['new_subscribers'].quantile(0.9)
high_growth_days = df[df['new_subscribers'] >= threshold]
low_growth_days = df[df['new_subscribers'] <= df['new_subscribers'].quantile(0.1)]

print(f"\nHigh Growth Days (top 10%): {len(high_growth_days)} days")
print(f"   Avg views: {high_growth_days['total_views'].mean():,.0f}")
print(f"   Avg engagement: {high_growth_days['engagement_rate'].mean():.3f}")

print(f"\nLow Growth Days (bottom 10%): {len(low_growth_days)} days")
print(f"   Avg views: {low_growth_days['total_views'].mean():,.0f}")
print(f"   Avg engagement: {low_growth_days['engagement_rate'].mean():.3f}")

# Correlation visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, ax=axes[0])
axes[0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

scatter = axes[1].scatter(df['total_views'], df['new_subscribers'], 
                         c=df['engagement_rate'], cmap='viridis', alpha=0.6, s=30)
axes[1].set_xlabel('Total Views')
axes[1].set_ylabel('New Subscribers')
axes[1].set_title('Views vs New Subscribers (colored by engagement)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=axes[1], label='Engagement Rate')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3_driver_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Driver analysis saved as '3_driver_analysis.png'")

# ============================================
# STEP 5: FORECASTING
# ============================================
print("\n🔮 STEP 5: Forecasting Future Growth...")

df_forecast = df.copy()
df_forecast['days_since_start'] = (df_forecast['date'] - df_forecast['date'].min()).dt.days

# Train/test split (last 60 days for testing)
train_size = len(df_forecast) - 60
train = df_forecast.iloc[:train_size]
test = df_forecast.iloc[train_size:]

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_train = train[['days_since_start']]
y_train = train['total_subscribers']
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(test[['days_since_start']])

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
test_pred_poly = poly_model.predict(X_test_poly)

# Evaluate
mae = mean_absolute_error(test['total_subscribers'], test_pred_poly)
rmse = np.sqrt(mean_squared_error(test['total_subscribers'], test_pred_poly))
print(f"\nModel Performance:")
print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")

# Forecast next 90 days
future_days = np.arange(len(df_forecast), len(df_forecast) + 90).reshape(-1, 1)
future_pred = poly_model.predict(poly.transform(future_days))
future_dates = pd.date_range(start=df['date'].max() + timedelta(days=1), periods=90)

print(f"\n90-Day Forecast:")
print(f"Current subscribers: {df['total_subscribers'].iloc[-1]:,.0f}")
print(f"Forecasted in 30 days: {future_pred[29]:,.0f}")
print(f"Forecasted in 60 days: {future_pred[59]:,.0f}")
print(f"Forecasted in 90 days: {future_pred[89]:,.0f}")

# Forecasting visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].plot(df_forecast['date'], df_forecast['total_subscribers'], label='Actual', linewidth=2, color='black')
axes[0].plot(test['date'], test_pred_poly, label='Predictions', linewidth=2, color='red', linestyle='--')
axes[0].set_title('Model Predictions vs Actual', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Total Subscribers')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_forecast['date'], df_forecast['total_subscribers'], label='Historical', linewidth=2, color='black')
axes[1].plot(future_dates, future_pred, label='Forecast', linewidth=2, color='green', linestyle='--')
axes[1].fill_between(future_dates, future_pred * 0.95, future_pred * 1.05, alpha=0.2, color='green')
axes[1].set_title('90-Day Subscriber Forecast', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Total Subscribers')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('4_forecasting.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Forecasting plots saved as '4_forecasting.png'")

# ============================================
# STEP 6: FINAL REPORT
# ============================================
print("\n📋 STEP 6: Generating Final Report...")

# Calculate insights
upload_boost = (df[df['video_uploaded']==1]['new_subscribers'].mean() / 
                df[df['video_uploaded']==0]['new_subscribers'].mean() - 1) * 100
best_dow = dow_analysis[('new_subscribers', 'mean')].idxmax()
best_month = monthly['new_subscribers'].idxmax()
engagement_corr = correlation_matrix.loc['new_subscribers', 'engagement_rate']

# Create HTML report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>YouTube Subscriber Growth Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2E86AB; }}
        h2 {{ color: #A23B72; border-bottom: 2px solid #A23B72; }}
        h3 {{ color: #F18F01; }}
        .metric {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .insight {{ background: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 4px solid #2E86AB; }}
        .action {{ background: #fff3e0; padding: 10px; margin: 10px 0; border-left: 4px solid #F18F01; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-number {{ font-size: 32px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 YouTube Subscriber & View Growth Analysis</h1>
        <p><strong>Analysis Period:</strong> {df['date'].min().date()} to {df['date'].max().date()}</p>
        <p><strong>Total Days Analyzed:</strong> {len(df)}</p>
        
        <div class="grid">
            <div class="stat-box">
                <div class="stat-number">{start_subs:,} → {end_subs:,}</div>
                <div class="stat-label">Subscriber Growth ({growth_rate:.1f}%)</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{df['total_views'].sum():,}</div>
                <div class="stat-label">Total Views</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{df['new_subscribers'].mean():.1f}</div>
                <div class="stat-label">Avg Daily New Subscribers</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{df['engagement_rate'].mean():.3f}</div>
                <div class="stat-label">Avg Engagement Rate</div>
            </div>
        </div>
        
        <h2>Key Insights</h2>
        <div class="insight">📹 <strong>Upload Impact:</strong> Upload days generate {upload_boost:.1f}% more subscribers than non-upload days</div>
        <div class="insight">📅 <strong>Best Day:</strong> {best_dow} is the optimal day for new subscriber growth</div>
        <div class="insight">🗓️ <strong>Best Month:</strong> {best_month} shows the strongest growth pattern</div>
        <div class="insight">💬 <strong>Engagement Correlation:</strong> Engagement rate has a {engagement_corr:.2f} correlation with new subscribers</div>
        <div class="insight">🎯 <strong>90-Day Forecast:</strong> Expected to reach {int(future_pred[89]):,} subscribers in 90 days</div>
        
        <h2>Recommendations</h2>
        <div class="action">✅ Increase upload frequency to 4-5 videos per week, focusing on {best_dow}</div>
        <div class="action">✅ Create content that drives comments and shares to boost engagement rate</div>
        <div class="action">✅ Plan major campaigns and launches during {best_month}</div>
        <div class="action">✅ Add stronger calls-to-action in videos to improve subscriber conversion</div>
        <div class="action">✅ Monitor the 30-day rolling averages to detect growth trends early</div>
        
        <h2>Action Plan (Next 30 Days)</h2>
        <div class="action">📌 Week 1: Audit top 10 performing videos and replicate their format</div>
        <div class="action">📌 Week 2: Implement end screens and cards to boost subscriber conversion</div>
        <div class="action">📌 Week 3: Schedule uploads on peak days ({best_dow})</div>
        <div class="action">📌 Week 4: Launch engagement campaign to boost interaction</div>
        
        <h2>Visualizations Generated</h2>
        <ul>
            <li><code>1_eda_plots.png</code> - Exploratory data analysis</li>
            <li><code>2_growth_patterns.png</code> - Day/week/month patterns</li>
            <li><code>3_driver_analysis.png</code> - Correlation analysis</li>
            <li><code>4_forecasting.png</code> - Future predictions</li>
        </ul>
        
        <p><em>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        <p><strong>Data file:</strong> youtube_data.csv</p>
    </div>
</body>
</html>
"""

with open('growth_report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("✅ HTML report saved as 'growth_report.html'")

# Print final summary
print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📁 Files created in your project folder:")
print("   • youtube_data.csv - Raw data")
print("   • 1_eda_plots.png - Exploratory analysis plots")
print("   • 2_growth_patterns.png - Growth pattern visualizations")
print("   • 3_driver_analysis.png - Correlation analysis")
print("   • 4_forecasting.png - Forecast charts")
print("   • growth_report.html - Complete interactive report")
print("\n🌐 Open 'growth_report.html' in your browser to view the full report")
print("\n✅ All done! 🎉")
