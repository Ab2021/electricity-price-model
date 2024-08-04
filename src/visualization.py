import matplotlib.pyplot as plt
import seaborn as sns

def plot_yearly_avg_price(data):
    yearly_avg_price = data.groupby('year')['price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_avg_price['year'], yearly_avg_price['price'])
    plt.title('Year-wise Average Electricity Price')
    plt.xlabel('Year')
    plt.ylabel('Average Electricity Price')
    plt.grid(True)
    plt.savefig('yearly_avg_price.png')
    plt.close()

def plot_monthly_avg_price(data):
    plt.figure(figsize=(12, 8))
    monthly_avg = data.groupby(['year', 'month'])['price'].mean().reset_index()
    for year in monthly_avg['year'].unique():
        data_by_year = monthly_avg[monthly_avg['year'] == year]
        plt.plot(data_by_year['month'], data_by_year['price'], label=year)
    plt.title('Monthly Average Electricity Price for Each Year')
    plt.xlabel('Month')
    plt.ylabel('Average Price')
    plt.legend(title='Year')
    plt.grid(True)
    plt.xticks(range(1, 13))
    plt.savefig('monthly_avg_price.png')
    plt.close()

def plot_correlation_heatmap(data):
    corr_matrix = data[['price', 'revenue', 'sales']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()