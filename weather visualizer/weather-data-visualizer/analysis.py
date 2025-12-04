# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path('data/raw_weather.csv')

def load_data(file_path):
    """Load the CSV file into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(df.head())
        print(df.info()) # Inspect structure [cite: 23]
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    # analysis.py (Cont.)

def clean_data(df):
    # Rename columns to standard names for simplicity (Adjust these based on your actual CSV)
    # Example: df = df.rename(columns={'Your_Date_Col': 'Date', 'Temp_C': 'Temperature'})
    
    # 1. Convert date column to datetime format (Task 2)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # 2. Filter for relevant columns (Task 2)
    # Filter only relevant columns needed for analysis (e.g., Temperature, Humidity, Rainfall) [cite: 25]
    relevant_cols = ['Temperature', 'Rainfall', 'Humidity'] 
    df = df[relevant_cols].copy()
    
    # 3. Handle Missing Values (Task 2)
    # Option A: Drop rows with missing values 
    df = df.dropna() 
    # Option B: Fill missing values (e.g., filling temperature with the mean)
    # df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
    
    print("\nData after cleaning:")
    print(df.info())
    return df
# analysis.py (Cont.)

def perform_analysis(df):
    # Task 3: Overall Statistics (Using NumPy/Pandas)
    print("\n--- Overall Statistics ---")
    print(df.describe())
    
    # Task 5: Grouping and Aggregation (Monthly Stats)
    # Resample is great for time series aggregation (Task 5) [cite: 29]
    monthly_summary = df.resample('M').agg({
        'Temperature': ['mean', 'max', 'min', np.std], # Use NumPy for standard deviation 
        'Rainfall': 'sum'
    })
    
    # Task 3: Example of yearly statistics (Grouping by year)
    yearly_avg = df['Temperature'].resample('Y').mean()
    
    return monthly_summary, yearly_avg
# analysis.py (Cont.)

def perform_analysis(df):
    # Task 3: Overall Statistics (Using NumPy/Pandas)
    print("\n--- Overall Statistics ---")
    print(df.describe())
    
    # Task 5: Grouping and Aggregation (Monthly Stats)
    # Resample is great for time series aggregation (Task 5) [cite: 29]
    monthly_summary = df.resample('M').agg({
        'Temperature': ['mean', 'max', 'min', np.std], # Use NumPy for standard deviation 
        'Rainfall': 'sum'
    })
    
    # Task 3: Example of yearly statistics (Grouping by year)
    yearly_avg = df['Temperature'].resample('Y').mean()
    
    return monthly_summary, yearly_avg
# analysis.py (Cont.)
import matplotlib.dates as mdates

def create_visualizations(df, monthly_summary):
    # Task 4: Line Chart for Daily Temperature Trends 
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Temperature'], label='Daily Temperature')
    plt.title('Daily Temperature Trend')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.savefig('images/daily_temp_line.png') # Save plot 
    plt.close()

    # Task 4: Bar Chart for Monthly Rainfall Totals 
    plt.figure(figsize=(10, 5))
    monthly_summary['Rainfall']['sum'].plot(kind='bar')
    plt.title('Monthly Rainfall Totals')
    plt.ylabel('Total Rainfall (mm)')
    plt.xlabel('Month')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images/monthly_rainfall_bar.png')
    plt.close()
    
    # Task 4: Scatter Plot for Humidity vs. Temperature 
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Humidity'], df['Temperature'], alpha=0.5)
    plt.title('Humidity vs. Temperature')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.savefig('images/humidity_temp_scatter.png')
    plt.close()
    
    # Task 4: Combined Plot (Subplots) - Bonus/Advanced Plotting [cite: 28, 33]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Temperature and Humidity Comparison')
    
    axes[0].plot(df.index, df['Temperature'], color='red')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Daily Temperature')
    
    axes[1].plot(df.index, df['Humidity'], color='blue')
    axes[1].set_ylabel('Humidity (%)')
    axes[1].set_title('Daily Humidity')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('images/combined_plot.png')
    plt.close()
    print("All visualizations saved to the 'images' folder.")
    # analysis.py (Cont.)

def main():
    # Setup folders
    Path('data').mkdir(exist_ok=True)
    Path('images').mkdir(exist_ok=True)
    
    df = load_data(DATA_PATH)
    if df is not None:
        cleaned_df = clean_data(df)
        
        # Task 6: Export cleaned data 
        cleaned_df.to_csv('data/cleaned_weather_data.csv')
        print("\nCleaned data exported to data/cleaned_weather_data.csv")

        monthly_stats, yearly_stats = perform_analysis(cleaned_df)
        create_visualizations(cleaned_df, monthly_stats)
        
        # Now, manually write your findings into the report.md file (Task 6)

if __name__ == "__main__":
    main()