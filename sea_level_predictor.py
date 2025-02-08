import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    # Read data from file
    df = pd.read_csv("epa-sea-level.csv")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Year'], df['CSIRO Adjusted Sea Level'], c='blue', alpha=0.5)

    # -------------------------------
    # 1) Line of Best Fit: All Data
    # -------------------------------
    # Perform linear regression on all available years
    slope_all, intercept_all, r_value_all, p_value_all, std_err_all = linregress(
        df['Year'],
        df['CSIRO Adjusted Sea Level']
    )

    # Create x values from the first year in dataset (1880) through 2050
    years_extend = range(df['Year'].min(), 2051)
    # Compute predicted y values
    sea_levels_all = [intercept_all + slope_all * year for year in years_extend]

    # Plot this line of best fit
    ax.plot(years_extend, sea_levels_all, 'r', label="Fit: 1880-2050")

    # -------------------------------------------------
    # 2) Line of Best Fit: From Year 2000 to Present
    # -------------------------------------------------
    # Subset the data from year 2000 onwards
    df_2000 = df[df['Year'] >= 2000]

    # Perform linear regression on subset
    slope_2000, intercept_2000, r_value_2000, p_value_2000, std_err_2000 = linregress(
        df_2000['Year'],
        df_2000['CSIRO Adjusted Sea Level']
    )

    # Create x values from 2000 through 2050
    years_extend_2000 = range(2000, 2051)
    # Compute predicted y values
    sea_levels_2000 = [intercept_2000 + slope_2000 * year for year in years_extend_2000]

    # Plot this line of best fit
    ax.plot(years_extend_2000, sea_levels_2000, 'green', label="Fit: 2000-2050")

    # Add labels and title
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (inches)")
    ax.set_title("Rise in Sea Level")

    # Add legend
    ax.legend()

    # Save plot and return for unit tests
    plt.savefig('sea_level_plot.png')
    return plt.gca()
