import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 1. Import the data from `medical_examination.csv` and assign it to the `df` variable.
df = pd.read_csv("medical_examination.csv")

# 2. Add an 'overweight' column to the data.
#    Overweight is defined as BMI > 25, where
#      BMI = weight (kg) / [height (m)]^2.
#    Use 0 if BMI <= 25 and 1 if BMI > 25.
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize data by making 0 always 'good' and 1 always 'bad' for cholesterol and gluc.
#    If the value of cholesterol or gluc is 1, set it to 0. If the value is > 1, set it to 1.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc']        = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

def draw_cat_plot():
    """
    4. Draw the Categorical Plot in the `draw_cat_plot` function.
    5. Create a DataFrame for the cat plot using `pd.melt` with the values from
       cholesterol, gluc, smoke, alco, active, and overweight. Assign this to `df_cat`.
    6. Group and reformat the data in `df_cat` to split it by cardio. Show the counts of each feature.
       Rename the 'size' column to 'total' so the catplot works correctly.
    7. Convert the data into long format and create a chart that shows the value counts
       of the categorical features using `sns.catplot(...)`.
    8. Get the figure for the output and store it in the `fig` variable.
    Do not modify the next two lines.
    """
    # Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # Group and reformat the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat = df_cat.rename(columns={'size': 'total'})

    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    )

    fig = catplot.fig  # Get the figure for output

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    """
    9. Draw the Heat Map in the `draw_heat_map` function.
    10. Clean the data in `df_heat` by filtering out the following patient segments
        that represent incorrect data:
         - diastolic pressure > systolic (keep `ap_lo <= ap_hi`)
         - height < 2.5th percentile or height > 97.5th percentile
         - weight < 2.5th percentile or weight > 97.5th percentile
    11. Calculate the correlation matrix and store it in the corr variable.
    12. Generate a mask for the upper triangle and store it in the mask variable.
    13. Set up the matplotlib figure.
    14. Plot the correlation matrix using `sns.heatmap(...)`.
    Do not modify the next two lines.
    """
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr(method='pearson')

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        cbar_kws={'shrink': 0.5},
        center=0,
        vmax=0.3,
        vmin=-0.1
    )

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig

