import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# creating basic classifier for rock sizes
class BasicClassifier():
    """
    Sorts images into 3 classes based on their area: 
    small_rock (1), medium_rock (2), large_rock (3)
    """
    
    def __init__(self, df: pd.DataFrame, img_width=720, img_height=480):
        self.df = df.copy()
        self.img_width = img_width
        self.img_height = img_height
        
        # un-normalizing box height/width to calculate absolute area
        self.df['Area'] = self.df['Length'] * self.df['Height']
        
        self.df = self.df.sort_values(by='Area', ascending=True)
        
        # split dataset into thirds
        df_length = len(self.df)
        self.lower_limit = self.df['Area'].iloc[df_length // 3 - 1] # first third
        self.middle_limit = self.df['Area'].iloc[2 * df_length // 3 - 1] # second third
        
    def classify_df(self):
        return self.df.apply(
            lambda row: self.classify_box(
                box_width = row['Length'],
                box_height = row['Height']
            ),
            axis=1 
        )

    def classify_box(self, box_width: float, box_height: float):
        box_area = box_width * box_height

        # categorize rock depending on size
        if box_area <= 0:
            return 0 # None
        elif 0 < box_area <= self.lower_limit:
            return 1 # Small Rock
        elif self.lower_limit < box_area <= self.middle_limit:
            return 2 # Medium Rocks
        else:
            return 3 # Large Rock




    # MISC: helpful dataset visualizer functions (below)

    # visualize distribution of bounding box sizes and help determine approx. class sizes
    def box_size_distr_visualizer(self):
        print("Pre-Filtering Values:")
        self.key_value_statistics(self.df['Area'])

        # Remove outliers using 1.5*IQR Rule
        AreaQ1 = self.df['Area'].quantile(0.25)
        AreaQ3 = self.df['Area'].quantile(0.75)
        area_iqr = AreaQ3 - AreaQ1
        iqr_for_q1 = AreaQ1 - 1.5 * area_iqr
        iqr_for_q3 = AreaQ3 + 1.5 * area_iqr

        df_cleaned = self.df['Area'][(self.df['Area'] >= iqr_for_q1) & (self.df['Area'] <= iqr_for_q3)]
        print("Post-Filtering Values:")
        self.key_value_statistics(df_cleaned)
        
        # Visualize dataset
        custom_bins = list(np.arange(0,10**4, 500))
        plt.hist(self.df['Area'], bins=custom_bins)
        plt.title('Area Distribution of Bounding Box Sizes')
        plt.xlabel('Box Area')
        plt.ylabel('Frequency')
        plt.show()
    
    # display key statistics for a list/column of values
    def key_value_statistics(self, values): 
        print(f"Max: {values.max()}. Min: {values.min()}. Median: {values.median()}. Mean: {values.mean()} ")