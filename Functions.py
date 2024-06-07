import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as sps

def Mum_HIED(df):
    
    
    mum_education_rank = {
        'She did not complete  ISCED level 1': 0,
        'ISCED level 1': 1,
        'ISCED level 2': 2,
        'ISCED level 3B, 3C': 3,
        'ISCED level 3A': 4,
        'ISCED level 4': 5,
        'ISCED level 5B': 6,
        'ISCED level 5A': 7,
        'ISCED level 6': 8,
        'No': -1,  # Intermediate values for binary columns
        'Yes_MumEd4': 5,  # ISCED level 4
        'Yes_MumEd5b': 6, # ISCED level 5B
        'Yes_MumEd5a': 7, # ISCED level 5A
        'Yes_MumEd6': 8,  # ISCED level 6
        np.nan: -2  # Value to handle NaNs
    }

    # Function to normalize the values in the 'MumEd3a' column
    def normalize_education_level(value):
        return mum_education_rank.get(value, -2)

    # Normalize the 'MumEd3a' column
    df['NormalizedEd3a'] = df['MumEd3a'].apply(normalize_education_level)

    # Convert 'Yes'/'No' to appropriate levels in other education columns
    binary_columns = ['MumEd4', 'MumEd5b', 'MumEd5a', 'MumEd6']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': mum_education_rank[f'Yes_{col}'], 'No': -1, np.nan: -2})

    # Function to determine the highest education level
    def get_highest_education(row):
        # Get the maximum value from the normalized columns
        max_level = max(row['NormalizedEd3a'], *[row[col] for col in binary_columns])
        # Return the corresponding education level
        for key, value in mum_education_rank.items():
            if value == max_level and value >= 0:  # Ensure valid ISCED levels are returned, ignoring 'No'
                if isinstance(key, str):
                    return key.replace('Yes_', '')  # Remove the 'Yes_' prefix for binary columns
                else:
                    return key
        return np.nan

    # Apply the function to each row in the DataFrame
    df['HighestEducationMum'] = df.apply(get_highest_education, axis=1)

    # Drop the normalized column if no longer needed
    df.drop(columns=['NormalizedEd3a'], inplace=True)

    return df


def Dad_HIED(df):
    
    
    dad_education_rank = {
        'He did not complete  ISCED level 1': 0,
        'ISCED level 1': 1,
        'ISCED level 2': 2,
        'ISCED level 3B, 3C': 3,
        'ISCED level 3A': 4,
        'ISCED level 4': 5,
        'ISCED level 5B': 6,
        'ISCED level 5A': 7,
        'ISCED level 6': 8,
        'No': -1,  # Intermediate values for binary columns
        'Yes_DadEd4': 5,  # ISCED level 4
        'Yes_DadEd5b': 6, # ISCED level 5B
        'Yes_DadEd5a': 7, # ISCED level 5A
        'Yes_DadEd6': 8,  # ISCED level 6
        np.nan: -2  # Value to handle NaNs
    }

    # Function to normalize the values in the 'MumEd3a' column
    def normalize_education_level(value):
        return dad_education_rank.get(value, -2)

    # Normalize the 'MumEd3a' column
    df['NormalizedEd3a'] = df['DadEd3a'].apply(normalize_education_level)

    # Convert 'Yes'/'No' to appropriate levels in other education columns
    binary_columns = ['DadEd4', 'DadEd5b', 'DadEd5a', 'DadEd6']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': dad_education_rank[f'Yes_{col}'], 'No': -1, np.nan: -2})

    # Function to determine the highest education level
    def get_highest_education(row):
        # Get the maximum value from the normalized columns
        max_level = max(row['NormalizedEd3a'], *[row[col] for col in binary_columns])
        # Return the corresponding education level
        for key, value in dad_education_rank.items():
            if value == max_level and value >= 0:  # Ensure valid ISCED levels are returned, ignoring 'No'
                if isinstance(key, str):
                    return key.replace('Yes_', '')  # Remove the 'Yes_' prefix for binary columns
                else:
                    return key
        return np.nan

    # Apply the function to each row in the DataFrame
    df['HighestEducationDad'] = df.apply(get_highest_education, axis=1)

    # Drop the normalized column if no longer needed
    df.drop(columns=['NormalizedEd3a'], inplace=True)

    return df




def add_parent_education_column(df):
    # Define the education levels with a numerical ranking for comparison
    education_rank = {
        'She did not complete  ISCED level 1': 0,
        'He did not complete  ISCED level 1': 0,
        'ISCED level 1': 1,
        'ISCED level 2': 2,
        'ISCED level 3B, 3C': 3,
        'ISCED level 3A': 4,
        'ISCED level 4': 5,
        'ISCED level 5B': 6,
        'ISCED level 5A': 7,
        'ISCED level 6': 8,
        np.nan: -1  # Value to handle NaNs
    }

    # Function to get the higher education level between mum and dad
    def get_highest_education(mum_edu, dad_edu):
        mum_rank = education_rank.get(mum_edu, -1)
        dad_rank = education_rank.get(dad_edu, -1)
        if mum_rank > dad_rank:
            return mum_edu
        else:
            return dad_edu

    # Apply the function to create the 'ParEd' column
    df['ParEd'] = df.apply(lambda row: get_highest_education(row['HighestEducationMum'], row['HighestEducationDad']), axis=1)

    return df

#Assign the nunbers of years of study to the highest Parental Edcuation





def split_combined_questions(row):
    if row['CSWIa'] == 'Missing':
        row['CSWIa_ENG'] = 'Missing'
        row['CSWIa_SCOT'] = 'Missing'
    else:
        row['CSWIa_ENG'] = 'Yes' if 'United Kingdom (Excl. Scotland) : Dishwasher-Yes' in row['CSWIa'] else 'No'
        row['CSWIa_SCOT'] = 'Yes' if 'United Kingdom (Scotland) : A TV streaming subscription that is paid for (e.g. Netflix, Amazon Prime)-Yes' in row['CSWIa'] else 'No'
    
    if row['CSWIb'] == 'Missing':
        row['CSWIb_ENG'] = 'Missing'
        row['CSWIb_SCOT'] = 'Missing'
    else:
        row['CSWIb_ENG'] = 'Yes' if 'United Kingdom (Excl. Scotland) : Tumble dryer-Yes' in row['CSWIb'] else 'No'
        row['CSWIb_SCOT'] = 'Yes' if 'United Kingdom (Scotland) : A musical instrument (e.g. piano, violin)-Yes' in row['CSWIb'] else 'No'
    
    return row


import pandas as pd



def calculate_poss_score(row, costs_df, cost_column):
    scores = []
    for question in row.index:
        response = row[question]
        if question in costs_df['Question'].values:
            cost = costs_df[costs_df['Question'] == question][cost_column].values[0]
            if response == 'Yes':
                scores.append(cost)
            elif response == 'Three or more':
                scores.append(cost * 3)
            elif response == 'Two':
                scores.append(cost * 2)
            elif response == 'One':
                scores.append(cost)
    
    # Sort the scores in descending order and sum the top 20 scores
    top_scores = sorted(scores, reverse=True)[:20]
    return sum(top_scores)





def create_par_os_score(df, col1, col2):
    
    # Convert the columns to numeric, coercing errors to NaN
    df[col1] = pd.to_numeric(df[col1], errors='coerce')
    df[col2] = pd.to_numeric(df[col2], errors='coerce')
    
    # Fill NaN values with a very small number (or any other strategy you prefer)
    df[col1].fillna(float('-inf'), inplace=True)
    df[col2].fillna(float('-inf'), inplace=True)
    
    # Calculate the maximum value between the two columns for each row
    df['par_os_score'] = df[[col1, col2]].max(axis=1)
    
    # Replace -inf back to NaN if necessary
    df['par_os_score'].replace(float('-inf'), float('nan'), inplace=True)
    
    return df


   
    

'''

    Weights_for_columns = {
    'IYHa': {'Yes': 1, 'No': 0},    # In your home do you have a desk to study at? £75
    'IYHb': {'Yes': 3, 'No': 0},    # A room of your own? £24000
    'IYHc': {'Yes': 1, 'No': 0},  # A quiet place to study £11000
    'IYHd': {'Yes': 2, 'No': 0},    # A computer you can use for school work £850
    'IYHe': {'Yes': 1, 'No': 0},    # Educational software £50
    'IYHf': {'Yes': 1.5, 'No': 0},  # Internet connection £3000
    'IYHg': {'Yes': 1, 'No': 0},    # Classic literature £10
    'IYHh': {'Yes': 1, 'No': 0},    # Books of poetry £15 
    'IYHi': {'Yes': 1, 'No': 0},    # Works of art £100
    'IYHj': {'Yes': 1, 'No': 0},  # Books to help you study £10
    'IYHk': {'Yes': 1, 'No': 0},    # Technical reference book £30
    'IYHl': {'Yes': 1, 'No': 0},    # A dictionary £30
    'IYHm': {'Yes': 1, 'No': 0},    # Books on art, music or design £30
    'CSWIa': {'Yes': 1.5, 'No': 0}, # Dishwasher (England), TV streaming subscription (Scotland) £500, £1000 
    'CSWIb': {'Yes': 1.5, 'No': 0},   # Tumble dryer (England), a musical instrument (Scotland) £425, 230
    'CSWIc': {'Yes': 3, 'No': 0},    # Someone your parents pay to help around the house (Whole UK) £35000 
    'HMIYHa': {'One': 1, 'Two': 3, 'Three or more': 4, np.nan: 0},   # How many TVs in your home? £400
    'HMIYHb': {'One': 2, 'Two': 4, 'Three or more': 5, np.nan: 0},   # How many cars in your home? £17,000
    'HMIYHc': {'One': 1, 'Two': 3, 'Three or more': 5, np.nan: 0},   # How many bathrooms in your home? £12000
    'HMIYHd': {'One': 1, 'Two': 2, 'Three or more': 3, np.nan: 0},   # How many smartphones in your home? £2000
    'HMIYHe': {'One': 1, 'Two': 3, 'Three or more': 4, np.nan: 0},   # How many computers in your home? £850
    'HMIYHf': {'One': 1, 'Two': 2, 'Three or more': 4, np.nan: 0},   # How many tablets in your home? £525
    'HMIYHg': {'One': 1, 'Two': 2, 'Three or more': 3, np.nan: 0},   # How many e-book readers in your home?£250
    'HMIYHh': {'One': 1, 'Two': 2, 'Three or more': 3, np.nan: 0},    # How many musical instruments in your home? £230
    'HMIYHi': {'One': 1, 'Two': 2, 'Three or more': 3, np.nan: 0}    #How many books in your home? £10
                             }
'''

def calculate_standard_error(mean, replicate_weights_df, values):
    # Ensure the input is a DataFrame
    if not isinstance(replicate_weights_df, pd.DataFrame):
        raise ValueError("replicate_weights_df must be a pandas DataFrame")

    # Ensure the input values is a pandas Series
    if not isinstance(values, pd.Series):
        raise ValueError("values must be a pandas Series")
    
    # Number of replicates
    R = 80
    
    # Calculate the estimates using the replicate weights
    replicate_estimates = []
    for i in range(R):
        T_r = (replicate_weights_df.iloc[:, i] * values).mean()
        replicate_estimates.append(T_r)
    
    # Calculate the sampling variance
    variance = (1 / 20) * sum((T_r - mean) ** 2 for T_r in replicate_estimates)
    
    # Calculate the standard error
    standard_error = np.sqrt(variance)
    
    return standard_error

import pandas as pd

import pandas as pd



def calculate_average_statistics(dataframe, subject):
    slopes = []
    intercepts = []
    r_values = []
    p_values = []
    std_errs = []

    for i in range(1, 11):
        pv_column = f'PV{i}{subject}_weighted'
        slope, intercept, r_value, p_value, std_err = sps.linregress(dataframe['wghted_SES_Score'], dataframe[pv_column])
        slopes.append(slope)
        intercepts.append(intercept)
        r_values.append(r_value)
        p_values.append(p_value)
        std_errs.append(std_err)

    # Calculate average of regression results
    avg_slope = np.mean(slopes)
    avg_intercept = np.mean(intercepts)
    avg_r_value = np.mean(r_values)
    avg_p_value = np.mean(p_values)
    avg_std_err = np.mean(std_errs)

    return {
        'Average Slope': avg_slope,
        'Average Intercept': avg_intercept,
        'Average R-value': avg_r_value,
        'Average P-value': avg_p_value,
        'Average Standard Error': avg_std_err
    }

'''
weighted_read_cols = [f'PV{i}SCIE_weighted' for i in range(1, 11)]
    ...: 
    ...: # Create a scatter plot
    ...: plt.figure(figsize=(15, 10))
    ...: 
    ...: for i, read_col in enumerate(weighted_scie_cols, start=1):
    ...:     plt.scatter(merged['wghted_SES_Score'], merged[read_col], label=f'Plausible Value {i}', alpha=0.5)
    ...: plt.plot(SES_vs_pv_x, scie_SES_y, color ='red', label = 'linear regression line')
    ...: # Set the labels and title with larger font sizes
    ...: plt.xlabel('Weighted SES Score', fontsize=16)
    ...: plt.ylabel('SCIE Score', fontsize=16)
    ...: plt.title('Plausible Values of SCIE Scores vs Weighted SES Score', fontsize=20)
    ...: plt.legend(fontsize=14)
    ...: 
    ...: # Save the plot as a PNG file
    ...: plt.savefig('scie_SES.png', format='png')




'''
'''
def correlation_standard_error(original_correlation, T_ICT_score, SES_score, replicate_weights):
    """
    Calculate the standard error using balanced repeated replication (BRR) method.

    Parameters:
    - original_correlation: The correlation coefficient calculated using the original sample (without replicate weights).
    - T_ICT_score: The T_ICT scores as a numpy array.
    - SES_score: The SES scores as a numpy array.
    - replicate_weights: A numpy array of shape (80, n) where each row represents the replicate weights for the samples.

    Returns:
    - standard_error: The standard error of the correlation coefficient.
    """
    replicate_correlations = []

    # Calculate correlation for each replicate weight
    for weights in replicate_weights:
        weighted_T_ICT_score = T_ICT_score * weights
        weighted_SES_score = SES_score * weights

        correlation = np.corrcoef(weighted_T_ICT_score, weighted_SES_score)[0, 1]
        replicate_correlations.append(correlation)

    # Calculate sampling variance
    replicate_correlations = np.array(replicate_correlations)
    sampling_variance = (1 / 20) * np.sum((replicate_correlations - original_correlation) ** 2)

    # Calculate standard error
    standard_error = np.sqrt(sampling_variance)

    return standard_error

'''

import numpy as np

def correlation_standard_error(original_corr, ict_scores, ses_scores, rep_weights):
    """
    Calculate the standard error using balanced repeated replication (BRR) method.

    Parameters:
    - original_corr: The correlation coefficient calculated using the original sample (without replicate weights).
    - ict_scores: The T_ICT scores as a numpy array.
    - ses_scores: The SES scores as a numpy array.
    - rep_weights: A numpy array of shape (80, n) where each row represents the replicate weights for the samples.

    Returns:
    - standard_error: The standard error of the correlation coefficient.
    """
    # Debugging: Print the shapes of the arrays
    print(f"Original correlation: {original_corr}")
    print(f"ict_scores shape: {ict_scores.shape}")
    print(f"ses_scores shape: {ses_scores.shape}")
    print(f"rep_weights shape: {rep_weights.shape}")

    # Ensure that the input arrays are properly aligned
    if len(ict_scores) != len(ses_scores):
        raise ValueError("ict_scores and ses_scores arrays must have the same length.")
    if rep_weights.shape[1] != len(ict_scores):
        raise ValueError("Each row in rep_weights must have the same length as ict_scores and ses_scores.")

    replicate_correlations = []

    # Calculate correlation for each replicate weight
    for weights in rep_weights:
        weighted_ict_scores = ict_scores * weights
        weighted_ses_scores = ses_scores * weights

        correlation = np.corrcoef(weighted_ict_scores, weighted_ses_scores)[0, 1]
        replicate_correlations.append(correlation)

    # Calculate sampling variance
    replicate_correlations = np.array(replicate_correlations)
    sampling_variance = (1 / 20) * np.sum((replicate_correlations - original_corr) ** 2)

    # Calculate standard error
    standard_error = np.sqrt(sampling_variance)

    return standard_error

