import pandas as pd

ISEI_index = pd.read_excel('/Users/alexpeacock/Dropbox/Capstone_data_stuff/ISCO08_with_ISEI.xlsx', header =1)

ISEI_index['major'] = ISEI_index['Major'].apply(lambda x: '{:04.0f}'.format(x) if pd.notnull(x) else '')
ISEI_index['sub- major'] = ISEI_index['sub- major'].apply(lambda x: '{:04.0f}'.format(x) if pd.notnull(x) else '')
ISEI_index['minor'] = ISEI_index['minor'].apply(lambda x: '{:04.0f}'.format(x) if pd.notnull(x) else '')
ISEI_index['unit'] = ISEI_index['unit'].apply(lambda x: '{:04.0f}'.format(x) if pd.notnull(x) else '')



melted_index = ISEI_index.melt(id_vars=['TITLE', 'ISEI- 08', 'ISSP- N'], value_vars=['Major', 'sub- major', 'minor', 'unit'], 
                    var_name='Type', value_name='Code')

# Drop rows with NaN in 'Code' column
cleaned_ISEI_index = melted_index.dropna(subset=['Code'])

# Drop the 'Type' column as it's no longer needed
ISEI_index = cleaned_ISEI_index.drop(columns=['Type'])

# Reset index if needed
ISEI_index.reset_index(drop=True, inplace=True)

