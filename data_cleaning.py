import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import Functions as F
import matplotlib.pyplot as plt
import scipy.stats as sps 


#Columns of interest for teacher data

teacher_cols = ['CNT','CNTSCHID','CNTTCHID','NatCen','Region','STRATUM',
                'TC045Q05NA','TC045Q05NB', 'TC185Q05HA', 'TC021Q01NA', 
                  'TC199Q03HA', 'TC166Q01HA', 'TC166Q02HA', 'TC166Q03HA',
                   'TC166Q04HA', 'TC166Q05HA', 'TC166Q06HA', 'TC166Q07HA',
                   'TC184Q01HA', 'TC167Q01HA', 'TC169Q01HA','TC169Q02HA',
                   'TC169Q03HA','TC169Q04HA','TC169Q05HA','TC169Q06HA',
                   'TC169Q07HA','TC169Q08HA', 'TC169Q09HA', 'TC169Q10HA',
                   'TC169Q11HA','TC169Q12HA','TC169Q13HA','TC169Q14HA',
]


#Columns of interest for student data

stu_cols = ['CNT', 'CNTSCHID', 'STRATUM', 'ST005Q01TA', 'ST006Q01TA',
            'ST006Q02TA', 'ST006Q03TA' ,'ST006Q04TA' ,'ST007Q01TA','ST008Q01TA',
            'ST008Q02TA', 'ST008Q03TA', 'ST008Q04TA','ST011Q01TA', 'ST011Q02TA',
            'ST011Q03TA', 'ST011Q04TA', 'ST011Q05TA', 'ST011Q06TA',
            'ST011Q07TA', 'ST011Q08TA', 'ST011Q09TA', 'ST011Q10TA',
            'ST011Q11TA', 'ST011Q12TA', 'ST011Q16NA', 'ST011D17TA',
            'ST011D18TA', 'ST011D19TA', 'ST012Q01TA', 'ST012Q02TA',
            'ST012Q03TA', 'ST012Q05NA', 'ST012Q06NA', 'ST012Q07NA',
            'ST012Q08NA', 'ST012Q09NA',  'ST158Q01HA', 'ST158Q02HA',
            'ST158Q02HA', 'ST158Q03HA', 'ST158Q04HA', 'ST158Q05HA',
            'ST158Q06HA', 'ST158Q07HA', 'PA042Q01TA', 'MISCED',
            'FISCED', 'OCOD1', 'OCOD2', 'HOMEPOS', 'W_FSTURWT1', 'W_FSTUWT', 
            'W_FSTURWT1', 'W_FSTURWT2', 'W_FSTURWT3', 'W_FSTURWT4', 'W_FSTURWT5',
            'W_FSTURWT6', 'W_FSTURWT7', 'W_FSTURWT8', 'W_FSTURWT9', 'W_FSTURWT10',
            'W_FSTURWT11', 'W_FSTURWT12', 'W_FSTURWT13', 'W_FSTURWT14', 'W_FSTURWT15',
            'W_FSTURWT16', 'W_FSTURWT17', 'W_FSTURWT18', 'W_FSTURWT19', 'W_FSTURWT20',
            'W_FSTURWT21', 'W_FSTURWT22', 'W_FSTURWT23', 'W_FSTURWT24', 'W_FSTURWT25',
            'W_FSTURWT26', 'W_FSTURWT27', 'W_FSTURWT28', 'W_FSTURWT29', 'W_FSTURWT30',
            'W_FSTURWT31', 'W_FSTURWT32', 'W_FSTURWT33', 'W_FSTURWT34', 'W_FSTURWT35',
            'W_FSTURWT36', 'W_FSTURWT37', 'W_FSTURWT38', 'W_FSTURWT39', 'W_FSTURWT40',
            'W_FSTURWT41', 'W_FSTURWT42', 'W_FSTURWT43', 'W_FSTURWT44', 'W_FSTURWT45',
            'W_FSTURWT46', 'W_FSTURWT47', 'W_FSTURWT48', 'W_FSTURWT49', 'W_FSTURWT50',
            'W_FSTURWT51', 'W_FSTURWT52', 'W_FSTURWT53', 'W_FSTURWT54', 'W_FSTURWT55',
            'W_FSTURWT56', 'W_FSTURWT57', 'W_FSTURWT58', 'W_FSTURWT59', 'W_FSTURWT60',
            'W_FSTURWT61', 'W_FSTURWT62', 'W_FSTURWT63', 'W_FSTURWT64', 'W_FSTURWT65',
            'W_FSTURWT66', 'W_FSTURWT67', 'W_FSTURWT68', 'W_FSTURWT69', 'W_FSTURWT70',
            'W_FSTURWT71', 'W_FSTURWT72', 'W_FSTURWT73', 'W_FSTURWT74', 'W_FSTURWT75',
            'W_FSTURWT76', 'W_FSTURWT77', 'W_FSTURWT78', 'W_FSTURWT79', 'W_FSTURWT80', 'ESCS', 'PV1MATH',
            'PV2MATH', 'PV3MATH', 'PV4MATH','PV5MATH', 'PV6MATH', 'PV7MATH', 'PV8MATH',
            'PV9MATH', 'PV10MATH', 'PV1READ', 'PV2READ', 'PV3READ', 'PV4READ','PV5READ', 'PV6READ', 'PV7READ', 'PV8READ',
            'PV9READ', 'PV10READ', 'PV1SCIE',
            'PV2SCIE', 'PV3SCIE', 'PV4SCIE','PV5SCIE', 'PV6SCIE', 'PV7SCIE', 'PV8SCIE',
            'PV9SCIE', 'PV10SCIE']
 

 #Importing relevant data      

stu_data =  pd.read_csv('/Users/alexpeacock/Dropbox/Capstone_data_stuff/UKStudent2018.csv', low_memory = False)
teacher_data = pd.read_csv('/Users/alexpeacock/Dropbox/Capstone_data_stuff/UKTeacher2018.csv', usecols = teacher_cols, low_memory=False)
#school_data = pd.read_csv('/Users/alexpeacock/Dropbox/Capstone_data_stuff/UKSChool2018.csv')

ic_columns = [col for col in stu_data.columns if col.startswith('IC')]

stu_data_filtered = stu_data[stu_cols + ic_columns]
#T_tech_data = teacher_data[teacher_cols]
#Renaming data to make mor legible 
T_rename_dict = {
    'TC045Q05NA': 'T1', #Did you receive ICT training in initial teaching qualification?
    'TC045Q05NB': 'T2', #                ''                                     in CPD?
    'TC185Q05HA': 'T3', #How much do you need CPD for ICT skills? (self assessed)
    'TC021Q01NA': 'T4', #Are your required to do CPD?
    'TC199Q03HA': 'T5', #Can you craft good questions?
    'TC166Q01HA': 'T6a', #Have you taught students how to use keywords?
    'TC166Q02HA': 'T6b', #    ""      trust info on the internet?
    'TC166Q03HA': 'T6c', #    ""       compare different web pages and decide what info is relevant
    'TC166Q04HA': 'T6d', #    ""      undersatnd consequences of maiking info publicly available
    'TC166Q05HA': 'T6e', #    ""      use the short description below links in the lsit results of a search
    'TC166Q06HA': 'T6f', #    ""       how to detect whether information is biased
    'TC166Q07HA': 'T6g', #    ""      How to detect spam emails
    'TC184Q01HA': 'T7', #Does your school have digital device policy for teaching?
    'TC167Q01HA': 'T8', #Have you used digital device for teaching in the past month? (Only english teachers)
    'TC169Q01HA': 'T9a',#This school year how often have you used for teaching Tutorial software or practice programs?
    'TC169Q02HA': 'T9b', #                               ""                    Digital Learning Games?
    'TC169Q03HA': 'T9c', #                               ""                    Word Processors or presentation software?
    'TC169Q04HA': 'T9d', #                               ""                    Spreadsheets?
    'TC169Q05HA': 'T9e', #                               ""                    Multimedia porduction tools?
    'TC169Q06HA': 'T9f', #                               ""                    Concept mapping software (e.g. inspiration, webspiration)
    'TC169Q07HA': 'T9g', #                               ""                    Data logging and mointoring tools?
    'TC169Q08HA': 'T9h', #                               ""                    Simulations and modelling software?
    'TC169Q09HA': 'T9i', #                               ""                    Social Media?
    'TC169Q10HA': 'T9j', #                               ""                    Communication software e.g. email, blogs?
    'TC169Q11HA': 'T9k', #                               ""                    Computer Based ICT resources?
    'TC169Q12HA': 'T9l', #                               ""                    Interactive digital learning resources?
    'TC169Q13HA': 'T9m', #                               ""                    Graphing or drawing software?
    'TC169Q14HA': 'T9n', #                               ""                    E-portfolios?'
}

Stu_rename_dict = {

'ST005Q01TA': 'MumEd3a', #Highest level of schooling of mother up to 3A
'ST006Q01TA': 'MumEd6', #Does your mum have level 6 ISCED
'ST006Q02TA': 'MumEd5a', #       ""                5A
'ST006Q03TA': 'MumEd5b', #       ""                5b
'ST006Q04TA': 'MumEd4', #       ""                  4
'ST007Q01TA': 'DadEd3a', # Highest level of schooling of Father up to 3A
'ST008Q01TA': 'DadEd6', # Does your dad have level 6 ISCED?        
'ST008Q02TA': 'DadEd5a', #      ""              5A ISCED?  
'ST008Q03TA': 'DadEd5b', #      ""              5b ISCED?
'ST008Q04TA': 'DadEd4', #      ""              4 ISCED?    
'ST011Q01TA': 'IYHa', #In your home do you have a desk to study at?
'ST011Q02TA': 'IYHb', #            ""            a room of your own?
'ST011Q03TA': 'IYHc', #            ""            a quiet place to study
'ST011Q04TA': 'IYHd', #            ""            a computer you can use for school week
'ST011Q05TA': 'IYHe', #            ""            educational software
'ST011Q06TA': 'IYHf', #            ""            internet connection
'ST011Q07TA': 'IYHg', #            ""            Classic literature 
'ST011Q08TA': 'IYHh', #            ""            books of poetry
'ST011Q09TA': 'IYHi', #            ""            works of art
'ST011Q10TA': 'IYHj', #            ""            books to help you study
'ST011Q11TA': 'IYHk', #            ""            technical reference book
'ST011Q12TA': 'IYHl', #            ""            a dictionary
'ST011Q16NA': 'IYHm', #            ""            books on art music or design
'ST011D17TA': 'CSWIa', #            ""           dishwasher (England), TV streamign subscription (Scotland)
'ST011D18TA': 'CSWIb', #            ""           tumble dryer (England), a musical instrument (Scotland)
'ST011D19TA': 'CSWIc', #            ""           someone your parents pay to help around the house (Whole UK)
'ST012Q01TA': 'HMIYHa', # How many of many TVs in your home? 
'ST012Q02TA': 'HMIYHb', #     ""           Cars      ""    ?
'ST012Q03TA': 'HMIYHc', #     ""           bathrooms      ""    ?
'ST012Q05NA': 'HMIYHd', #     ""           smartphones    ?
'ST012Q06NA': 'HMIYHe', #     ""           Computers  ?
'ST012Q07NA': 'HMIYHf', #     ""           tablets      ""    ?
'ST012Q08NA': 'HMIYHg', #     ""           E-book readers     ""    ?
'ST012Q09NA': 'HMIYHh', #     ""           Musical instruments      ""    ?
'ST013Q01TA': 'HMIYHi', #     ""           books     ""    ?
'ST158Q01HA': 'TASa', # is the following taught at school: How to use keywords? 
'ST158Q02HA': 'TASb', #           ''                       Whether to trust info on the internet?
'ST158Q03HA': 'TASc', #           ''                       How to caompare different web pages?
'ST158Q04HA': 'TASd', #           ''                       Understand consequences of making info publicly available
'ST158Q05HA': 'TASe', #           ''                       How to use the short description below the links in the lsit results of a search?
'ST158Q06HA': 'TASf', #           ''                       Detect whether information is subjective or biased?
'ST158Q07HA': 'TASg', #           ''                       Detect phishing or spam emails?
'PA042Q01TA': 'HouseIn', #What is your annual househod income (Parents)?
}

Renamed_Ttech = teacher_data.rename(columns=T_rename_dict)

Renamed_stu = stu_data_filtered.rename(columns = Stu_rename_dict)
#Finding highest parentla education of each student
F.Mum_HIED(Renamed_stu)
F.Dad_HIED(Renamed_stu)
F.add_parent_education_column(Renamed_stu)

edu_to_years_dict = {'ISCED level 6': 16,
                     'ISCED level 5B': 14,
                     'ISCED level 5A': 16,
                     'ISCED level 4': 12,
                     'ISCED level 3B, 3C': 11,
                     'ISCED level 3A': 12,
                     'ISCED level 2': 10,
                     'He did not complete  ISCED level 1': 3,
                     'ISCED level 1': 7,
                     'She did not complete  ISCED level 1': 3
                     }
#Mapping highest parental education to ISCED years dictionary
Renamed_stu['ParYearsEdu'] = Renamed_stu['ParEd'].map(edu_to_years_dict)
 
Renamed_stu = Renamed_stu.apply(F.split_combined_questions, axis=1)  #split country specific wealth items for england and scotland

edu_cols =['MumEd3a','MumEd6', 'MumEd5a', 'MumEd5b', 'MumEd4', 'DadEd3a', 'DadEd6','DadEd5a',  'DadEd5b',  'DadEd4']
#Dropping Nan values
Renamed_stu = Renamed_stu.dropna(subset = edu_cols, thresh = 1)
#Dropping columns with poor data
cols_to_drop = ['CSWIa_ENG',
 'CSWIa_SCOT',
 'CSWIa',
 'CSWIc'
]
Renamed_stu = Renamed_stu.drop(columns = cols_to_drop) #no data for first country specific wealth item for scotland so dropped both, no data for 3rd country specific wealth item

poss_costs = {
                'Question': ["IYHa", "IYHb", "IYHc", "IYHd", "IYHe", "IYHf", "IYHg", "IYHh", "IYHi", "IYHj", "IYHk", "IYHl", "IYHm", "CSWIa_ENG", "CSWIa_SCOT", "CSWIb_ENG", "CSWIb_SCOT", "CSWIc", "HMIYHa", "HMIYHb", "HMIYHc", "HMIYHd", "HMIYHe", "HMIYHf", "HMIYHg", "HMIYHh", "HMIYHi"],
                'Item': ['Desk', 'A room of your own', 'A quiet place to study', 'A computer you can use for school work', 'Educational software', 'Internet connection', 'Classic literature', 'Books of poetry', 'Works of art', 'Books to help you study', 'Technical reference book', 'A dictionary', 'Books on art, music or design', 'Dishwasher', 'TV streaming subscription', 'Tumble dryer', 'A musical instrument', 'Someone your parents pay to help around the house', 'TV', 'Car', 'Bathroom', 'Smartphone', 'Computer', 'Tablet', 'E-book reader', 'Musical instrument', 'Books'],
                'Cost': [75, 24000, 11000, 850, 50, 3000, 10, 15, 100, 10, 30, 30, 30, 500, 1000, 425, 230, 35000, 400, 17000, 12000, 2000, 850, 525, 250, 230, 10]
    }



poss_costs = pd.DataFrame(poss_costs) 
scaler = MinMaxScaler()
poss_costs['normalised_costs'] = scaler.fit_transform(poss_costs[['Cost']]) #
poss_costs['log_cost'] = np.log(poss_costs['Cost'] + 0.0001) #spread of costs necesitates log scale gives importance to books and other low cost items


poss_columns = ['IYHa',
 'IYHb',
 'IYHc',
 'IYHd',
 'IYHe',
 'IYHf',
 'IYHg',
 'IYHh',
 'IYHi',
 'IYHj',
 'IYHk',
 'IYHl',
 'IYHm',
 'HMIYHa',
 'HMIYHb',
 'HMIYHc',
 'HMIYHd',
 'HMIYHe',
 'HMIYHf',
 'HMIYHg',
 'HMIYHh', 'CSWIb_ENG',
 'CSWIb_SCOT']

Renamed_stu = Renamed_stu.dropna(subset = poss_columns, thresh = 20) #Drop rows with more than 5 Nan values in poss columns 


poss_qus = Renamed_stu[poss_columns]

poss_qus.dropna(thresh = poss_qus.shape[1] -5)

#poss_qus['poss_score'] = poss_qus.apply(F.calculate_poss_score_top_20, axis =1, costs = poss_costs)

#poss_qus_melted = poss_qus.melt(var_name='Question', value_name = 'Response')

#calculating possession score for each student
poss_qus = poss_qus.copy()
poss_qus.loc[:, 'poss_score_log'] = poss_qus.apply(F.calculate_poss_score, axis=1, costs_df=poss_costs, cost_column='log_cost')
poss_qus.loc[:, 'poss_score_norm'] = poss_qus.apply(F.calculate_poss_score, axis=1, costs_df=poss_costs, cost_column='normalised_costs')

#poss_qus['poss_score_log'] = poss_qus.apply(F.calculate_poss_score, axis=1, costs_df=poss_costs, cost_column = 'log_cost')

#poss_qus['poss_score_norm'] = poss_qus.apply(F.calculate_poss_score, axis=1, costs_df=poss_costs, cost_column = 'normalised_costs')
#importing ISCO occupational status ranking
occupational_status = pd.read_csv('/Users/alexpeacock/Dropbox/Capstone_data_stuff/ISCO08_cleaned.csv')
#creating dictionary keys being jobs and values assocaited status
status_dict = {name: isei for name, isei in zip(occupational_status['Name'], occupational_status['ISEI-08'])}

#applying dictionary to give numeric value to each parents occupational status
Renamed_stu = Renamed_stu.assign(
    OS_SCORE1=Renamed_stu['OCOD1'].map(status_dict),
    OS_SCORE2=Renamed_stu['OCOD2'].map(status_dict)
)
#
Renamed_stu['OS_SCORE1'] = pd.to_numeric(Renamed_stu['OS_SCORE1'], errors = 'coerce')
Renamed_stu['OS_SCORE2'] = pd.to_numeric(Renamed_stu['OS_SCORE2'], errors = 'coerce')

import pandas as pd


#taking highest of either parental education and placing in new column 
Renamed_stu = F.create_par_os_score(Renamed_stu, 'OS_SCORE1', 'OS_SCORE2')

#isolating plausible values 

math_plaus_vals = [f'PV{i}MATH' for i in range(1, 11)]
read_plaus_vals = [f'PV{i}READ' for i in range(1, 11)]
scie_plaus_vals = [f'PV{i}SCIE' for i in range(1, 11)]
plaus_vals = math_plaus_vals + read_plaus_vals + scie_plaus_vals


replicate_weights = [col for col in Renamed_stu if col.startswith('W_FSTURWT')]

score_and_eduyears = Renamed_stu[['par_os_score', 'ParYearsEdu', 'CNT', 'CNTSCHID', 'STRATUM','W_FSTUWT', 'ESCS'] + plaus_vals + replicate_weights]


SES_df = pd.concat([score_and_eduyears , poss_qus[['poss_score_log']]], axis = 1)
#create student weighted plausible values
for pv in plaus_vals:
    new_col_name = f'{pv}_weighted'
    SES_df[new_col_name] = SES_df[pv] * SES_df['W_FSTUWT']



scaler = MinMaxScaler()

SES_df = SES_df.dropna()
#normalising all scores between 0-1 
SES_df['norm_par_os'] = scaler.fit_transform(SES_df[['par_os_score']])
SES_df['norm_poss_score'] = scaler.fit_transform(SES_df[['poss_score_log']])
SES_df['norm_ParYearsEdu'] = scaler.fit_transform(SES_df[['ParYearsEdu']])
#creating SES_score
SES_df['SES_Score'] = SES_df[['norm_par_os', 'norm_poss_score', 'norm_ParYearsEdu']].sum(axis=1)
#weigthing SES_score and normalising
SES_df['wghted_SES_Score'] = SES_df['SES_Score'] * SES_df['W_FSTUWT'] #weight to allow for sampling error
SES_df['wghted_SES_Score'] = scaler.fit_transform(SES_df[['wghted_SES_Score']])


#SES_df['wghted_SES_Score'].describe()

replicate_weights = [col for col in Renamed_stu if col.startswith('W_FSTURWT')]

replicate_weights_df = Renamed_stu[replicate_weights +['CNTSCHID']]

uw_mean = SES_df['SES_Score'].mean()

w_mean = SES_df['wghted_SES_Score'].mean()
#finding standard errorr by BRR
uw_standard_error = F.calculate_standard_error(uw_mean, replicate_weights_df, SES_df['SES_Score'])

w_standard_error = F.calculate_standard_error(w_mean, replicate_weights_df, SES_df['SES_Score']) #student weight included in replicate weight so no need to us wghted SES_score

usage_cols = [col for col in Renamed_Ttech if col.startswith('T9')]




scores = [3, 2, 1, 1, 3, 2, 3, 5, 3, 1, 1, 3, 2, 3] #ranking of 1-5 of ability required to use tool

usage_score_dict = dict(zip(usage_cols, scores))


value_mapping = {
    'Never': 0,
    'In some lessons': 1,
    'In most lessons': 2,
    'In every or almost every lesson': 3
}
#Finding tecaher ICT usage scores 
usage_score_df =pd.DataFrame(list(usage_score_dict.items()), columns = ['Questions', 'Score'])
#ans_vals_df = pd.DataFrame(value_mapping)

Renamed_Ttech[usage_cols] = Renamed_Ttech[usage_cols].replace(value_mapping)


#making usage score and dropping Nan values
dropna_df = Renamed_Ttech.dropna(subset = usage_cols)

dropna_df = dropna_df.assign(**{col: dropna_df[col] * usage_score_dict[col] for col in usage_cols})


dropna_df['usage_score'] = dropna_df[usage_cols].sum(axis = 1)

#mkainv spread score 
spread_cols = [col for col in Renamed_Ttech if col.startswith('T6')]

dropna_df = dropna_df.dropna(subset = spread_cols)

yes_no_dict = {'Yes':1,
               'No':0}


dropna_df.loc[:,spread_cols] = dropna_df[spread_cols].replace(yes_no_dict)


spread_scores = [1] * 7

spread_score_dict = dict(zip(spread_cols, spread_scores))

dropna_df = dropna_df.assign(**{col: dropna_df[col] * spread_score_dict[col] for col in spread_cols})
dropna_df['spread_score'] = dropna_df[spread_cols].sum(axis = 1)


#transforming spread and usgae and adding togtehr
dropna_df['spread1_5'] = scaler.fit_transform(dropna_df[['spread_score']])* 5

dropna_df['T_ICT_SCORE'] = dropna_df['spread1_5'] + dropna_df['usage_score']

dropna_df['T_ICT_SCORE'] = scaler.fit_transform(dropna_df[['T_ICT_SCORE']])

av_TICT_by_school = dropna_df.groupby('CNTSCHID')['T_ICT_SCORE'].mean()

av_SES_by_school = SES_df.groupby('CNTSCHID')['wghted_SES_Score'].mean()

av_Raw_SES_by_school = pd.DataFrame(SES_df.groupby('CNTSCHID')['SES_Score'].mean())
av_TICT_by_school = pd.DataFrame(av_TICT_by_school).reset_index()

av_SES_by_school = pd.DataFrame(av_SES_by_school).reset_index()

merged = av_SES_by_school.merge(av_TICT_by_school, on ='CNTSCHID', how = 'inner')
merged = merged.merge(av_Raw_SES_by_school, on = 'CNTSCHID', how = 'inner')

merged['wghted_SES_Score'] = pd.to_numeric(merged['wghted_SES_Score'], errors='coerce')
merged['T_ICT_SCORE'] = pd.to_numeric(merged['T_ICT_SCORE'], errors='coerce')



#for col in plaus_vals:
#    SES_df[col] = SES_df[col] * SES_df['W_FSTUWT']
weighted_plaus_vals = [col for col in SES_df.columns if 'weighted' in col]

#finding mean of plaus vals at school level rather than student level 
plaus_val_by_school = pd.DataFrame(SES_df.groupby('CNTSCHID')[plaus_vals].mean().reset_index())
weighted_plaus_val_by_school = pd.DataFrame(SES_df.groupby('CNTSCHID')[weighted_plaus_vals].mean().reset_index())


merged = merged.merge(plaus_val_by_school, on='CNTSCHID', how = 'inner')
merged = merged.merge(weighted_plaus_val_by_school, on = 'CNTSCHID', how = 'inner')

#regression with weighted plausible values for each subject
maths_SES_regression = F.calculate_average_statistics(merged, 'MATH')
reading_SES_regression = F.calculate_average_statistics(merged, 'READ')
scie_SES_regression = F.calculate_average_statistics(merged, 'SCIE')

math_cols = [col for col in merged.columns if 'MATH' in col]

scie_cols = [col for col in merged.columns if 'SCIE' in col]

reading_cols =[col for col in merged.columns if 'READ' in col]

weighted_read_cols = [col for col in merged.columns if 'READ' in col and 'weighted' in col]
weighted_math_cols = [col for col in merged.columns if 'MATH' in col and 'weighted' in col]
weighted_scie_cols = [col for col in merged.columns if 'SCIE' in col and 'weighted' in col]

SES_vs_pv_x = np.linspace(merged['wghted_SES_Score'].min(), merged['wghted_SES_Score'].max(), 100)
#regression lines for each plausible values with weigthed SES score and values, 
read_SES_y = (112489.51730583786* SES_vs_pv_x) + 1051.7495715307468
math_SES_y = (106154.90188502024 * SES_vs_pv_x) + 1190.6378311689634
scie_SES_y = (112952.00502662756 * SES_vs_pv_x) + 792.1843380270618
#dropping outliers
merged_cleaned = merged[merged['wghted_SES_Score'] != merged['wghted_SES_Score'].max()]
#x values for plot regression line
SES_vs_ICT_x = np.linspace(merged_cleaned['wghted_SES_Score'].min(), merged_cleaned['wghted_SES_Score'].max(), 100)

#
SES_TICT_regression = sps.linregress(merged_cleaned['wghted_SES_Score'], merged_cleaned['T_ICT_SCORE'])

y_SES_TICT = (SES_vs_pv_x * SES_TICT_regression.slope) + SES_TICT_regression.intercept
#replicate weights for each school for brr calculations 
unique_weights_df = replicate_weights_df.drop_duplicates(subset='CNTSCHID').reset_index(drop=True)


merged = merged.merge(unique_weights_df, on= 'CNTSCHID')
merged = merged.T.drop_duplicates().T




rep_weights_for_brr_df = merged[replicate_weights]
rep_weights_for_brr_df = rep_weights_for_brr_df.T.drop_duplicates().T

rep_weights_for_brr_df = rep_weights_for_brr_df.T
cf_SES_TICT = SES_TICT_regression.rvalue



#brr calculation for SES and ICT score regression

T_ICT_for_brr = np.array(merged['T_ICT_SCORE'])
raw_SES_for_brr = np.array(merged['SES_Score'])
replicate_weights_for_brr =  np.array(rep_weights_for_brr_df)
brr_error_TICT_SES = F.correlation_standard_error(cf_SES_TICT, T_ICT_for_brr, 
                                                  raw_SES_for_brr, replicate_weights_for_brr)