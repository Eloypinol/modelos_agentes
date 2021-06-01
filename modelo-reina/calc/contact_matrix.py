import pandas as pd

df_original = pd.read_csv('data/contact_matrix.csv')
df_nuria = pd.read_csv('data/contact_matrix_nuria.csv', index_col='Row Labels')

df_original = df_original.loc[df_original['country'] == 'IT']
df_nuria = df_nuria.applymap(lambda x: float(x[:-1]) / 100.0)

df = pd.DataFrame(
    columns=['18to20', '21to29', '30to39', '40to49', '50to59', '60to69', '70to79', '80more'],
    index=['0to4', '5to9', '10to20', '20to49', '50plus'])

d = df_original.loc[df_original['participant_age'] == '0-4', '15-19']
print(sum(d))
exit()
df.loc['0to4', '18to20'] = sum(df_original.loc[df_original['participant_age'] == '0-4', '15-19'])
df.loc['0to4', '21to29'] = sum(df_original.loc[df_original['participant_age'] == '0-4', '20-24']) + \
                            sum(df_original.loc[df_original['participant_age'] == '0-4', '25-29'])
df.loc['0to4', '30to39'] = sum(df_original.loc[df_original['participant_age'] == '0-4', '30-34']) + \
                            sum(df_original.loc[df_original['participant_age'] == '0-4', '35-39'])


print(df.shape)