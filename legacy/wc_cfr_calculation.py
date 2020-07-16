import pandas as pd

if __name__ == '__main__':
    df_wc = pd.read_csv('data/202005 WCDoH Covid19 admissions data v3.csv',
                        parse_dates=['date_of_diagnosis', 'Admission_date', 'discharge_date', 'Date_of_ICU_admission'])

    # only consider records that are discharged or have died
    df_wc = df_wc[(df_wc['admission_status'] == 'Discharged') | (df_wc['admission_status'] == 'Died')]

    # define age groups
    age_groups = [['0 - 5', '5 - 10'], ['10 - 15', '15 - 20'], ['20 - 25', '25 - 30'], ['30 - 35', '35 - 40'],
                  ['40 - 45', '45 - 50'], ['50 - 55', '55 - 60'], ['60 - 65', '65 - 70'], ['70 - 75', '75 - 80'],
                  ['80 - 85', '85 - 90', '90 - 95', '95 - 100']]
    labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']

    # define stats dict for findings
    stats = {}

    # calculate cfr as ratio of sizes between those that have died and all cases
    cfr = lambda df: df[df['admission_status'] == 'Died'].size / df.size

    # overall cfr
    stats['overall'] = cfr(df_wc)

    # calculate cfr for different age groups
    for label, group in zip(labels, age_groups):
        filter = df_wc['agegroup'] == group[0]
        for i in range(1, len(group)):
            filter = (filter) | (df_wc['agegroup'] == group[i])
        stats[label] = cfr(df_wc[filter])

    # save stats to csv
    df_stats = pd.DataFrame.from_dict(stats, orient='index', columns=['CFR'])
    df_stats.index.name = 'age_group'
    df_stats.to_csv('data/cfr_stats.csv')

