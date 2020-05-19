import logging
import argparse

import pandas as pd

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/202005 WCDoH Covid19 admissions data v3.csv',
                    help='Location of the data')


def main():
    # parse args
    args = parser.parse_args()
    data_path = Path(args.data_path)

    # check args
    assert data_path.exists() and data_path.is_file(), \
        f"Given data file '{args.data_file}' does not exist or is not a file."

    # setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -- %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    # load data
    logging.info(f'Loading data from {args.data_path}')

    date_cols = ['date_of_diagnosis', 'discharge_date', 'Date_of_ICU_admission', 'Admission_date']
    df_WC = pd.read_csv(data_path,
                        parse_dates=date_cols)

    # construct date range
    min_date = df_WC[date_cols].min().min()
    max_date = df_WC[date_cols].max().max()
    logging.info(f'Constructing data from {min_date} to {max_date}')
    date_range = pd.date_range(
        start=min_date,
        end=max_date
    )

    # prepare output df
    df_out = pd.DataFrame({'date': date_range, 'Current Hospitalisations': 0, 'Current ICU': 0,
                           'Cum Deaths': 0, 'Cum Recoveries': 0})

    logging.info('Calculating...')
    for date in date_range:
        df_hosp_current = df_WC.apply(current_hospital_patient, axis=1, date=date)
        df_icu_current = df_WC.apply(current_icu_patient, axis=1, date=date)
        df_deaths = df_WC.apply(current_deaths, axis=1, date=date)
        df_recoveries = df_WC.apply(current_recoveries, axis=1, date=date)

        df_out.loc[df_out['date'] == date, 'Current Hospitalisations'] = df_hosp_current.sum()
        df_out.loc[df_out['date'] == date, 'Current ICU'] = df_icu_current.sum()
        df_out.loc[df_out['date'] == date, 'Cum Deaths'] = df_deaths.sum()
        df_out.loc[df_out['date'] == date, 'Cum Recoveries'] = df_recoveries.sum()

    save_path = Path('data/WC_data.csv')
    logging.info(f"Saving to data to '{save_path}'")
    df_out.to_csv(save_path, index=False)


def current_hospital_patient(row, date):
    hospital_case = False
    if row['Admission_date'] <= date and (row['discharge_date'] >= date or pd.isna(row['discharge_date'])):
        # have a valid inpatient for this date, check if they are in hospital or in ICU
        if row['Admitted_to_ICU'] == 'No' or date <= row['Date_of_ICU_admission']:
            hospital_case = True
    return hospital_case


def current_icu_patient(row, date):
    icu_case = False
    if row['Admission_date'] <= date and (row['discharge_date'] >= date or pd.isna(row['discharge_date'])):
        # have a valid inpatient for this date, check if they are in hospital or in ICU
        if row['Admitted_to_ICU'] == 'Yes' and date >= row['Date_of_ICU_admission']:
            icu_case = True
    return icu_case


def current_deaths(row, date):
    death = False
    if row['discharge_date'] < date and row['admission_status'] == 'Died':
        death = True
    return death


def current_recoveries(row, date):
    recovery = False
    if row['discharge_date'] < date and row['admission_status'] == 'Discharged':
        recovery = True
    return recovery


if __name__ == '__main__':
    main()
