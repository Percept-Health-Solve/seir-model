import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime

sns.set(style='whitegrid')
mpl.rcParams['figure.figsize'] = (18, 20)
mpl.rcParams['figure.dpi'] = 100

actual_infections = pd.read_excel('../../Calibration.xlsx',sheet_name='Confirmed cases')
actual_infections['Date'] = [pd.to_datetime(x).date() for x in actual_infections['Date']]
actual_hospitalisations = pd.read_excel('../../Calibration.xlsx',sheet_name='Hospitalisations')
actual_hospitalisations['Date'] = [pd.to_datetime(x).date() for x in actual_hospitalisations['Date']]
actual_deaths = pd.read_excel('../../Calibration.xlsx',sheet_name='Deaths')
actual_deaths['Date'] = [pd.to_datetime(x).date() for x in actual_deaths['Date']]

scenario_descr = 'Scenario 1: 5 week lockdown, social distancing = 0.7 R0 post \n \
                  Scenario 2: 7 week lockdown, social distancing = 0.7 R0 post \n \
                  Scenario 3: 5 week lockdown, social distancing = 0.6 R0 post \n \
                  Scenario 4: 5 week lockdown, social distancing = 0.8 R0 post'

num_scenarios = 4

for i in range(num_scenarios):
  df = pd.read_csv('data/daily_scenario_' + str(i+1) +'.csv')
  df['Day'] = [pd.to_datetime(x).date() for x in df['Day']]
  df['Total hospitalised'] = [a+b for a,b in zip(df['Hospitalised'],df['ICU'])]
  if i==0:
    df_list = [df]
  else:
    df_list.append(df)


fig, axes = plt.subplots(5, 3)

for i in range(num_scenarios):

  axes[0, 0].plot(
      df_list[i]['Day'][:46],
      df_list[i]['Cumulative Infections'][:46],
      label='Cumulative infections Scenario ' + str(i+1)
  )
  axes[0, 1].plot(
      df_list[i]['Day'][:91],
      df_list[i]['Cumulative Infections'][:91],
      label='Cumulative infections Scenario ' + str(i+1)
  )
  axes[0, 2].plot(
      df_list[i]['Day'],
      df_list[i]['Cumulative Infections'],
      label='Cumulative infections Scenario ' + str(i+1)
  )

  axes[1, 0].plot(
      df_list[i]['Day'][:46],
      df_list[i]['Cumulative Detected'][:46],
      label='Cumulative detected Scenario ' + str(i+1)
  )
  axes[1, 1].plot(
      df_list[i]['Day'][:91],
      df_list[i]['Cumulative Detected'][:91],
      label='Cumulative detected Scenario ' + str(i+1)
  )
  axes[1,2].plot(
      df_list[i]['Day'],
      df_list[i]['Cumulative Detected'],
      label='Cumulative detected Scenario ' + str(i+1)
  )

  axes[2, 0].plot(
      df_list[i]['Day'][:46],
      df_list[i]['Total hospitalised'][:46],
      label='Total hospitalised Scenario ' + str(i+1)
  )
  axes[2, 1].plot(
      df_list[i]['Day'][:91],
      df_list[i]['Total hospitalised'][:91],
      label='Total hospitalised Scenario ' + str(i+1)
  )
  axes[2, 2].plot(
      df_list[i]['Day'],
      df_list[i]['Total hospitalised'],
      label='Total hospitalised Scenario ' + str(i+1)
  )

  axes[3, 0].plot(
      df_list[i]['Day'][:46],
      df_list[i]['ICU'][:46],
      label='ICU Scenario ' + str(i+1)
  )
  axes[3, 1].plot(
      df_list[i]['Day'][:91],
      df_list[i]['ICU'][:91],
      label='ICU Scenario ' + str(i+1)
  )
  axes[3, 2].plot(
      df_list[i]['Day'],
      df_list[i]['ICU'],
      label='ICU Scenario ' + str(i+1)
  )

  axes[4, 0].plot(
      df_list[i]['Day'][:46],
      df_list[i]['Dead'][:46],
      label='Cum. deaths Scenario ' + str(i+1)
  )
  axes[4, 1].plot(
      df_list[i]['Day'][:91],
      df_list[i]['Dead'][:91],
      label='Cum. deaths Scenario ' + str(i+1)
  )
  axes[4, 2].plot(
      df_list[i]['Day'],
      df_list[i]['Dead'],
      label='Cum. deaths Scenario ' + str(i+1)
  )

for x in [0,1]:
  axes[1, x].plot(
      actual_infections['Date'],
      actual_infections['Cum. Confirmed'],
      label='Reported cases'
  )
  axes[2, x].plot(
      actual_hospitalisations['Date'],
      actual_hospitalisations['Private hospital'],
      label='Reported private'
  )
  axes[2, x].plot(
      actual_hospitalisations['Date'],
      actual_hospitalisations['Est. total hospital'],
      label='Estimated total'
  )
  axes[3, x].plot(
      actual_hospitalisations['Date'],
      actual_hospitalisations['Private ICU'],
      label='Reported ICU private'
  )
  axes[3, x].plot(
      actual_hospitalisations['Date'],
      actual_hospitalisations['Est. total ICU'],
      label='Estimated ICU total'
  )
  axes[4, x].plot(
      actual_deaths['Date'],
      actual_deaths['Cum. Deaths'],
      label='Reported deaths'
  )




for i in range(3):

  axes[0, i].set_ylabel("Infections")
  axes[0, i].legend()
  axes[1, i].set_ylabel("Detected infections")
  axes[1, i].legend()
  axes[2, i].set_ylabel("Total hospitalisations")
  axes[2, i].legend()
  axes[3, i].set_ylabel("ICU")
  axes[3, i].legend()
  axes[4, i].set_ylabel("Cumulative deaths")
  axes[4, i].legend()

for i in range(5):

  axes[i, 0].set_xticks((datetime.date(2020,3,5),datetime.date(2020,3,12),datetime.date(2020,3,20),datetime.date(2020,3,27),
                          datetime.date(2020,4,4),datetime.date(2020,4,11),datetime.date(2020,4,19)))
  axes[i, 0].set_xticklabels(('05-Mar', '12-Mar', '20-Mar', '27-Mar', '04-Apr', '11-Apr', '19-Apr'))
  axes[i, 1].set_xticks((datetime.date(2020,3,5),datetime.date(2020,3,20),datetime.date(2020,4,4),datetime.date(2020,4,19),
                          datetime.date(2020,5,4),datetime.date(2020,5,19),datetime.date(2020,6,3)))
  axes[i, 1].set_xticklabels(('05-Mar', '20-Mar', '04-Apr', '19-Apr', '04-May', '19-May', '03-Jun'))
  axes[i, 2].set_xticks((datetime.date(2020,3,5),datetime.date(2020,4,24),datetime.date(2020,6,13),datetime.date(2020,8,2),
                          datetime.date(2020,9,21),datetime.date(2020,11,10),datetime.date(2020,12,30)))
  axes[i, 2].set_xticklabels(('05-Mar', '24-Apr', '13-Jun', '02-Aug', '21-Sep', '10-Nov', '30-Dec'))

fig.suptitle(scenario_descr)

plt.show()
fig.savefig('data/Scenario_result_plot.png')
