import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, List
from numbers import Number
from collections import defaultdict


@dataclass
class TimestampData:

    timestamp: np.ndarray
    data: np.ndarray

    def __post_init__(self):
        assert self.timestamp.ndim == 1
        assert len(self.timestamp) == len(self.data)


@dataclass
class CovidData:

    nb_samples: int = None
    nb_groups: int = None
    deaths: Union[None, TimestampData] = None
    recovered: Union[None, TimestampData] = None
    infected: Union[None, TimestampData] = None
    hospitalised: Union[None, TimestampData] = None
    critical: Union[None, TimestampData] = None

    def __post_init__(self):
        self._assert_shapes()

    def _assert_shapes(self):
        assert self.nb_samples > 0
        assert self.nb_groups > 0
        if self.deaths is not None:
            assert self.deaths.data.shape[1:] == (self.nb_groups, self.nb_samples)
        if self.recovered is not None:
            assert self.recovered.data.shape[1:] == (self.nb_groups, self.nb_samples)
        if self.infected is not None:
            assert self.infected.data.shape[1:] == (self.nb_groups, self.nb_samples)
        if self.hospitalised is not None:
            assert self.hospitalised.data.shape[1:] == (self.nb_groups, self.nb_samples)
        if self.critical is not None:
            assert self.critical.data.shape[1:] == (self.nb_groups, self.nb_samples)

    def all_timestamps(self):
        timestamps = []
        if self.deaths:
            timestamps.append(self.deaths.timestamp)
        if self.recovered:
            timestamps.append(self.recovered.timestamp)
        if self.infected:
            timestamps.append(self.infected.timestamp)
        if self.hospitalised:
            timestamps.append(self.hospitalised.timestamp)
        if self.critical:
            timestamps.append(self.critical.timestamp)
        output = np.concatenate(timestamps)
        output = np.unique(output)
        output = np.sort(output, axis=None)
        return output

    def to_dataframe(self,
                     group_labels=None,
                     group_total: bool = False,
                     timestamp_shift: Union[datetime.datetime, float, int] = None):
        if group_labels is None:
            if group_total:
                group_labels = ['total']
            else:
                group_labels = [f'{i}' for i in range(self.nb_groups)]
        else:
            assert len(group_labels) == self.nb_groups

        names = ['deaths', 'critical', 'hospitalised', 'infected', 'recovered']
        td_data = [self.deaths, self.critical, self.hospitalised, self.infected, self.recovered]

        convert = defaultdict(lambda: {})
        for name, td in zip(names, td_data):
            if td is not None:
                for t_idx, t in enumerate(td.timestamp):
                    d = td.data
                    if group_total:
                        d = np.sum(d, axis=1, keepdims=True)
                    add_dict = {}
                    for i in range(d.shape[1]):
                        add_dict.update({
                            f'{name}_median_{group_labels[i]}': np.median(d[t_idx, i], axis=-1),
                            f'{name}_2.5CI_{group_labels[i]}': np.percentile(d[t_idx, i], 2.5, axis=-1),
                            f'{name}_97.5CI_{group_labels[i]}': np.percentile(d[t_idx, i], 97.5, axis=-1)
                        })
                    if timestamp_shift is not None:
                        if isinstance(timestamp_shift, datetime.datetime):
                            t = timestamp_shift + datetime.timedelta(days=float(t))
                        else:
                            t = t + timestamp_shift
                    convert[t].update(add_dict)

        return pd.DataFrame.from_dict(convert).transpose()


    def plot(self,
             axes: List[plt.axes] = None,
             plot_daily_deaths: bool = True,
             plot_daily_infected: bool = True,
             plot_daily_recovered: bool = True,
             group_total=False,
             group_labels: list = None,
             timestamp_shift: Union[datetime.datetime, float, int] = None,
             plot_fmt=None,
             plot_kwargs=None):
        req_plots = 5
        if plot_daily_deaths:
            req_plots += 1
        if plot_daily_infected:
            req_plots += 1
        if plot_daily_recovered:
            req_plots += 1

        if axes is None:
            fig, axes = plt.subplots(1, req_plots, figsize=(3*req_plots, 3))
        if group_labels is None:
            if group_total:
                group_labels = [None]
            else:
                group_labels = [None] * self.nb_groups
        if plot_kwargs is None:
            plot_kwargs = {}
        if plot_fmt is None:
            plot_fmt = []
        if not isinstance(plot_fmt, list):
            plot_fmt = [plot_fmt]

        titles = ['Deaths', 'Critical Care Need', 'Hospital Need', 'Infected', 'Recovered']
        data = [self.deaths, self.critical, self.hospitalised, self.infected, self.recovered]
        plot_data = [x.data if x is not None else None for x in data]
        times = [x.timestamp if x is not None else None for x in data]
        if group_total:
            plot_data = [np.sum(x, axis=1, keepdims=True) if x is not None else None for x in plot_data]
        if plot_daily_deaths and data[0] is not None:
            plot_data.append(np.diff(plot_data[0], axis=0) / np.expand_dims(np.diff(times[0]), axis=(1, 2)))
            times.append(times[0][1:])
            titles.append('Daily Deaths')
        if plot_daily_infected and data[3] is not None:
            plot_data.append(np.diff(plot_data[3], axis=0) / np.expand_dims(np.diff(times[3]), axis=(1, 2)))
            times.append(times[3][1:])
            titles.append('Daily Infected')
        if plot_daily_recovered and data[4] is not None:
            plot_data.append(np.diff(plot_data[4], axis=0) / np.expand_dims(np.diff(times[4]), axis=(1, 2)))
            times.append(times[4][1:])
            titles.append('Daily Recovered')

        if timestamp_shift is not None:
            if isinstance(timestamp_shift, datetime.datetime):
                times = [np.array([timestamp_shift + datetime.timedelta(days=float(v)) for v in x]) if x is not None
                         else None for x in times]
            else:
                times = [x + timestamp_shift if x is not None else None for x in times]

        for i, ax in enumerate(axes.flat):
            x = times[i]
            y = plot_data[i]
            ax.set_title(titles[i])
            if y is not None:
                for j in range(y.shape[1]):
                    if self.nb_samples > 1:
                        median = np.percentile(y[:, j], 50, axis=-1)
                        low = np.percentile(y[:, j], 2.5, axis=-1)
                        high = np.percentile(y[:, j], 97.5, axis=-1)
                        ax.fill_between(x, low, high, alpha=.2, facecolor=f"C{j}")
                        ax.plot(x, median, *plot_fmt, label=group_labels[j], **plot_kwargs)
                    else:
                        ax.plot(x, y[:, j, 0], *plot_fmt, label=group_labels[j], **plot_kwargs)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        return axes


def extend_data_samples(a: CovidData, b: CovidData):
    # TODO: Investigate using iter methods to make this easier?
    deaths_data = None
    recovered_data = None
    infected_data = None
    hospitalised_data = None
    critical_data = None
    if a.deaths is not None:
        assert b.deaths is not None
        assert np.all(a.deaths.timestamp == b.deaths.timestamp)
        deaths_data = np.concatenate([a.deaths.data, b.deaths.data], axis=-1)
    if a.recovered is not None:
        assert b.recovered is not None
        assert np.all(a.recovered.timestamp == b.recovered.timestamp)
        recovered_data = np.concatenate([a.recovered.data, b.recovered.data], axis=-1)
    if a.infected is not None:
        assert b.infected is not None
        assert np.all(a.infected.timestamp == b.infected.timestamp)
        infected_data = np.concatenate([a.infected.data, b.infected.data], axis=-1)
    if a.hospitalised is not None:
        assert b.hospitalised is not None
        assert np.all(a.hospitalised.timestamp == b.hospitalised.timestamp)
        hospitalised_data = np.concatenate([a.hospitalised.data, b.hospitalised.data], axis=-1)
    if a.critical is not None:
        assert b.critical is not None
        assert np.all(a.critical.timestamp == b.critical.timestamp)
        critical_data = np.concatenate([a.critical.data, b.critical.data], axis=-1)

    nb_samples = deaths_data.shape[-1]
    nb_groups = deaths_data.shape[-2]

    if deaths_data is not None:
        deaths_data = TimestampData(a.deaths.timestamp, deaths_data)
    if recovered_data is not None:
        recovered_data = TimestampData(a.recovered.timestamp, recovered_data)
    if infected_data is not None:
        infected_data = TimestampData(a.infected.timestamp, infected_data)
    if hospitalised_data is not None:
        hospitalised_data = TimestampData(a.hospitalised.timestamp, hospitalised_data)
    if critical_data is not None:
        critical_data = TimestampData(a.critical.timestamp, critical_data)

    return CovidData(
        nb_samples=nb_samples,
        nb_groups=nb_groups,
        deaths=deaths_data,
        recovered=recovered_data,
        infected=infected_data,
        hospitalised=hospitalised_data,
        critical=critical_data
    )


def append_data_time(a: CovidData, b:CovidData):
    assert a.nb_samples == b.nb_samples
    assert a.nb_groups == b.nb_groups
    attributes = ['deaths', 'recovered', 'infected', 'hospitalised', 'critical']
    kwargs = {}
    for attr in attributes:
        attr_a = getattr(a, attr)
        attr_b = getattr(b, attr)
        if attr_a is not None:
            kwargs[attr] = a
            if attr_b is not None:
                kwargs[attr] = TimestampData(np.concatenate([attr_a.timestamp, attr_b.timestamp], axis=0),
                                             np.concatenate([attr_a.data, attr_b.data], axis=0))
        elif attr_b is not None:
            kwargs[attr] = b
        else:
            kwargs[attr] = None
    return CovidData(
        nb_samples=a.nb_samples,
        nb_groups=b.nb_groups,
        **kwargs
    )


@dataclass
class DsfsiData(CovidData):

    province: str = 'total'
    filter_kwargs: dict = field(default_factory=lambda: {})
    nb_samples: int = field(init=False)
    nb_groups: int = field(init=False)
    lockdown_date: datetime.datetime = field(init=False)
    country = 'ZAR'

    def __post_init__(self):
        self.nb_samples = 1
        self.nb_groups = 1
        self.lockdown_date = self._get_lockdown_date()
        assert self.province in ['EC', 'FS', 'GP', 'KZN', 'LP', 'MP', 'NC', 'NW', 'WC', 'total'], \
            f"Given 'province' should be one of ['EC', 'FS', 'GP', 'KZN', 'LP', 'MP', 'NC', 'NW', 'WC'], or 'total'. " \
            f"Got {self.province} instead."
        self.deaths = self._get_dsfsi_data(
            url='https://raw.githubusercontent.com/dsfsi/covid19za/master/data'
                '/covid19za_provincial_cumulative_timeline_deaths.csv'
        )
        self.infected = self._get_dsfsi_data(
            url='https://raw.githubusercontent.com/dsfsi/covid19za/master/data'
                '/covid19za_provincial_cumulative_timeline_confirmed.csv'
        )
        self.recovered = self._get_dsfsi_data(
            url='https://raw.githubusercontent.com/dsfsi/covid19za/master/data'
                '/covid19za_provincial_cumulative_timeline_recoveries.csv'
        )
        self._assert_shapes()

    def _get_lockdown_date(self, lockdown_path: str = 'data/country_lockdown_dates.csv') -> datetime.datetime:
        df_lockdown_dates = pd.read_csv(lockdown_path)
        lockdown_date = pd.to_datetime(
            df_lockdown_dates[df_lockdown_dates['country'] == self.country].date.values[0],
            format='%Y/%m/%d'
        )
        return lockdown_date

    def _get_dsfsi_data(self, url: str) -> TimestampData:
        df = pd.read_csv(
            url,
            parse_dates=['date'],
            date_parser=lambda t: pd.to_datetime(t, format='%d-%m-%Y')
        )

        df = df[['date', self.province]]
        df = df.dropna()

        min_date = _parse_min_max_date(df, self.filter_kwargs.get('min_date', None))
        max_date = _parse_min_max_date(df, self.filter_kwargs.get('max_date', None))

        if not min_date:
            min_date = df['date'].min()
        if not max_date:
            max_date = df['date'].max()

        df = df[df['date'] >= min_date]
        df = df[df['date'] <= max_date]

        timestamp = df['date']
        timestamp = (timestamp - self.lockdown_date).dt.days
        timestamp = timestamp.values

        data = df[self.province].astype(int).values
        data = np.expand_dims(data, axis=(1, 2))

        return TimestampData(timestamp, data)


def _parse_min_max_date(df, date) -> datetime.datetime:
    if date and isinstance(date, Number):
        if date < 0:
            date = df['date'].max() + datetime.timedelta(days=date)
        else:
            date = df['date'].min() + datetime.timedelta(days=date)
    if date and not isinstance(date, datetime.datetime):
        raise ValueError(f'Error in parsing date {date}')
    return date
