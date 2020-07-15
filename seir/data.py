import numpy as np
import pandas as pd
import datetime

from dataclasses import dataclass, field
from typing import Union
from numbers import Number


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
