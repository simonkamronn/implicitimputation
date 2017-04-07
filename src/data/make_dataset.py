# -*- coding: utf-8 -*-
import os
import click
import logging
from dotenv import find_dotenv, load_dotenv
import pickle
import os
import numpy as np
import pandas as pd

from goactiwe import GoActiwe
from goactiwe.steps import remove_drops
import dask.dataframe as dd
from fastparquet import write, ParquetFile


def fill_df_with_datetime_vars(df):
    df['5min'] = df.index.hour * 12 + df.index.minute // 5
    df['quarter_hour'] = df.index.hour * 4 + df.index.minute // 15
    df['hour'] = df.index.hour
    df['day'] = df.index.weekday
    df['month'] = df.index.month
    df['date'] = df.index.date
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    return df


class DataLoader:
    def __init__(self, logger=None):
        self.log = logger.info or print
        self._ga = GoActiwe()
        self._location = self._ga.get_location()
        self._activity = self._ga.get_activity()
        self._steps = self._ga.get_steps()
        self._screen = self._ga.get_screen()

    @staticmethod
    def save_scalers(user, scalers):
        pickle.dump(scalers, open('scalers_{}.pkl'.format(user), 'w'))

    @staticmethod
    def load_scalers():
        return pickle.load(open('scalers.pkl'))

    @staticmethod
    def load_cpm(user):
        def load_smartphone_cpm(user):
            cpm_root_dir = '/home/sdka/data/cpm/smartphone/20170222110955/'
            df = dd.read_csv(os.path.join(cpm_root_dir, str(user), '*.csv'))
            df = df.compute().set_index('timestamp')
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

        cpm = load_smartphone_cpm(user)
        cpm = fill_df_with_datetime_vars(cpm)
        cpm = cpm.pivot_table(index='date', columns='5min', values='cpm', aggfunc=np.sum)
        return cpm

    def load_location_lat(self, user):
        location = self._location.copy()
        location = location[location.user == user]
        location = location[location.accuracy < 50]
        location = fill_df_with_datetime_vars(location)
        location_pt_lat = location.pivot_table(index='date', columns='5min', values='lat', aggfunc='median')
        return location_pt_lat

    def load_location_lon(self, user):
        location = self._location.copy()
        location = location[location.user == user]
        location = location[location.accuracy < 50]
        location = fill_df_with_datetime_vars(location)
        location_pt_lon = location.pivot_table(index='date', columns='5min', values='lon', aggfunc='median')
        return location_pt_lon

    def load_activity(self, user):
        activity = self._activity.copy()
        activity = activity[activity.user == user]
        activity = activity[activity.confidence > 70]
        activity = fill_df_with_datetime_vars(activity)
        activity = activity.pivot_table(index='date', columns='5min', values='activity', aggfunc='median')
        return activity

    def load_steps(self, user):
        steps = self._steps.copy()
        steps = steps[steps.user == user]
        steps = remove_drops(steps.step_count).to_frame()
        steps = steps[steps.step_count < steps.step_count.mean() + 2 * steps.step_count.std()]
        steps = fill_df_with_datetime_vars(steps)
        steps = steps.pivot_table(index='date', columns='5min', values='step_count', aggfunc='sum')
        return steps

    def load_screen(self, user):
        screen = self._screen.copy()
        screen = screen[screen.user == user]
        screen = screen.groupby(pd.TimeGrouper('15min')).filter(lambda x: x.screen_on.sum() < 20)
        screen = fill_df_with_datetime_vars(screen)
        screen = screen.pivot_table(index='date', columns='5min', values='screen_on', aggfunc='sum')
        return screen

    def collect_modalities_in_panel(self, user):
        cpm = self.load_cpm(user)
        steps = self.load_steps(user)
        activity = self.load_activity(user)
        screen = self.load_screen(user)
        location_lat, location_lon = self.load_location(user)

        # Limit data to inner bounds
        first_idx = np.max(pd.to_datetime((cpm.index[0],
                                           steps.index[0],
                                           activity.index[0],
                                           screen.index[0],
                                           location_lat.index[0]))).date()
        last_idx = np.min(pd.to_datetime((cpm.index[-1],
                                          steps.index[-1],
                                          activity.index[-1],
                                          screen.index[-1],
                                          location_lat.index[-1]))).date()
        common_idx = slice(first_idx.strftime('%Y-%m-%d'), last_idx.strftime('%Y-%m-%d'))

        # Return a 3D panel object
        return pd.Panel({'cpm': cpm.loc[common_idx],
                         'steps': steps.loc[common_idx],
                         'activity': activity.loc[common_idx],
                         'screen': screen.loc[common_idx],
                         'location_lat': location_lat.loc[common_idx],
                         'location_lon': location_lon.loc[common_idx]})

    def load_all(self):
        data = list()
        for user in self._ga.done:
            data.append(self.collect_modalities_in_panel(user).as_matrix())
        return np.concatenate(data, axis=0)

    def save_all(self):
        data_dir = os.path.join(project_dir, 'data', 'interim', 'data.parq')

        for att in ['cpm', 'steps', 'activity', 'screen', 'location_lat', 'location_lon']:
            self.log(att)
            list_of_frames = list()
            for user in self._ga.done:
                self.log(user)
                list_of_frames.append(getattr(self, 'load_{}'.format(att))(user).assign(user=user))

            df = pd.concat(list_of_frames).assign(modality=att)
            df.columns = [str(c) for c in df.columns]

            write(filename=data_dir,
                  data=df,
                  partition_on=['user', 'modality'],
                  has_nulls=True,
                  file_scheme='hive',
                  append=os.path.exists(data_dir))

    @staticmethod
    def convert_to_npy(df=None, save=True):
        if df is None:
            df = ParquetFile(os.path.join(project_dir, 'data', 'interim', 'data.parq')).to_pandas().set_index('date')

        user_data = list()
        for user, group in df.groupby(['user']):
            modality_data = list()
            modality_grouped = group.groupby('modality')
            for modality in ('cpm', 'steps', 'screen'):
                modality_data.append(modality_grouped.get_group(modality).drop(['modality'], axis=1))

            # We concatenate on dates to ensure the same dimension across modalities
            user_data.append(pd.concat(modality_data, axis=1).values
                             .reshape(-1, len(modality_data), 289)
                             .transpose(0, 2, 1))

        data = np.concatenate(user_data, axis=0)
        if save:
            np.save(os.path.join(project_dir, 'data', 'interim', 'data.npy'), data)

        return data


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    # dataloader = DataLoader(logger)
    # dataloader.save_all()

    DataLoader.convert_to_npy()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
