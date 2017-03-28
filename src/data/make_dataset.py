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
from goactiwe.preprocessing import MinMaxNormalize
import dask.dataframe as dd


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
    def __init__(self):
        self._ga = GoActiwe()
        self._location = self._ga.get_location()
        self._activity = self._ga.get_activity()
        self._steps = self._ga.get_steps()
        self._screen = self._ga.get_screen()

    @staticmethod
    def save_scalers(scalers):
        pickle.dump(open('scalers.pkl', 'w'), scalers)

    @staticmethod
    def load_scalers():
        return pickle.load(open('scalers.pkl'))

    @staticmethod
    def load_cpm(user):
        def load_smartphone_cpm(user):
            cpm_root_dir = 'data/cpm/smartphone/20170222110955/'
            df = dd.read_csv(os.path.join(cpm_root_dir, str(user), '*.csv'))
            df = df.compute().set_index('timestamp')
            df.index = pd.to_datetime(df.index)
            return df.sort_index()

        cpm = load_smartphone_cpm(user)
        cpm = fill_df_with_datetime_vars(cpm)

        cpm = cpm.pivot_table(index='date', columns='5min', values='cpm', aggfunc=np.sum)
        cpm_scaler = MinMaxNormalize()
        cpm = cpm_scaler.fit_transform(cpm)
        return cpm, cpm_scaler

    def load_location(self, user):
        location = self._location.copy()
        location = location[location.user == user]
        location = location[location.accuracy < 50]
        location = fill_df_with_datetime_vars(location)

        location_pt_lon = location.pivot_table(index='date', columns='5min', values='lon', aggfunc='median')
        location_pt_lat = location.pivot_table(index='date', columns='5min', values='lat', aggfunc='median')

        location_lon_scaler = MinMaxNormalize()
        location_lat_scaler = MinMaxNormalize()

        return location_lat_scaler.fit_transform(location_pt_lat), \
               location_lon_scaler.fit_transform(location_pt_lon), \
               location_lat_scaler, location_lon_scaler

    def load_activity(self, user):
        activity = self._activity.copy()
        activity = activity[activity.user == user]
        activity = activity[activity.confidence > 70]
        activity = fill_df_with_datetime_vars(activity)
        activity = activity.pivot_table(index='date', columns='5min', values='activity', aggfunc='median')
        activity_scaler = MinMaxNormalize()
        return activity_scaler.fit_transform(activity), activity_scaler

    def load_steps(self, user):
        steps = self._steps.copy()
        steps = steps[steps.user == user]
        steps = remove_drops(steps.step_count).to_frame()
        steps = steps[steps.step_count < steps.step_count.mean() + 2 * steps.step_count.std()]
        steps = fill_df_with_datetime_vars(steps)
        steps = steps.pivot_table(index='date', columns='5min', values='step_count', aggfunc='sum')
        steps_scaler = MinMaxNormalize()
        return steps_scaler.fit_transform(steps), steps_scaler

    def load_screen(self, user):
        screen = self._screen.copy()
        screen = screen[screen.user == user]
        screen = screen.groupby(pd.TimeGrouper('15min')).filter(lambda x: x.screen_on.sum() < 20)
        screen = fill_df_with_datetime_vars(screen)
        screen = screen.pivot_table(index='date', columns='5min', values='screen_on', aggfunc='sum')
        screen_scaler = MinMaxNormalize()
        return screen_scaler.fit_transform(screen), screen_scaler

    def collect_modalities(self, user):
        cpm, cpm_scaler = self.load_cpm(user)
        steps, step_scaler = self.load_steps(user)
        activity, activity_scaler = self.load_activity(user)
        screen, screen_scaler = self.load_screen(user)
        location_lat, location_lon, lat_scaler, lon_scaler = self.load_location(user)

        # Save scalers for later use
        self.save_scalers([cpm_scaler, step_scaler, activity_scaler, screen_scaler, lat_scaler, lon_scaler])

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
            data.append(self.collect_modalities(user).as_matrix())

        return np.concatenate(data, axis=0)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    dataloader = DataLoader()
    data = dataloader.load_all()
    np.save(os.path.join(os.pardir, os.pardir, 'data', 'processed', 'data.npy'), data)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
