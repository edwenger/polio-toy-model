import os
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_villages():

    with open(os.path.join('output', 'initialization.json')) as json_file:
        data = json.loads(json_file.read())

    villages_df = pd.DataFrame.from_records(data).set_index('ix')
    logging.debug(villages_df.head())

    villages_df[['x', 'y']] = pd.DataFrame(villages_df['loc'].tolist(), index=villages_df.index)

    return villages_df


def plot_villages():

    villages_df = get_villages()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    villages_df.plot(kind='scatter', x='x', y='y', s=100*(np.log10(villages_df.N)-1)/3, ax=ax)

    ax.set(aspect='equal', yticks=[], xticks=[], ylabel='', xlabel='')
    fig.set_tight_layout = True


def get_summary():
    return pd.read_csv(os.path.join('output', 'summary.csv')).set_index(['generation', 'village'])


def get_generation(generation):

    summary_df = get_summary()

    generation_df = summary_df.ix[generation]

    village_locations = get_villages()[['x', 'y']]
    generation_df = generation_df.join(village_locations)
    logging.info(generation_df)

    return generation_df


def plot_snapshot(generation):

    generation_df = get_generation(generation)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    kwargs = dict(kind='scatter', x='x', y='y', ax=ax)

    generation_df.plot(s=200*(np.log10(generation_df.N)-1)/3,
                       c='gray', alpha=0.8, **kwargs)
    generation_df.plot(s=(generation_df.S/generation_df.N)*200*(np.log10(generation_df.N)-1)/3,
                       c='orange', alpha=0.8, **kwargs)
    generation_df.plot(s=generation_df.I,
                       c='firebrick', alpha=0.5, **kwargs)

    ax.set(aspect='equal', yticks=[], xticks=[], ylabel='', xlabel='', title='generation %d' % generation)
    fig.set_tight_layout = True


def plot_timeseries():

    summary_df = get_summary().reset_index()
    timeseries_df = summary_df.groupby('generation').sum()
    timeseries_df.I.plot()


if __name__ == '__main__':

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # plot_villages()

    plot_snapshot(generation=10)

    # plot_timeseries()

    plt.show()



