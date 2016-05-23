#!/usr/bin/env python
"""Script for Expedia Hotel Recommendations Kaggle competition.
Particular verison is based on Dataquest tutorial by Vik Paruchuri
<https://www.dataquest.io/blog/kaggle-tutorial/>

Main idea here lies in fiding most popular hotel clusters
in the training data for all the data and for each particular
`destination id`. Also it uses some information from data leak.
"""

import operator

import numpy as np
import pandas as pd


# Helper functions
def read_csv(filename, cols=None, nrows=None):
    """Data import and basic mangling.

    Parameters
    ----------
    filename : str, file name of the CSV file
    cols : array-like, default None
        Return a subset of columns.
    nrows : int, default None
        Number of rows of file to read.

    Returns
    -------
    result : DataFrame
    """

    datecols = ['date_time', 'srch_ci', 'srch_co']
    dateparser = lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S',
                                          errors='coerce')
    dtypes = {
        'id': np.uint32,
        'site_name': np.uint8,
        'posa_continent': np.uint8,
        'user_location_country': np.uint16,
        'user_location_region': np.uint16,
        'user_location_city': np.uint16,
        'orig_destination_distance': np.float32,
        'user_id': np.uint32,
        'is_mobile': bool,
        'is_package': bool,
        'channel': np.uint8,
        'srch_adults_cnt': np.uint8,
        'srch_children_cnt': np.uint8,
        'srch_rm_cnt': np.uint8,
        'srch_destination_id': np.uint32,
        'srch_destination_type_id': np.uint8,
        'is_booking': bool,
        'cnt': np.uint64,
        'hotel_continent': np.uint8,
        'hotel_country': np.uint16,
        'hotel_market': np.uint16,
        'hotel_cluster': np.uint8,
    }

    df = pd.read_csv(
        filename,
        nrows=nrows,
        usecols=cols,
        dtype=dtypes,
        parse_dates=[col for col in datecols if col in cols],
        date_parser=dateparser,
    )

    if 'date_time' in df.columns:
        df['month'] = df['date_time'].dt.month.astype(np.uint8)
        df['year'] = df['date_time'].dt.year.astype(np.uint16)

    return df


def find_most_common(df):
    """Find the most common hotel clusters in the whole dataset.
    """
    return list(df['hotel_cluster'].value_counts().head().index)


def make_key(items):
    return "_".join([str(i) for i in items])


def find_most_common_in_match(data_frame, match_cols):
    """Find the most common hotel clusters for each destination.
    """

    cluster_cols = match_cols + ['hotel_cluster']
    groups = data_frame.groupby(cluster_cols)

    top_clusters = {}
    for name, group in groups:
        bookings = group['is_booking'].sum()
        clicks = len(group) - bookings

        score = bookings + .15*clicks

        clus_name = make_key(name[:len(match_cols)])
        if clus_name not in top_clusters:
            top_clusters[clus_name] = {}
        top_clusters[clus_name][name[-1]] = score

    cluster_dict = {}
    for n in top_clusters:
        tc = top_clusters[n]
        top = [
            l[0]
            for l
            in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]
        ]
        cluster_dict[n] = top

    return cluster_dict


def find_exact_match(row, groups, match_cols):
    """Find an exact mach for a row in groups based on match_cols.
    """
    index = tuple(row[t] for t in match_cols)
    try:
        group = groups.get_group(index)
    except KeyError:
        return []
    clus = list(set(group.hotel_cluster))
    return clus


def f5(seq, idfun=None):
    """Uniquify a list by Peter Bengtsson
    https://www.peterbe.com/plog/uniqifiers-benchmark
    """
    if idfun is None:
        def idfun(x):
            return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def main():
    """Main script.
    """

    # Defining data columns that will be used
    #traincols = ['date_time', 'site_name', 'posa_continent', 'user_location_country',
                 #'user_location_region', 'user_location_city', 'orig_destination_distance',
                 #'user_id', 'is_mobile', 'is_package', 'channel', 'srch_ci', 'srch_co',
                 #'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id',
                 #'srch_destination_type_id', 'is_booking', 'cnt', 'hotel_continent',
                 #'hotel_country', 'hotel_market', 'hotel_cluster']
    #testcols = ['id', 'date_time', 'site_name', 'posa_continent', 'user_location_country',
                #'user_location_region', 'user_location_city', 'orig_destination_distance',
                #'user_id', 'is_mobile', 'is_package', 'channel', 'srch_ci', 'srch_co',
                #'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'srch_destination_id',
                #'srch_destination_type_id', 'hotel_continent', 'hotel_country', 'hotel_market']
    traincols = [
        'user_location_country', 'user_location_region', 'user_location_city',
        'orig_destination_distance', 'srch_destination_id', 'is_booking',
        'hotel_market', 'hotel_cluster',
    ]
    testcols = [
        'user_location_country', 'user_location_region', 'user_location_city',
        'orig_destination_distance', 'srch_destination_id', 'hotel_market',
    ]

    # Reading training data
    train = read_csv('data/train.csv.gz', cols=traincols, nrows=None)

    # Find the most common hotel clusters in the dataset
    top_clust = find_most_common(train)

    # Find the most common hotel clusters for each destination
    match_cols_dest = ['srch_destination_id']
    top_clust_in_dest = find_most_common_in_match(train, match_cols_dest)

    # Utilizing the data leak
    match_cols_leak = [
        'user_location_country',
        'user_location_region',
        'user_location_city',
        'hotel_market',
        'orig_destination_distance',
    ]

    groups = train.groupby(match_cols_leak)

    # Reading test data
    test = read_csv('data/test.csv.gz', cols=testcols, nrows=None)

    # Make predictions
    preds = []
    for _, row in test.iterrows():
        # Use the most common hotel cluster data
        key = make_key([row[m] for m in match_cols_dest])
        pred_dest = top_clust_in_dest.get(key, top_clust)

        # Use the data leak
        pred_leak = find_exact_match(row, groups, match_cols_leak)

        full_pred = f5(pred_leak + pred_dest)[:5]

        preds.append(full_pred)

    # Write out the submission file
    write_p = [" ".join([str(l) for l in p]) for p in preds]
    write_frame = [
        "{},{}".format(train.index[i], write_p[i])
        for i in range(len(preds))]
    write_frame = ["id,hotel_cluster"] + write_frame
    with open('out/predictions_1.csv', 'w+') as f:
        f.write('\n'.join(write_frame))


if __name__ == '__main__':
    main()
