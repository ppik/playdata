#!/usr/bin/env python
"""Check the number of empty fields in the data.
"""

import gzip
import numpy as np

def calc_na_count(filename):

    with gzip.open(filename, 'rt') as f:
        header = f.readline().strip().split(',')
        count = np.zeros(len(header), dtype=np.int)

        for line in f:            
            line = np.array(line.strip().split(','))
            count += line == ''
    
    return dict(zip(header, count))


def main():
    
    print('Training data')
    print(calc_na_count('data/train.csv.gz'))
    
    print('Testing data')
    print(calc_na_count('data/test.csv.gz'))
    
    
if __name__ == "__main__":
    main()


"""
Results:
Training data
{'srch_children_cnt': 0, 'hotel_cluster': 0, 'user_location_country': 0, 'user_location_region': 0, 'site_name': 0, 'is_package': 0, 'hotel_continent': 0, 'srch_co': 47084, 'posa_continent': 0, 'srch_adults_cnt': 0, 'user_id': 0, 'srch_destination_id': 0, 'hotel_country': 0, 'orig_destination_distance': 13525001, 'srch_rm_cnt': 0, 'date_time': 0, 'user_location_city': 0, 'is_booking': 0, 'srch_destination_type_id': 0, 'channel': 0, 'hotel_market': 0, 'srch_ci': 47083, 'cnt': 0, 'is_mobile': 0}
Testing data
{'srch_children_cnt': 0, 'user_location_country': 0, 'user_location_region': 0, 'site_name': 0, 'is_package': 0, 'hotel_continent': 0, 'srch_co': 17, 'posa_continent': 0, 'srch_adults_cnt': 0, 'user_id': 0, 'srch_destination_id': 0, 'hotel_country': 0, 'srch_rm_cnt': 0, 'date_time': 0, 'id': 0, 'user_location_city': 0, 'orig_destination_distance': 847461, 'srch_destination_type_id': 0, 'channel': 0, 'hotel_market': 0, 'srch_ci': 21, 'is_mobile': 0}
"""
