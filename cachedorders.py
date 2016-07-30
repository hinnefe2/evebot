import datetime as dt
import logging
import requests as rq
from xml.etree import ElementTree

import pandas as pd

from eveexceptions import NoOCRMatchException

INV_TYPES = pd.read_csv('static/invTypes.csv')

class CachedCharacterOrders():
    """Class to store the result of a cached API call"""

    def __init__(self, credentials=None):
        self.logger = logging.getLogger(__name__)
        self.key_id = '442571'
        self.access_code = '7qdTnpfrBfL3Gw2elwKaT9SsGkn6O5gwV3QUM77S3pHPanRBzzDyql5pCUU7V0bS'
        self.api_url = 'https://api.eveonline.com/char/MarketOrders.xml.aspx?keyID={}&vCode={}'

        self.api_cached_until = dt.datetime(2000, 1, 1)
        self.last_updated_locally = dt.datetime.utcnow()
        self.data = pd.DataFrame()

        self.pull()

    def pull(self):
        """Pull the data from the API"""
        self.logger.debug('Querying character orders from XML API')
        resp = rq.get(self.api_url.format(self.key_id, self.access_code))

        # ugly, but can't get lxml to work with bs4
        tree = ElementTree.fromstring(resp.content)
        char_orders = pd.DataFrame(
            [r.attrib
             for r
             in tree.getchildren()[1].getchildren()[0].getchildren()
            ]).convert_objects(convert_numeric=True)

        # add a column with price as string
        char_orders['price_str'] = char_orders['price'].map('{:.2f}'.format)

        # join the typeNames to the orders dataframe
        names = INV_TYPES[['typeID', 'typeName']]
        char_orders = char_orders.merge(names, on='typeID', how='left')

        self.logger.debug('Got character orders from XML API')

        # in UTC, but w/out tzinfo in the datetime object
        self.api_cached_until = pd.to_datetime(tree.getchildren()[2].text)
        self.data = char_orders

        self.logger.debug('API character orders cached until %s UTC', self.api_cached_until)

    def update(self, old_row, new_price):
        """Update the locally stored information to reflect
        changes that haven't yet appeared in the API due to caching"""

        if dt.datetime.utcnow() > self.api_cached_until:
            self.pull()

        # update the price information
        order_id = old_row.orderID
        print(order_id)

        self.data.loc[self.data.orderID == order_id, 'price'] = new_price

    def get(self):
        """Return the character order dataframe"""
        return self.data

    def match_ocr_text(self, ocr_text, order_type):
        """Try to match the OCR'd order text to known orders from the API / cache"""

        # threshold values for match differences
        pct_threshold = 0.3
        diff_threshold = 3

        # load the orders from the local cache
        char_orders = self.get()
    
        # only search in buys or sells to prevent accidentally matching on wrong order type
        if order_type == 'buy':
            buys_or_sells = char_orders.loc[char_orders.bid == 1]
        elif order_type == 'else':
            buys_or_sells = char_orders.loc[char_orders.bid == 0]
        else:
            raise Exception('invalid order type: {}'.format(order_type))
    
        # compare to api strings w levenshtein distance
        to_search = buys_or_sells.typeName
        to_match = ocr_text
        lev_distance = to_search.apply(lev.distance, args=[to_match])
    
        best_match_index = np.argmin(lev_distance)
        best_match_diff = np.min(lev_distance)
        best_match = char_orders.iloc[best_match_index]
    
        # raise an exception if the ocr strings don't match anything
        pct_diff = best_match_diff / len(to_match)
    
        if (pct_diff > pct_threshold) and (best_match_diff > diff_threshold):
            logger.error('trying to match (%s) Best match (%s). Distance pct %.2f}',
                         ocr_text, best_match.typeName, pct_diff)
            raise NoOCRMatchException
    
        logger.debug('Matched (%s) to (%s)', ocr_text, best_match.typeName)
    
        return char_orders.iloc[best_match_index]
