import logging
import requests as rq

import pandas as pd

INV_TYPES = pd.read_csv('static/invTypes.csv')
logger = logging.getLogger(__name__)

# turn off logging from these two libraries
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def get_market_orders(region_id='10000030', type_id='34'):
    """Get the current market orders for an item in a region.
    API has 6 min cache time.
    
    Args:
        region_id(str): the region_id to get market orders for
        type_id(str): the type_id to get market orders for

    ReturnsL
        orders(pd.DataFrame): a dataframe containing order information
    """

    crest_order_url = 'https://crest-tq.eveonline.com/market/{}/orders/{}/?type=https://public-crest.eveonline.com/inventory/types/{}/'

    dfs = []
    for order_type in ['buy', 'sell']:
        logger.debug('Querying market %s orders from XML API for %s',
                order_type.upper(), typeID_to_name(type_id))

        resp = rq.get(crest_order_url.format(region_id, order_type, type_id))
        resp_json = resp.json()
        dfs.append(pd.DataFrame(resp_json['items']))

        logger.debug('Got market %s orders from XML API for %s', order_type.upper(), typeID_to_name(type_id))

    orders = pd.concat(dfs)

    def extract_station_id(loc_dict):
        """Pull the station ID out of the CREST callback url, add as a column"""
        return int(loc_dict['id_str'])

    orders['stationID'] = orders.location.apply(extract_station_id)

    # TODO add logging of snapshot to database
    return orders.convert_objects(convert_numeric=True).sort_values('price')


def typeID_to_name(type_id):
    """Look up item type name by typeID"""
    return INV_TYPES.loc[INV_TYPES.typeID == int(type_id)].typeName.values[0]

