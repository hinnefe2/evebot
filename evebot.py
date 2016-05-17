"""Module to automate trade-oriented interactions with the
EVE Online client"""

from __future__ import division

import os
import re
import time
import logging
import datetime as dt
from xml.etree import ElementTree

import requests as rq
import PIL
import psutil
import numpy as np
import pandas as pd
import cv2
import pyautogui as pg
import pyperclip as pc
import pytesseract as pt
import Levenshtein as lev


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# set format of log entries
formatter = logging.Formatter('%(asctime)s %(funcName)-20s %(levelname)-8s %(message)s')

# for logging to file
fh = logging.FileHandler('evebot.log')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)

# for logging order statuses
fh2 = logging.FileHandler('evebot-orderstatus.log')
fh2.setFormatter(formatter)
fh2.setLevel(logging.INFO)

# for logging to console
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# add handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)
logger.addHandler(fh2)


##############################################
# Constants and Static data imports
##############################################

pg.PAUSE = 2
MAX_API_ATTEMPTS = 5
MAX_DELTA_ISK = 20000000
INV_TYPES = pd.read_csv('static/invTypes.csv')


def retry_timeout(timeout, sleep_time=3):
    """Decorator to attempt a function until a timeout expires"""
    def retry(func):
        def func_wrapper(*args, **kwargs):

            t_start = dt.datetime.now()
            t_wait = dt.timedelta(seconds=timeout)
            attempt = 0

            while dt.datetime.now() < t_start + t_wait:

                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempt += 1
                    time.sleep(sleep_time)

            logger.error('Failed %s after %d attempts in %d seconds', str(func), attempt, timeout)

        return func_wrapper
    return retry


class CachedCharacterOrders():
    """Class to store the result of a cached API call"""

    def __init__(self):
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

        self.data.loc[self.data.orderID == order_id].price = new_price

    def get(self):
        """Return the character order dataframe"""
        return self.data


##############################################
# Custom exception classes
##############################################

class APICallException(Exception):
    """Exception for when an API call doesn't return the appropriate data"""
    pass


class TemplateNotFoundException(Exception):
    """Exception for when the specified template can't be found on the screen"""
    pass


class NoOCRMatchException(Exception):
    """Exception for when order information extracted via OCR can't be matched to orders"""
    pass


class LauncherNotStartedException(Exception):
    """Exception for when the EVE launcher doesn't start"""
    pass


##############################################
# Utility functions
##############################################

def flip_color(cv2_arr):
    """Exchange the R and B values in a cv2 RGB tuple"""
    return cv2_arr[:, :, ::-1]


def wh_from_pos(pos1, pos2):
    """Calculate width and height from top-left, bottom-right positions"""
    left = pos1[0]
    top = pos1[1]
    w = pos2[0] - pos1[0]
    h = pos2[1] - pos1[1]
    return (left, top, w, h)


def paste_text(text):
    """Place 'text' in the clipboard then paste it to the active input area"""
    pc.copy(str(text))
    pg.keyDown('ctrl')
    pg.press('v')
    pg.keyUp('ctrl')


##############################################
# GUI interaction functions (moving, locating, etc)
##############################################

@retry_timeout(60)
def match_template(template_path, screen=None, threshold=0.9, how=cv2.TM_CCOEFF_NORMED):
    """Find region on the screen matching template"""

    template = flip_color(cv2.imread(template_path))

    if screen is None:
        screen = np.array(pg.screenshot())

    template_arr = np.array(template)
    screen_arr = np.array(screen)

    result_arr = cv2.matchTemplate(screen_arr, template_arr, how)

    matches = np.array(np.where(result_arr >= threshold))
    if matches.size == 0:
        raise TemplateNotFoundException

    (_, maxMatch, _, maxLoc) = cv2.minMaxLoc(result_arr)
    logger.debug('Found template %s at %s with confidence %s', template_path, maxLoc, maxMatch)

    return np.array(maxLoc)


def moveTo(target_pos, jitter=(0, 0), relative=False):
    """Move the cursor, accounting for travel time"""

    def dist(pos1, pos2):
        """Return the distance between two points"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    MOVE_SPEED = 5000   # px / sec
    time_jitter = np.random.randn(1)[0] / 20

    current_pos = np.array(pg.position())
    target_pos = np.array(target_pos)

    move_length = dist(current_pos, target_pos)
    move_time = move_length / MOVE_SPEED + time_jitter

    to_pos = target_pos + jitter

    if relative:
        pg.moveRel(*to_pos, duration=move_time)
    else:
        pg.moveTo(*to_pos, duration=move_time)


def click_img(img_path, button='left'):
    """Click on the region of the screen matching the supplied image"""

    # find the top left corner of the image
    img_pos = match_template(img_path)

    if img_pos is None:
        return False

    # find the center of the image as top left corner + w/2, + h/2
    img = cv2.imread(img_path)
    offset_x = img.shape[1] / 2
    offset_y = img.shape[0] / 2
    img_center = [coord + offset for coord, offset in zip(img_pos, [offset_x, offset_y])]

    # add a uniform jitter over the whole image
    jitter_x = np.random.randint(-offset_x, offset_x)
    jitter_y = np.random.randint(-offset_y, offset_y)

    moveTo(img_center, jitter=(jitter_x, jitter_y))
    pg.click(button=button)


def ocr_row(row_top_left):
    """Extract order type name and price from screenshot using OCR"""
    name_w = 284
    row_y_offset = 17

    price_x_offset = 405
    price_w = 100

    name_top_left = row_top_left
    name_bottom_right = (row_top_left[0] + name_w, row_top_left[1] + row_y_offset)

    # take a screenshot of the 'name' column in the given row, then blow it up 5x
    name_snap = pg.screenshot(region=wh_from_pos(name_top_left, name_bottom_right))
    name_snap_x5 = name_snap.resize(np.array(name_snap.size) * 5, PIL.Image.ANTIALIAS)
    name_text = pt.image_to_string(name_snap_x5)

    price_top_left = (row_top_left[0] + price_x_offset, row_top_left[1])
    price_bottom_right = (row_top_left[0] + price_x_offset + price_w, row_top_left[1] + row_y_offset)

    # take a screenshot of the 'price' column in the given row, then blow it up 5x
    price_snap = pg.screenshot(region=wh_from_pos(price_top_left, price_bottom_right))
    price_snap_x5 = price_snap.resize(np.array(price_snap.size) * 5, PIL.Image.ANTIALIAS)
    price_text = pt.image_to_string(price_snap_x5)

    # strip commas, periods, spaces out of price_text, then extract just numbers
    price_text = price_text.replace('.', '').replace(', ', '').replace(' ', '')
    price_text = str(re.search('^([0-9]*).*', price_text).groups(1)[0])
    price_float = float(price_text) / 100

    logger.debug('Extracted (typename, price) : %50s %11.2f', name_text, price_float)

    return (name_text, price_text)


##############################################
# Client interaction functions
##############################################

def open_client_window(window_name):
    """Open a window in the client"""

    hotkey_dict = {'market': 'r',
                   'wallet': 'w'}

    img_dict = {'market': 'images/client-market-header.png'}

    window_img = img_dict[window_name]

    # if we can find the window header, close then reopen
    # the window to make sure it's on top
    try:
        match_template(window_img)

        pg.keyDown('alt')
        pg.press(hotkey_dict[window_name])
        pg.keyUp('alt')

        pg.keyDown('alt')
        pg.press(hotkey_dict[window_name])
        pg.keyUp('alt')

    # if we can't find the window header open it once
    except TemplateNotFoundException:

        pg.keyDown('alt')
        pg.press(hotkey_dict[window_name])
        pg.keyUp('alt')

    # wait for the window to load
    time.sleep(2)

    logger.debug("Opened client window '%s'", window_name)
    return match_template(window_img) is not None


def update_all_orders(interactive=False):
    """Iterate through all orders on the Market -> My Orders screen
    and update their prices if appropriate"""

    # find location of first SELL order
    selling_pos = match_template('images/client-market-selling.png')

    # TODO make order pos offset tidier
    selling_to_order_offset = np.array([80, 19])
    order_pos = selling_pos + selling_to_order_offset

    char_orders = cached_char_orders.get()
    n_selling = len(char_orders.loc[(char_orders.bid == 0) & (char_orders.orderState == 0)])
    logger.debug('Checking %d SELL orders', n_selling)

    for i in range(n_selling):
        this_order_pos = np.array([order_pos[0], order_pos[1] + 20 * i])

        modify_order(this_order_pos, 'sell', interactive)

    # find location of first BUY order
    buying_pos = match_template('images/client-market-buying.png')

    # TODO make order pos offset tidier
    buying_to_order_offset = np.array([80, 19])
    order_pos = buying_pos + buying_to_order_offset

    n_buying = len(char_orders.loc[(char_orders.bid == 1) & (char_orders.orderState == 0)])
    logger.debug('Checking %d BUY orders', n_buying)

    for i in range(n_buying):
        this_order_pos = np.array([order_pos[0], order_pos[1] + 20 * i])

        modify_order(this_order_pos, 'buy', interactive)


def modify_order(this_order_pos, order_type, interactive=False):
    """Modify the order at the given screen position"""

    # get ocr'd item name and price
    try:
        name_text, price_text = ocr_row(this_order_pos)
    except ValueError:
        logger.error('Failed to ocr row for %s order at %s', order_type, this_order_pos)
        return

    try:
        # match ocr'd info to api info
        char_order = match_ocr_order(name_text, price_text, order_type)
    except NoOCRMatchException:
        logger.error('Failed to match OCR order info %s %s', name_text, price_text)
        return

    if interactive:
        pg.alert('checking on {}'.format(char_order.typeName))

    # get market orders
    # TODO implement caching
    market_orders = get_market_orders(type_id=char_order.typeID)

    # calculate an appropriate new price
    new_price = choose_price(order_type, char_order, market_orders)

    # if the best new price is different update the order
    if new_price != char_order.price:

        jitter = [np.random.randint(-10, 10), np.random.randint(-2, 2)]
        # distance from top 'this_order_pos' to center of order text (for clicking)
        order_center_offset = np.array([50, 9])

        moveTo(this_order_pos + order_center_offset, jitter)
        pg.click(button='right')

        # distance from click location to 'modify order' location
        modify_offset = np.array([70, 10])
        moveTo(modify_offset, relative=True)
        pg.click()

        paste_text(new_price)

        if interactive:
            # click on 'OK' button in modify popup
            if pg.confirm() == 'OK':
                click_img('images/client-popup-ok.png')
                cached_char_orders.update(char_order, new_price)
                # wait for potential 'WARNING' window to pop open
                time.sleep(2)
            # click on 'CANCEL' button in modify popup
            else:
                click_img('images/client-popup-cancel.png')
        else:
            cached_char_orders.update(char_order, new_price)
            click_img('images/client-popup-ok.png')
            time.sleep(2)

        # if order is > x% off from market average
        try:
            match_template('images/client-popup-warning.png')
            click_img('images/client-popup-yes.png')
            time.sleep(2)
        except TemplateNotFoundException:
            pass


##############################################
# Trade update logic
##############################################

def choose_price(order_type, char_order, market_orders):
    """Choose an appropriate price for the updated order, based on
    competing order price and volume, and historical volume."""

    old_price = char_order.price
    type_name = char_order.typeName

    # minimum sell-buy margin to update on
    margin_threshold = 0.1

    # separate buy and sell orders
    buy_orders = market_orders.loc[market_orders.buy is True]
    sell_orders = market_orders.loc[(market_orders.buy is False) & (market_orders.stationID == char_order.stationID)]

    # CHECK - ours is already the best price
    if order_type == 'buy':
        best_price = buy_orders.price.max()
    else:
        best_price = sell_orders.price.min()
    if old_price == best_price:
        logging.info('HOLD - BEST - %s : %s price of %.2f is already best price', type_name, order_type, old_price)
        return old_price

    # CHECK - ISK delta < threshold
    delta_isk = (best_price - old_price) * char_order.volRemaining
    if abs(delta_isk) > MAX_DELTA_ISK:
        logging.info('HOLD - DELTA - %s : %s delta of %.2f is too high', type_name, order_type, delta_isk)
        return old_price

    # CHECK - range

    # CHECK - competing volume

    # CHECK - margin is acceptable
    margin = (sell_orders.price.min() - buy_orders.price.max()) / buy_orders.price.max()
    if margin < margin_threshold:
        logger.info('HOLD - MARGIN - %s : margin of %.3f (margin: %.2f best: %.2f current: %.2f) is too small',
                    type_name, margin, margin * buy_orders.price.max(), best_price, old_price)
        return old_price

    if order_type == 'buy':
        new_price = buy_orders.price.max() + 0.01
    else:
        new_price = sell_orders.price.min() - 0.01

    logger.info('UPDATE - %s %s to %.2f (Old: %.2f, Delta %.2f)',
                order_type.upper(), type_name, new_price, old_price, new_price - old_price)

    return new_price


##############################################
# API call functions
##############################################

@retry_timeout(20)
def get_market_orders(region_id='10000030', type_id='34'):
    """Get the current market orders for an item in a region.

    API has 6 min cache time."""

    crest_order_url = 'https://crest-tq.eveonline.com/market/{}/orders/{}/?type=https://public-crest.eveonline.com/types/{}/'

    dfs = []
    for order_type in ['buy', 'sell']:
        logger.debug('Querying market %s orders from XML API for typeID %d', order_type.upper(), type_id)

        resp = rq.get(crest_order_url.format(region_id, order_type, type_id))
        resp_json = resp.json()
        dfs.append(pd.DataFrame(resp_json['items']))

        logger.debug('Got market %s orders from XML API for typeID %d', order_type.upper(), type_id)

    orders = pd.concat(dfs)

    def extract_station_id(loc_dict):
        """Pull the station ID out of the CREST callback url, add as a column"""
        return int(loc_dict['id_str'])
    orders['stationID'] = orders.location.apply(extract_station_id)

    # TODO add logging of snapshot to database

    return orders.convert_objects(convert_numeric=True).sort_values('price')


@retry_timeout(20)
def get_char_orders():
    """Get a character's current market orders.

    API has 1 hour cache time."""

    theo_keyID = '442571'
    theo_code = '7qdTnpfrBfL3Gw2elwKaT9SsGkn6O5gwV3QUM77S3pHPanRBzzDyql5pCUU7V0bS'

    xml_orders_url = 'https://api.eveonline.com/char/MarketOrders.xml.aspx?keyID={}&vCode={}'

    logger.debug('Querying character orders from XML API')
    resp = rq.get(xml_orders_url.format(theo_keyID, theo_code))

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

    logger.debug('Got character orders from XML API')

    return char_orders


def match_ocr_order(name_text, price_text, order_type, char_orders=None, pct_threshold=0.2):
    """Match the name and price scraped from the screen to info gathered from the API"""

    if char_orders is None:
        char_orders = cached_char_orders.get()

    buysell_dict = {'sell': 0, 'buy': 1}

    # only search in buys or sells to prevent accidentally matching on wrong order type
    buys_or_sells = char_orders.loc[char_orders.bid == buysell_dict[order_type]]
    # catenate ocr strings, compare to api strings w levenshtein distance
    to_search = buys_or_sells.apply(lambda row: row.typeName + row.price_str, axis=1)

    to_match = name_text + price_text
    lev_distance = to_search.apply(lev.distance, args=[to_match])

    best_match_index = np.argmin(lev_distance)
    best_match_diff = np.min(lev_distance)
    best_match = char_orders.iloc[best_match_index]

    # raise an exception if the ocr strings don't match anything
    pct_diff = best_match_diff / len(to_match)

    if pct_diff > pct_threshold:
        logger.error('trying to match (%s %s) Best match (%s %s). Distance pct %.2f}',
                     name_text, price_text, best_match.typeName, best_match.price, pct_diff)
        raise NoOCRMatchException

    logger.debug('Matched (%s, %.2f) to (%s, %.2f)',
                 name_text, int(price_text) / 100, best_match.typeName, best_match.price)

    return char_orders.iloc[best_match_index]


#################################################
# Startup / shutdown functions
#################################################

@retry_timeout(20)
def start_launcher(launcher_path=r"C:\Program Files (x86)\EVE\Launcher\evelauncher.exe"):
    """Start the EVE launcher"""

    logger.debug('Starting EVE launcher')

    try:
        os.startfile(launcher_path)
    except:
        logger.error('Could not run %s', launcher_path)
        return False

    def get_proc_name(process):
        """Extract process name from psutil.process instance"""
        try:
            return process.name()
        except:
            return None

    # check if the launcher is running
    process_names = list([get_proc_name(proc) for proc in psutil.process_iter()])

    # if the launcher process is found return True
    if 'evelauncher.exe' in process_names:
        logger.debug('Successfully started launcher')
        return
    else:
        raise LauncherNotStartedException


def start_eve():
    """Launch the EVE client by clicking on the launcher login button."""

    logger.debug('Starting EVE client')
    click_img('images/launcher-login-small.png')
    return True


def wait_for_window(window_name, window_img_path, timeout=60):
    """Wait for the window identified by window_img to load

    Returns:
    - success(bool): Whether the window has loaded
    """

    logger.debug('Waiting for %s to load', window_name)

    # so that cursor isn't over anything important
    pg.moveTo(10, 10)

    t_start = dt.datetime.now()
    t_wait = dt.timedelta(seconds=timeout)

    while dt.datetime.now() < t_start + t_wait:

        try:
            match_template(window_img_path)
            logger.debug('%s window ready', window_name)
            return True
        except TemplateNotFoundException:
            time.sleep(5)

    # should only get here if we timed out
    logger.error('Could not find %s window', window_name)
    return False


def quit_launcher():
    """Close the launcher"""

    logger.debug('Killing the EVE launcher process')
    os.system("taskkill /im evelauncher.exe")


def quit_client():
    """Close the EVE client"""

    pg.press('esc')
    time.sleep(2)
    click_img('images/client-popup-quit.png')


def login_trader():
    """Login to the client with the trading character"""

    # so that cursor isn't on top of character portrait
    pg.moveTo(100, 100)

    logger.debug('Logging in trading character')
    click_img('images/client-theoface-small.png')


def cold_start():
    """Get client to trading state (client running, market window open)"""

    start_launcher()
    wait_for_window('EVE launcher', 'images/launcher-header.png')
    start_eve()
    wait_for_window('Character selection', 'images/client-theoname.png')
    login_trader()
    wait_for_window('In-game GUI', 'images/client-ingame-gui.png')
    open_client_window('market')
    click_img('images/client-market-myorders-dim.png')


def cold_stop():
    """Close the client and the launcher"""

    quit_client()
    quit_launcher()


if __name__ == '__main__':

    # add order range check
    # add pct change in order isk check

    cached_char_orders = CachedCharacterOrders()

    updates = 0

    while updates < 6:
        cold_start()
        logging.info('Updating orders for %d time', updates)
        update_all_orders(interactive=False)
        cold_stop()

        # wait for the order api cache to update
        until_cache_update = cached_char_orders.api_cached_until - dt.datetime.utcnow()
        while until_cache_update > dt.timedelta(0):
            logger.debug('waiting for order cache to update. %d more seconds', until_cache_update.seconds)
            time.sleep(300)
            until_cache_update = cached_char_orders.api_cached_until - dt.datetime.utcnow()

        # sleep a random amount of time so we're not updating immediately after the cache
        time.sleep(np.random.randint(5, 600))
        updates += 1
