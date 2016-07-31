"""Module to automate trade-oriented interactions with the EVE Online client"""

import datetime as dt
import logging
import os
import sys
import time
import yaml

import cv2
import numpy as np
import pyautogui as pg
import pyperclip as pc
import pytesseract as pt
import PIL
import Levenshtein as lev

import api
from cachedorders import CachedCharacterOrders
from eveexceptions import WindowNotFoundException
from eveexceptions import TemplateNotFoundException
from eveexceptions import NoOCRMatchException

logging.basicConfig(format='%(asctime)s %(funcName)-20s %(levelname)-8s %(message)s',
		    stream=sys.stdout,
		    level=logging.DEBUG)
logger = logging.getLogger(__name__)

pg.PAUSE = 1

#TODO: refactor this into the config file
MAX_DELTA_ISK = 30000000

class evebot:
    """A class to handle logging in to the EVE client, and the automated 
    updating and entering of market ordrs"""

    def __init__(self, config_file=None, database=None):

        # read in API keys
        with open(config_file) as infile:
            self.config = yaml.load(infile.read())

        # instantiate local order, transaction, inventory caches
        self.cached_orders = CachedCharacterOrders()
        self.cached_txns = None #CachedCharacterTxns(self.config['credentials'])
        self.cached_inventory = None #CachedCharacterInventory(self.config['credentials'])

        # instantiate the class which interacts with the client
        self.client = ClientManager(self.config)

        # instantiate the market-interacting component classes
        self.updater = OrderUpdater(self.client, self.config, self.cached_orders)
        self.seller = ItemSeller(self.client)#, self.config)
        self.buyer = ItemBuyer(self.client)#, self.config)


    def main(self):
        """Update existing orders, put current inventory up for sale, and enter new buy orders"""

        for i in range(self.config['n_main_loops']):
            if self.client.login():
                self.updater.main()
                self.seller.main()
                self.buyer.main()
                self.client.logout()

            time.sleep(self.config['t_sleep_main'])


class ClientManager:
    """A class to interact with the EVE client (and launcher)"""

    def __init__(self, config):
        self.config = config

    
    def wait_for_window(self, window_name, window_img_path, timeout=10):
        """Wait for the window identified by window_img to load
    
        Args:
            window_name(str): the name of the window, for logging purposes
            window_img_path(str): the path the the template image which identifies the window
            timeout(int): the number of seconds to wait for the window
    
        Raises:
            WindowNotFoundException: if the specified window isn't detected after the timeout period
        """
    
        logger.debug('Waiting for %s to load', window_name)
    
        t_start = dt.datetime.now()
        t_wait = dt.timedelta(seconds=timeout)
    
        # TODO: put window name, image paths in config file
        while dt.datetime.now() < t_start + t_wait:
    
            try:
                self.find_template(window_img_path)
                logger.debug('%s window ready', window_name)
                return True

            except TemplateNotFoundException:
                time.sleep(5)
    
        # should only get here if we timed out
        logger.debug('Could not find %s window', window_name)
    
        raise WindowNotFoundException()


    def login(self):
        """Log in to the EVE client
        
        Returns:
            success(bool): whether or not the login was succesfull"""
        
        # TODO: put image paths in config file

        try:
            # start the launcher
            os.startfile(self.config['launcher_path'])
            self.wait_for_window('EVE launcher', 'images/launcher-login-small.png', 60)
            logger.debug('started launcher')
    
            # start the EVE client
            self.click_template('images/launcher-login-small.png')
            self.wait_for_window('Character selection', 'images/client-theoface-small.png', 60)
            logger.debug('started client')
        
            # log in trading character
            self.click_template('images/client-theoface-small.png')
            self.wait_for_window('In-game GUI', 'images/client-ingame-gui.png', 60)
            logger.debug('logged in character')
        
            # kill the launcher (it uses lots of CPU for some reason)
            os.system("taskkill /im evelauncher.exe")
            logger.debug('killed launcher')

            return True

        except WindowNotFoundException:
            logger.error('Failed to log in')

            return False
    

    def logout(self):
        """Close the EVE client and launcher"""

        # close all windows, in case any are open
        self.close_all_windows()

        # kill the EVE client process
        os.system("taskkill /im exefile.exe")
        logger.debug('killed EVE client')


    def move_to(self, position, jitter=(0,0), relative=False):
        """Move the cursor to the specified position

        Args:
            position(tuple): (X,Y) position to move the cursor to
            jitter(tuple): maximum (X,Y) values to add for randomization
            relative(bool): whether the move should be relative to current position
        """

        def dist(pos1, pos2):
            """Return the distance between two points"""
            return np.linalg.norm(np.array(pos1) - np.array(pos2))
    
        # TODO: put in config file
        MOVE_SPEED = 5000   # px / sec
        time_jitter = np.random.randn(1)[0] / 20
    
        current_pos = np.array(pg.position())
        target_pos = np.array(position)
    
        move_length = dist(current_pos, target_pos)
        move_time = move_length / MOVE_SPEED + time_jitter
    
        to_pos = target_pos + jitter
    
        if relative:
            pg.moveRel(*to_pos, duration=move_time)
        else:
            pg.moveTo(*to_pos, duration=move_time)

    
    def find_template(self, template_path, threshold=0.9, how=cv2.TM_CCOEFF_NORMED):
        """Find the location of the provided template image on the screen.

        Args:
            template_path(str): path to image file
            threshold(float): match threshold to consider template 'found'
            how: cv2 argument to determine how to do the matching

        Returns:
            position(np.array): (X, Y) position of the top left corner of the template
        """

        # load the template image, flip the color b/c cv2 pulls tuples as BGR instead of RGB
        template = flip_color(cv2.imread(template_path))
    
        # take a screenshot
        screen = pg.screenshot()
    
        # convert both images to numpy arrays
        template_arr = np.array(template)
        screen_arr = np.array(screen)
    
        result_arr = cv2.matchTemplate(screen_arr, template_arr, how)
        matches = np.array(np.where(result_arr >= threshold))

        if matches.size == 0:
            logger.error('Could not find template %s', template_path)
            raise TemplateNotFoundException
    
        (_, maxMatch, _, maxLoc) = cv2.minMaxLoc(result_arr)
        logger.debug('Found template %s at %s with confidence %s', template_path, maxLoc, maxMatch)
    
        return np.array(maxLoc)


    def click_template(self, template_path, button='left'):
        """Click on the region of the screen that matches the provided template"""

        # find the top left corner of the image
        img_pos = self.find_template(template_path)
    
        # find the center of the image as top left corner + w/2, + h/2
        img = cv2.imread(template_path)
        offset_x = img.shape[1] / 2
        offset_y = img.shape[0] / 2
        img_center = [coord + offset for coord, offset in zip(img_pos, [offset_x, offset_y])]
    
        # add a uniform jitter over 80% of the whole image
        jitter_x = np.random.randint(-offset_x, offset_x) * 0.8
        jitter_y = np.random.randint(-offset_y, offset_y) * 0.8
    
        self.move_to(img_center, jitter=(jitter_x, jitter_y))
        pg.click(button=button)


    def close_all_windows(self):
        """Close all open windows in the EVE client"""

        pg.keyDown('alt')
        pg.keyDown('ctrl')
        pg.press('w')
        pg.keyUp('ctrl')
        pg.keyUp('alt')


    def open_window(self, window_name):
        """Open a window in the EVE client

        Args:
            window_name(str): the name of the window to open {'market', 'assets'}
        """

        hotkey_dict = {'market': 'r',
                       'wallet': 'w',
                       'assets': 'c'}
    
        img_dict = {'market': 'images/client-market-header2.png',
                    'assets': 'images/client-assets/header.png'}
    
        # figure out which key and template file are relevant
        window_hkey = hotkey_dict[window_name]
        window_img = img_dict[window_name]

        pg.keyDown('alt')
        pg.press(hotkey_dict[window_name])
        pg.keyUp('alt')

        logger.debug("Opened client window '%s'", window_name)


    def open_context_menu(self, position, jitter=(0,0)):
        """Open a context menu by right-clicking at position

        Args:
            position(tuple): the (x,y) position to right click
            jitter(tuple): the maximum x and y values to randomly add
                           to the position to evade detection
        """
        self.move_to(position, jitter)
        pg.click(button='right')


    def enter_text(self, text):
        """Paste the provided text. Assumes that a text input box is active

        Args:
            text(str): the text to be pasted
        """

        pc.copy(str(text))
        pg.keyDown('ctrl')
        pg.press('v')
        pg.keyUp('ctrl')


    def ocr_row(self, position, height=10, width=200):
        """Attempt to read the text of a row in a window in the EVE client using OCR

        Args:
            position(tuple): (X,Y) position of the top left corner of the row
            height(int): height of the row in px
            width(int): width of the row in px

        Returns:
            name_text(str): the text of the item name, as identified by OCR
        """
        # TODO: refactor this method
        # TODO: put this in config file
        name_w = 284
        row_y_offset = 17
    
        price_x_offset = 405
        price_w = 100
    
        name_top_left = position
        name_bottom_right = (position[0] + name_w, position[1] + row_y_offset)
    
        # take a screenshot of the 'name' column in the given row, then blow it up 5x
        name_snap = pg.screenshot(region=wh_from_pos(name_top_left, name_bottom_right))
        name_snap_x5 = name_snap.resize(np.array(name_snap.size) * 5, PIL.Image.ANTIALIAS)
        name_text = pt.image_to_string(name_snap_x5, config='letters')
    
        #price_top_left = (row_top_left[0] + price_x_offset, row_top_left[1])
        #price_bottom_right = (row_top_left[0] + price_x_offset + price_w, row_top_left[1] + row_y_offset)
    
        # take a screenshot of the 'price' column in the given row, then blow it up 5x
        #price_snap = pg.screenshot(region=wh_from_pos(price_top_left, price_bottom_right))
        #price_snap_x5 = price_snap.resize(np.array(price_snap.size) * 5, PIL.Image.ANTIALIAS)
        #price_text = pt.image_to_string(price_snap_x5, config='letters')
    
        # strip commas, periods, spaces out of price_text, then extract just numbers
        #price_text = price_text.replace('.', '').replace(',', '').replace(' ', '')
        #price_text = str(re.search('^([0-9]*).*', price_text).groups(1)[0])
        #price_float = float(price_text) / 100
    
        logger.debug('Extracted item name : (%s)', name_text)
    
        return name_text

    def open_orders_screen(self):
        """Open the orders screen so that orders can be updated"""

	# open up the market window
        self.open_window('market')
        self.wait_for_window('In-game market window', 'images/client-market-header2.png', 20)
    	
	# open up the 'My Orders' tab of the market window
	self.click_template('images/client-market-myorders-dim.png')
        self.wait_for_window('Character orders tab', 'images/client-market-selling.png', 20)


class OrderUpdater:
    """A class to automate the updating of existing market orders"""

    def __init__(self, client, config, cached_orders):
        self.client = client
        self.config = config
        self.cached_orders = cached_orders

    
    def modify_order(self, position, order_type):
        """Update the order at the specified position
        
        Args:
            position(tuple): the top left corner of the order row
            order_type(str): 'buy' or 'sell'
        """

        # get ocr'd item name and price
        try:
            name_text = self.client.ocr_row(position)
        except ValueError:
            logger.error('Failed to ocr row for %s order at %s', order_type, position)
            return False
    
        # match ocr'd info to api info
        try:
            this_order = self.cached_orders.match_ocr_text(name_text, order_type)
	    print("this_order: ", this_order)
        except NoOCRMatchException:
            logger.error('Failed to match OCR order info %s ', name_text)
            return False
    
        # calculate an appropriate new price
        new_price = self.choose_price(this_order)
    
        # if the best new price is different update the order
        if new_price != this_order.price:
    
            jitter = [np.random.randint(-10, 10), np.random.randint(-2, 2)]
            # distance from top 'this_order_pos' to center of order text (for clicking)
            order_center_offset = np.array([50, 9])
            this_order_pos = position + order_center_offset
    
            # open the context menu, click on modify order, paste price
            self.client.open_context_menu(this_order_pos, jitter)
            self.client.click_template('images/client-context-modifyorder.png')
            time.sleep(1)
            self.client.enter_text(new_price)
    
            self.client.wait_for_window('Order confirmation', 'images/client-popup-ok.png')
            self.client.click_template('images/client-popup-ok.png')
            time.sleep(2)
    
            self.cached_orders.update(this_order, new_price)

            # if order is > x% off from market average
            # TODO: put wait time in config file
            try:
                self.client.wait_for_window('Price warning window', 'images/client-popup-warning.png', 5)
                self.client.click_template('images/client-popup-yes.png')
                time.sleep(2)
            except WindowNotFoundException:
                pass


    def choose_price(self, this_order):
        """Choose the optimal price for `this_order`

        Args:
            this_order(pd.Series): pandas series containing info about a 
                                   currently active character order

        Returns:
            new_price(float): suggested price for the supplied order
        """

        old_price = this_order.price
        type_name = this_order.typeName
        region_id = lookup_region_id(this_order.stationID)
        order_type = {1:'buy', 0:'sell'}[this_order.bid]

        # minimum sell-buy margin to update on
        margin_threshold = 0.08
        
        # get market orders11
        # TODO implement caching
        market_orders = api.get_market_orders(region_id = region_id,
                                              type_id = this_order.typeID)

        # separate buy and sell orders
        buy_orders = market_orders.loc[market_orders.buy == True]

	#TODO !!!!!!!!!! FIX THIS
	print(market_orders[['price', 'stationID']])
	print(this_order[['price','stationID']])
        sell_orders = market_orders.loc[(market_orders.buy == False) & (market_orders.stationID == this_order.stationID)]

        assert len(buy_orders) > 0, "No buy orders extracted from API for price comparison"
        assert len(sell_orders) > 0, "No sell orders extracted from API for price comparison"

        # CHECK - ours is already the best price
        if order_type == 'buy':
            best_price = buy_orders.price.max()
        else:
            best_price = sell_orders.price.min()
        if old_price == best_price:
            logger.info('%s - HOLD - BEST : %s price of %.2f is already best price', order_type.upper(), type_name, old_price)
            return old_price

        # CHECK - ISK delta < threshold
        delta_isk = (best_price - old_price) * this_order.volRemaining
        if abs(delta_isk) > MAX_DELTA_ISK:
            logger.info('%s - HOLD - DELTA : %s delta of %.2f is too high', order_type.upper(), type_name, delta_isk)
            return old_price

        # CHECK - range

        # CHECK - competing volume

        # CHECK - margin is acceptable
        margin = (sell_orders.price.min() - buy_orders.price.max()) / buy_orders.price.max()
        if margin < margin_threshold:
            logger.info('%s - HOLD - MARGIN - %s : margin of %.3f (margin: %.2f best: %.2f current: %.2f) is too small',
                        order_type.upper(), type_name, margin, margin * buy_orders.price.max(), best_price, old_price)
            return old_price

        if order_type == 'buy':
            new_price = buy_orders.price.max() + 0.01
        else:
            new_price = sell_orders.price.min() - 0.01

        logger.info('%s - UPDATE %s to %.2f (Old: %.2f, Delta %.2f)',
                    order_type.upper(), type_name, new_price, old_price, new_price - old_price)

        return new_price
     
    
    def main(self):
        """Update all orders"""

	self.client.close_all_windows()
	self.client.open_orders_screen()

        # find location of first SELL order
        selling_pos = self.client.find_template('images/client-market-selling.png')
    
        # TODO make order pos offset tidier
        selling_to_order_offset = np.array([80, 19])
        order_pos = selling_pos + selling_to_order_offset
    
        char_orders = self.cached_orders.data
        n_selling = len(char_orders.loc[(char_orders.bid == 0) & (char_orders.orderState == 0)])
        logger.debug('Checking %d SELL orders', n_selling)
    
        for i in range(n_selling):
            this_order_pos = np.array([order_pos[0], order_pos[1] + 20 * i])
    
            self.modify_order(this_order_pos, 'sell')
    
        # find location of first BUY order
        buying_pos = self.client.find_template('images/client-market-buying.png')
    
        # TODO make order pos offset tidier
        buying_to_order_offset = np.array([80, 19])
        order_pos = buying_pos + buying_to_order_offset
    
        n_buying = len(char_orders.loc[(char_orders.bid == 1) & (char_orders.orderState == 0)])
        logger.debug('Checking %d BUY orders', n_buying)
    
        for i in range(n_buying):
            this_order_pos = np.array([order_pos[0], order_pos[1] + 20 * i])
    
            self.modify_order(this_order_pos, 'buy', interactive)
    
    
class ItemSeller:
    """A class to automate the selling of items in the inventory"""

    def __init__(self, client):
        self.client = client

    def open_assets_screen(self):
        """Open the assets screen so that items can be listed for sale"""
        pass

    def match_ocr_text(self):
        """Try to match the OCR'd item text to known items from the API / cache"""
        pass

    def sell_item(self, position):
        """Sell the item at the specified position"""
        pass

    def main(self):
        """Sell some or all items, in order of profit potential"""
        pass


class ItemBuyer:

	def __init__(self, client):
		pass

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

def lookup_region_id(station_id):
    """Look up the region_id of the region containing the station identified by station_id
    
    Args:
        station_id(str): the station_id, as returned by the character market orders api call
        
    Returns:
        region_id(str): the region_id of the region containing the specified station
    """

    lookup_dict = {'station_id': 'region_id',
                   60004588: 10000042,  # Rens
                   60008494: 10000043   # Amarr
                   }

    return lookup_dict[station_id]


if __name__ == '__main__':

	bot = evebot(config_file="config.yaml")
	bot.main()
