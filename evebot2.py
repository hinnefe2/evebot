"""Module to automate trade-oriented interactions with the EVE Online client"""

import datetime
import logging
import time
import yaml

import cv2
import numpy as np
import pyautogui as pg
import pytesseract as pt

from exceptions import WindowNotFoundException
from exceptions import TemplateNotFoundException

class evebot:
    """A class to handle logging in to the EVE client, and the automated 
    updating and entering of market ordrs"""

    def __init__(self, config_file, database):

        # read in API keys
        with open(config_file) as infile:
            self.config = yaml.load(infile.read())

        # instantiate local order, transaction, inventory caches
        self.cached_orders = CachedCharacterOrders(self.config['credentials'])
        self.cached_txns = CachedCharacterTxns(self.config['credentials'])
        self.cached_inventory = CachedCharacterInventory(self.config['credentials'])

        # instantiate the class which interacts with the client
        self.client = ClientManager(self.config)

        # instantiate the market-interacting component classes
        self.updater = OrderUpdater(self.client)
        self.seller = ItemSeller(self.client)
        self.buyer = ItemBuyer(self.client)


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
        self.logger = logging.getLogger(__name__)

    
    def _wait_for_window(window_name, window_img_path, timeout=10):
        """Wait for the window identified by window_img to load
    
        Args:
            window_name(str): the name of the window, for logging purposes
            window_img_path(str): the path the the template image which identifies the window
            timeout(int): the number of seconds to wait for the window
    
        Raises:
            WindowNotFoundException: if the specified window isn't detected after the timeout period
        """
    
        self.logger.debug('Waiting for %s to load', window_name)
    
        t_start = dt.datetime.now()
        t_wait = dt.timedelta(seconds=timeout)
    
        while dt.datetime.now() < t_start + t_wait:
    
            try:
                self.find_template(window_img_path)
                self.logger.debug('%s window ready', window_name)
                return

            except TemplateNotFoundException:
                time.sleep(5)
    
        # should only get here if we timed out
        self.logger.debug('Could not find %s window', window_name)
    
        raise WindowNotFoundException()


    def login(self):
        """Log in to the EVE client
        
        Returns:
            success(bool): whether or not the login was succesfull"""
        
        # TODO: put image paths in config file

        try:
            # start the launcher
            os.startfile(config['launcher_path'])
            _wait_for_window('EVE launcher', 'images/launcher-header.png', 60)
            self.logger.debug('started launcher')
    
            # start the EVE client
            self.click_template('images/launcher-login-small.png')
            _wait_for_window('Character selection', 'images/client-theoname.png', 60)
            self.logger.debug('started client')
        
            # log in trading character
            self.click_template('images/client-theoface-small.png')
            _wait_for_window('In-game GUI', 'images/client-ingame-gui.png', 60)
            self.logger.debug('logged in character')
        
            # kill the launcher (it uses lots of CPU for some reason)
            os.system("taskkill /im evelauncher.exe")
            self.logger.debug('killed launcher')

            return True

        except WindowNoteFoundException:
            self.logger.error('Failed to log in')

            return False
    

    def logout(self):
        """Close the EVE client and launcher"""

        # close all windows, in case any are open
        self.close_all_windows()

        # kill the EVE client process
        os.system("taskkill /im exefile.exe")
        self.logger.debug('killed EVE client')


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

    
    def find_template(self, template, threshold=0.9, how=cv2.TM_CCOEFF_NORMED):
        """Find the location of the provided template image on the screen.

        Args:
            template(str): path to image file
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
            self.logger.error('Could not find template %s', template_path)
            raise TemplateNotFoundException
    
        (_, maxMatch, _, maxLoc) = cv2.minMaxLoc(result_arr)
        logger.debug('Found template %s at %s with confidence %s', template_path, maxLoc, maxMatch)
    
        return np.array(maxLoc)


    def click_template(self, template, button='left'):
        """Click on the region of the screen that matches the provided template"""

        # find the top left corner of the image
        img_pos = find_template(template)
    
        # find the center of the image as top left corner + w/2, + h/2
        img = cv2.imread(img_path)
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

        self.logger.debug("Opened client window '%s'", window_name)


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
            ocr_text(str): the text of the row, as identified by OCR
        """
        # TODO: refactor this method
        # TODO: put this in config file
        name_w = 284
        row_y_offset = 17
    
        price_x_offset = 405
        price_w = 100
    
        name_top_left = position
        name_bottom_right = (row_top_left[0] + name_w, row_top_left[1] + row_y_offset)
    
        # take a screenshot of the 'name' column in the given row, then blow it up 5x
        name_snap = pg.screenshot(region=wh_from_pos(name_top_left, name_bottom_right))
        name_snap_x5 = name_snap.resize(np.array(name_snap.size) * 5, PIL.Image.ANTIALIAS)
        name_text = pt.image_to_string(name_snap_x5, config='letters')
    
        price_top_left = (row_top_left[0] + price_x_offset, row_top_left[1])
        price_bottom_right = (row_top_left[0] + price_x_offset + price_w, row_top_left[1] + row_y_offset)
    
        # take a screenshot of the 'price' column in the given row, then blow it up 5x
        price_snap = pg.screenshot(region=wh_from_pos(price_top_left, price_bottom_right))
        price_snap_x5 = price_snap.resize(np.array(price_snap.size) * 5, PIL.Image.ANTIALIAS)
        price_text = pt.image_to_string(price_snap_x5, config='letters')
    
        # strip commas, periods, spaces out of price_text, then extract just numbers
        price_text = price_text.replace('.', '').replace(',', '').replace(' ', '')
        price_text = str(re.search('^([0-9]*).*', price_text).groups(1)[0])
        price_float = float(price_text) / 100
    
        logger.debug('Extracted (typename, price) : (%s, %.2f)', name_text, price_float)
    
        return (name_text, price_text)


class OrderUpdater:
    """A class to automate the updating of existing market orders"""

    def __init__(self, client):
        self.client = client

    def open_orders_screen(self):
        """Open the orders screen so that orders can be updated"""
        pass

    def modify_order(self, position):
        """Update the order at the specified position"""
        pass

    def match_ocr_text(self):
        """Try to match the OCR'd order text to known orders from the API / cache"""
        pass

    def main(self):
        """Update all orders"""
        pass


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
