import requests
from selenium.webdriver import (Chrome, Firefox, ChromeOptions, FirefoxProfile)
import pymongo
import datetime
import yaml
import time
import random

# Basic web scraper for Facebook
class FBWebScraper():

    def __init__(self, my_email, my_password, my_profile_url, statuses=50, scroll_time=7, browser='Chrome'):

        self.my_email = my_email
        self.my_password = my_password
        self.my_profile_url = my_profile_url
        print("Hi") # check if this is working
        self.number_of_statuses = statuses 
        self.scroll_time = scroll_time
        self.mc = pymongo.MongoClient()
        self.db = self.mc['my-facebook-webscrape']
        self.fb_statuses = self.db['fb-statuses']

        self.set_browser(browser)

    def set_browser(self, browser):
        # CHROME
        if browser == 'Chrome':
            options = ChromeOptions();
            options.add_argument("--disable-notifications");
            self.browser = Chrome(options=options)

    def open_fb(self):
        url = 'https://www.facebook.com/' # Login using FB in Selenium
        self.browser.get(url)
        email = self.browser.find_element_by_id('email')
        password = self.browser.find_element_by_id('pass')
        email.send_keys(self.my_email)
        password.send_keys(self.my_password)
        self.browser.find_element_by_id("loginbutton").click()    

if __name__ == '__main__':
    with open('fb_login_creds.yaml', 'r') as stream: # Load credentials from yaml file
        try:
            y = yaml.load(stream, Loader=yaml.FullLoader)
            my_password = y['password']
            my_email = y['email']
            my_profile_url = y['profile_url']
        except yaml.YAMLError as exc:
            print(exc)
            print("hi") # check if this is working

    FBWS = FBWebScraper(
        my_email=my_email,
        my_password=my_password,
        my_profile_url=my_profile_url,
        browser='Chrome'
    )

    FBWS.open_fb()
