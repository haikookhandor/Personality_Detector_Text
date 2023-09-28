# Personality_Detector_Text

This project aims to classify a person's behavior into one of five major behavior types. It leverages text as an input and gives output as the type of behavior.

### Directions of use:

1. In order to build a frontend, a partial and naive facebook parser has been made. The webscraper is located in fb_webscraper.py. It requires your login credentials and profile url to be in the yaml file fb_login_creds.yaml.

Run the webscraper by: python fb_webscraper.py

This opens a Selenium automated browser that will login to your Facebook account. 

2. Train models: model.py

Run and train the models by: python model.py
