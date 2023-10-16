from dotenv import load_dotenv
import os

load_dotenv('../../.env')

env = os.environ.get('ENV')
def debug_print(text):
    if env == 'DEVELOPMENT':
        print(text)