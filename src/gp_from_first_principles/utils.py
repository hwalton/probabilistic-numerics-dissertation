from dotenv import load_dotenv
import os


load_dotenv('../../.env')
#developer = True


env = os.environ.get('ENV')

# def debug_print(text):
#     if developer:
#         print(text)
X = 1
def debug_print(text):
    if env == 'DEVELOPMENT':
        print(text)