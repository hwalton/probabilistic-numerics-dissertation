from dotenv import load_dotenv


load_dotenv('/.env')

#developer = True

env = os.environ['ENV']

def debug_print(text):
    if env == 'DEVELOPMENT':
        print(text)