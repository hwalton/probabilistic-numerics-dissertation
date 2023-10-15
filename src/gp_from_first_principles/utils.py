#from dotenv import load_dotenv


#load_dotenv('/.env')
developer = True


#env = os.environ['ENV']

def debug_print(text):
    if developer:
        print(text)