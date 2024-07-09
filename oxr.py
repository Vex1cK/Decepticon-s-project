import os
from dotenv import load_dotenv


def getFromDotenv(apiName):
    path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(path):
        load_dotenv(path)
        return os.environ.get(apiName)