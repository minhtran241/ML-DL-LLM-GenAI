import os
from dotenv import load_dotenv


def load_config():
    load_dotenv()
    google_api_key = os.environ["GOOGLE_API_KEY"]
    return google_api_key
