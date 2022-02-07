from dotenv import load_dotenv
from dotenv import find_dotenv
import os
load_dotenv(find_dotenv(".env"))

APIKEY = os.environ.get("APIKEY")
APISECRETKEY = os.environ.get("APISECRETKEY")
ACCESSTOKEN = os.environ.get("ACCESSTOKEN")
ACCESSTOKENSECRET = os.environ.get("ACCESSTOKENSECRET")