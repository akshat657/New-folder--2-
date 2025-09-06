import os
from dotenv import load_dotenv
load_dotenv() 
print("Loaded GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))