from google import genai
from dotenv import load_dotenv
import os
load_dotenv()
client=genai.Client(api_key=os.getenv("API_KEY"))

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a one-sentence summary of vector embeddings."
)
print(resp)