from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class Query(BaseModel):
    cuisine: str
    location: str
    price_range: str | None = None
    dietary: str | None = None

def generate_restaurant_response(query: Query):
    prompt = f"""
    You are a restaurant recommendation agent.

    User preferences:
    - Cuisine: {query.cuisine}
    - Location: {query.location}
    - Price range: {query.price_range}
    - Dietary needs: {query.dietary}

    Provide:
    - 3 restaurant recommendations
    - A short description for each
    - Why it matches the user's preferences
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


@app.post("/recommend")
def recommend_restaurants(query: Query):
    result = generate_restaurant_response(query)
    return {"recommendations": result}
