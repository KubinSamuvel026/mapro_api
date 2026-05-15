from fastapi import FastAPI
from pydantic import BaseModel

from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import EmotionHistory, User
from database import Base

import requests
import os

Base.metadata.create_all(bind=engine)

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = (
    "https://api-inference.huggingface.co/models/"
    "j-hartmann/emotion-english-distilroberta-base"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


class RegisterUser(BaseModel):

    name: str
    email: str
    password: str


class LoginUser(BaseModel):

    email: str
    password: str


class UserText(BaseModel):

    text: str

from fastapi import HTTPException

@app.post("/predict")
def predict(data: TextInput):
    try:
        text = data.text

        response = requests.post(
            API_URL,
            json={"inputs": text},
            headers=HEADERS,
            timeout=20,
        )

        print("STATUS:", response.status_code)
        print("TEXT:", response.text)

        # Check failed response
        if response.status_code != 200:
            return {
                "emotion": "neutral",
                "confidence": 0,
                "motivation": "Model API temporarily unavailable",
            }

        # Safe JSON parsing
        try:
            result = response.json()
        except Exception:
            return {
                "emotion": "neutral",
                "confidence": 0,
                "motivation": "Invalid API response",
            }

        print("RESULT:", result)

        # Adjust according to your model response
        if isinstance(result, list) and len(result) > 0:
            prediction = result[0]

            emotion = prediction.get("label", "neutral")
            confidence = round(
                prediction.get("score", 0) * 100,
                2,
            )

            return {
                "emotion": emotion,
                "confidence": confidence,
                "motivation": f"You seem {emotion}. Stay strong.",
            }

        return {
            "emotion": "neutral",
            "confidence": 0,
            "motivation": "Could not analyze emotion",
        }

    except Exception as e:
        print("FULL ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
@app.get("/history")
def get_history():

    db: Session = SessionLocal()

    records = db.query(EmotionHistory).all()

    data = []

    for item in records:

        data.append({
            "id": item.id,
            "text": item.text,
            "emotion": item.emotion,
            "confidence": round(item.confidence * 100, 2)
        })

    return data


@app.get("/analytics")
def analytics():

    db: Session = SessionLocal()

    records = db.query(EmotionHistory).all()

    analytics_data = {
        "joy": 0,
        "sadness": 0,
        "fear": 0,
        "anger": 0,
        "surprise": 0
    }

    for item in records:

        emotion = item.emotion

        if emotion in analytics_data:

            analytics_data[emotion] += 1

    return analytics_data


@app.post("/register")
def register(user: RegisterUser):

    db: Session = SessionLocal()

    existing_user = db.query(User).filter(
        User.email == user.email
    ).first()

    if existing_user:

        return {
            "message": "Email already exists"
        }

    new_user = User(
        name=user.name,
        email=user.email,
        password=user.password
    )

    db.add(new_user)

    db.commit()

    return {
        "message": "Registration successful"
    }


@app.post("/login")
def login(user: LoginUser):

    db: Session = SessionLocal()

    existing_user = db.query(User).filter(
        User.email == user.email,
        User.password == user.password
    ).first()

    if not existing_user:

        return {
            "message": "Invalid credentials"
        }

    return {
        "message": "Login successful",
        "name": existing_user.name
    }