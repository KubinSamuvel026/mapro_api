from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import EmotionHistory, User
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


def get_motivation(emotion):

    motivation_map = {
        "joy":
            "You seem happy today. Keep spreading positivity and enjoy the moment.",

        "sadness":
            "It’s okay to feel sad. Better days will come. Be gentle with yourself.",

        "anger":
            "Take a deep breath. Calmness helps create better solutions.",

        "fear":
            "You are stronger than your worries. Take one step at a time.",

        "surprise":
            "Unexpected moments can create new opportunities.",

        "neutral":
            "You seem calm today. Keep moving forward."
    }

    return motivation_map.get(
        emotion.lower(),
        "Stay strong. You can handle this."
    )


@app.post("/predict")
def predict(data: UserText):

    try:
        text = data.text

        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )

        print("HF STATUS:", response.status_code)
        print("HF RESPONSE:", response.text)

        if response.status_code != 200:
            return {
                "emotion": "neutral",
                "confidence": 0,
                "motivation": "AI service busy. Try again."
            }

        result = response.json()

        predictions = result[0]

        best_prediction = max(
            predictions,
            key=lambda x: x["score"]
        )

        emotion = best_prediction["label"]

        confidence = round(
            best_prediction["score"] * 100,
            2
        )

        motivation = get_motivation(emotion)

        db: Session = SessionLocal()

        history = EmotionHistory(
            text=text,
            emotion=emotion,
            confidence=best_prediction["score"]
        )

        db.add(history)
        db.commit()

        return {
            "emotion": emotion,
            "confidence": confidence,
            "motivation": motivation
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "emotion": "neutral",
            "confidence": 0,
            "motivation": str(e)
        }


@app.get("/history")
def get_history():

    db: Session = SessionLocal()

    records = db.query(
        EmotionHistory
    ).all()

    data = []

    for item in records:
        data.append({
            "id": item.id,
            "text": item.text,
            "emotion": item.emotion,
            "confidence": round(
                item.confidence * 100,
                2
            )
        })

    return data


@app.get("/analytics")
def analytics():

    db: Session = SessionLocal()

    records = db.query(
        EmotionHistory
    ).all()

    analytics_data = {
        "joy": 0,
        "sadness": 0,
        "fear": 0,
        "anger": 0,
        "surprise": 0,
        "neutral": 0
    }

    for item in records:

        emotion = item.emotion.lower()

        if emotion in analytics_data:
            analytics_data[emotion] += 1

    return analytics_data