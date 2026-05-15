from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import EmotionHistory, User

from transformers import pipeline

Base.metadata.create_all(bind=engine)

app = FastAPI()

# Load model once
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


class RegisterUser(BaseModel):
    name: str
    email: str
    password: str


class LoginUser(BaseModel):
    email: str
    password: str


class TextInput(BaseModel):
    text: str


def get_motivation(emotion):
    motivation = {
        "joy":
            "You seem happy today. Keep enjoying the moment and spread positivity.",

        "sadness":
            "Hard times pass. Be patient with yourself and remember brighter days come.",

        "anger":
            "Pause for a moment. Calmness helps make better decisions than anger.",

        "fear":
            "You are stronger than your worries. Take things one step at a time.",

        "surprise":
            "Unexpected moments can bring growth. Stay open and adaptable.",

        "disgust":
            "Not every experience feels good, but every experience teaches something.",

        "neutral":
            "You seem balanced today. Keep taking care of yourself."
    }

    return motivation.get(
        emotion.lower(),
        "Stay strong. Every feeling matters."
    )


@app.post("/predict")
def predict(data: TextInput):

    text = data.text

    try:
        result = emotion_classifier(text)

        predictions = result[0]

        best_prediction = max(
            predictions,
            key=lambda x: x["score"]
        )

        emotion = best_prediction["label"].lower()

        confidence = round(
            best_prediction["score"] * 100,
            2
        )

        motivation = get_motivation(emotion)

        db: Session = SessionLocal()

        new_record = EmotionHistory(
            text=text,
            emotion=emotion,
            confidence=best_prediction["score"]
        )

        db.add(new_record)
        db.commit()

        return {
            "emotion": emotion,
            "confidence": confidence,
            "motivation": motivation
        }

    except Exception as e:
        return {
            "emotion": "neutral",
            "confidence": 0,
            "motivation": f"Error: {str(e)}"
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