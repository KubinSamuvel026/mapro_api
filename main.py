from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import EmotionHistory, User
import requests
import os

# ─────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────
Base.metadata.create_all(bind=engine)

app = FastAPI()

# ─────────────────────────────────────────
# HUGGING FACE CONFIG
# ─────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = (
    "https://api-inference.huggingface.co/models/"
    "j-hartmann/emotion-english-distilroberta-base"
)

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ─────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────
class RegisterUser(BaseModel):
    name: str
    email: str
    password: str


class LoginUser(BaseModel):
    email: str
    password: str


class TextInput(BaseModel):
    text: str


# ─────────────────────────────────────────
# MOTIVATION FUNCTION
# ─────────────────────────────────────────
def get_motivation(emotion):

    motivation_dict = {
        "joy":
            "You seem happy today. Keep spreading positivity and enjoy this beautiful moment in your life.",

        "sadness":
            "It’s okay to feel sad sometimes. Tough times never stay forever. Be kind to yourself.",

        "anger":
            "Take a deep breath. Your emotions are valid, but peace gives better solutions than anger.",

        "fear":
            "Fear is temporary. You are stronger than your worries. Take one step at a time.",

        "surprise":
            "Life is full of unexpected moments. Stay open and trust your ability to adapt.",

        "disgust":
            "Some situations are uncomfortable, but they help us understand what truly matters.",

        "neutral":
            "You seem calm today. Take care of yourself and keep moving forward."
    }

    return motivation_dict.get(
        emotion.lower(),
        "Stay positive. Every emotion teaches something important."
    )


# ─────────────────────────────────────────
# PREDICT EMOTION
# ─────────────────────────────────────────
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

        print("HF STATUS:", response.status_code)
        print("HF RESPONSE:", response.text)

        if response.status_code != 200:
            return {
                "emotion": "neutral",
                "confidence": 0,
                "motivation": "Emotion service temporarily unavailable"
            }

        try:
            result = response.json()
        except Exception:
            return {
                "emotion": "neutral",
                "confidence": 0,
                "motivation": "Could not process emotion"
            }

        print("RESULT:", result)

        if isinstance(result, list) and len(result) > 0:

            prediction = max(
                result[0],
                key=lambda x: x["score"]
            )

            emotion = prediction["label"]
            confidence = round(
                prediction["score"] * 100,
                2
            )

            motivation = get_motivation(emotion)

            # Save history
            db: Session = SessionLocal()

            new_record = EmotionHistory(
                text=text,
                emotion=emotion,
                confidence=prediction["score"]
            )

            db.add(new_record)
            db.commit()

            return {
                "emotion": emotion,
                "confidence": confidence,
                "motivation": motivation
            }

        return {
            "emotion": "neutral",
            "confidence": 0,
            "motivation": "Could not analyze emotion"
        }

    except Exception as e:
        print("FULL ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
# ─────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────
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