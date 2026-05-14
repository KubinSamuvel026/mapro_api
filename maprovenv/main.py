from fastapi import FastAPI
from pydantic import BaseModel

from transformers import pipeline

from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import EmotionHistory, User
from database import Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)
class RegisterUser(BaseModel):

    name: str

    email: str

    password: str
class UserText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: UserText):

    result = emotion_model(data.text)

    emotion = result[0][0]["label"]

    confidence = result[0][0]["score"]

    motivation = "Stay positive and keep moving forward."

    if emotion == "sadness":

        motivation = (
            "You are stronger than your current thoughts. "
            "Take one small step today."
        )

    elif emotion == "fear":

        motivation = (
            "Fear does not define your future. "
            "Take things one step at a time."
        )

    elif emotion == "anger":

        motivation = (
            "Take a pause and breathe slowly. "
            "Your feelings are temporary."
        )

    elif emotion == "joy":

        motivation = (
            "Keep enjoying the positive moments in your life."
        )

    db: Session = SessionLocal()

    new_record = EmotionHistory(
        text=data.text,
        emotion=emotion,
        confidence=confidence
    )

    db.add(new_record)

    db.commit()

    return {
        "emotion": emotion,
        "confidence": round(confidence * 100, 2),
        "motivation": motivation
    }
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
class LoginUser(BaseModel):

    email: str

    password: str


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