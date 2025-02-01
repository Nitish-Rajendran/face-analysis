import cv2
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from deepface import DeepFace

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    face_detector = MTCNN(device=device)
    feature_extractor = AutoFeatureExtractor.from_pretrained("dima806/facial_emotions_image_detection")
    emotion_model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
    return face_detector, feature_extractor, emotion_model, device

def process_frame(frame, face_detector, feature_extractor, emotion_model, device):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = face_detector.detect(rgb_frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box]
            face = rgb_frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(face)
            inputs = feature_extractor(face_pil, return_tensors="pt")
            with torch.no_grad():
                outputs = emotion_model(**inputs)
                emotion_idx = outputs.logits.argmax(-1).item()
                emotion = emotion_model.config.id2label[emotion_idx]
                confidence = torch.softmax(outputs.logits, -1).max().item()
            try:
                analysis = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
                age = analysis[0]['age']
                gender = analysis[0]['dominant_gender']
            except:
                age = "Unknown"
                gender = "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Age: {age} | Gender: {gender} | {emotion}: {confidence:.2f}"
            cv2.putText(frame, label,
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)
    return frame


face_detector, feature_extractor, emotion_model, device = load_models()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break    
    processed_frame = process_frame(frame, face_detector, feature_extractor,
                                     emotion_model, device)
    cv2.imshow('Face Analysis', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

