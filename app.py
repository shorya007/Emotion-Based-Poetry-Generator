import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from googletrans import Translator

st.set_page_config(page_title="Emotion-Based Story Generator", page_icon="🎭", layout="wide")

# ✅ Cache model to avoid reloading every time
@st.cache_resource
def load_emotion_model():
    return load_model(r"D:\6th Sem\NLP\NLP_PROJ\FER_model.h5", compile=False)

model = load_emotion_model()

# Initialize state variables
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

translator = Translator()
selected_language = st.selectbox("Select Language", ["English", "Hindi", "Spanish", "French", "German"])

def translate_text(text, target_language):
    lang_codes = {"English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
    return translator.translate(text, dest=lang_codes[target_language]).text

# Emotion-based stories with images
def generate_story(emotion):
    stories = {
        "Angry": (
            "A mighty warrior once ruled the kingdom with an iron fist. One day, his sword was stolen, "
            "igniting a rage that shook the land. Storming through forests and mountains, he faced every obstacle "
            "in his path. But when he found the thief, it was a frightened child trying to defend her village. "
            "His anger melted into understanding, and he chose mercy over fury.",
            r"D:\6th Sem\NLP\NLP_PROJ\angry.jpg"
        ),
        "Disgust": (
            "A chef stumbled upon a mysterious, rotting fruit deep in the jungle. Intrigued by its hideous stench, "
            "he took it back to his kitchen. Though the smell was repelling, he crafted a dish that enchanted all who "
            "dared to taste it. It was a reminder that even in the most repulsive things, hidden beauty can be found.",
            r"D:\6th Sem\NLP\NLP_PROJ\disgust.jpg"
        ),
        "Fear": (
            "A girl ventured into the dark woods every night searching for her lost dog. Shadows whispered and the trees "
            "seemed to move. Her heart pounded like thunder, but she pressed on. One night, a howl broke the silence, and "
            "through the fog, her dog appeared—safe, wagging its tail. Her fear faded, replaced by fierce love and courage.",
            r"D:\6th Sem\NLP\NLP_PROJ\fear.jpeg"
        ),
        "Happy": (
            "A young artist lived in a small village, painting smiles on every wall. One day, she painted a rainbow across "
            "a dull, broken fence. Children gathered and danced beneath it. Her joy was so infectious that soon the whole "
            "town colored their homes. What started as one brushstroke of happiness transformed the entire village into a canvas of joy.",
            r"D:\6th Sem\NLP\NLP_PROJ\happy.jpg"
        ),
        "Neutral": (
            "A clockmaker spent every day tuning gears and fixing time. His life moved like clockwork—calm, routine, unchanging. "
            "But one evening, he found an old, broken pocket watch with a love letter hidden inside. Though his expression never "
            "changed, something inside him stirred—proving that even the most neutral lives have untold stories ticking within.",
            r"D:\6th Sem\NLP\NLP_PROJ\neutral.jpg"
        ),
        "Sad": (
            "A lost love letter was found in an attic, sealed but never sent. It spoke of a love that was pure but interrupted by war. "
            "The woman who wrote it had passed, never knowing if her words reached him. But now, decades later, her grandson delivered it "
            "to the old soldier—who wept, knowing she had always loved him.",
            r"D:\6th Sem\NLP\NLP_PROJ\sad.jpg"
        ),
        "Surprise": (
            "A scientist accidentally spilled chemicals on a blueprint, creating a strange symbol. Curious, she followed the pattern and "
            "discovered it mirrored the layout of ancient ruins. Venturing there, she unlocked a hidden chamber that revealed a message from "
            "a lost civilization. Her tiny accident had uncovered a secret buried for millennia.",
            r"D:\6th Sem\NLP\NLP_PROJ\surprise.jpg"
        )
    }
    return stories.get(emotion, ("No story available.", r"D:\6th Sem\NLP\NLP_PROJ\happy.jpg"))


# --- UI ---
st.title("🎭 Emotion-Based Story Generator")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Start Camera"):
        st.session_state.camera_running = True
        st.session_state.cap = cv2.VideoCapture(0)
with col3:
    if st.button("Stop Camera"):
        st.session_state.camera_running = False
        if st.session_state.cap:
            st.session_state.cap.release()

frame_placeholder = st.empty()
story_placeholder = st.empty()
image_placeholder = st.empty()

# --- Processing ---
if st.session_state.camera_running and st.session_state.cap:
    ret, frame = st.session_state.cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = "Neutral"

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0) / 255.0

            predictions = model.predict(roi, verbose=0)
            detected_emotion = emotion_labels[np.argmax(predictions)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame_placeholder.image(frame, channels="BGR")

        story_text, image_path = generate_story(detected_emotion)
        translated = translate_text(story_text, selected_language)

        story_placeholder.markdown(f"### Story for **{detected_emotion}**:\n{translated}")
        image_placeholder.image(Image.open(image_path), caption=detected_emotion, width=300)

#To run: streamlit d:\6th Sem\NLP\NLP_PROJ\app.py