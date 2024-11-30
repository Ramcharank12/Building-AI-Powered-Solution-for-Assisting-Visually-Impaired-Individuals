import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import os
import torch
import pytesseract
from gtts import gTTS
import tempfile
import time
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(dotenv_path='/Users/charan/Desktop/ann/app.env')

# Retrieve the API key from the environment variables
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("/Users/charan/Desktop/ann/app.env")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Set Tesseract command path for OCR (Update this path to the correct Tesseract installation location)
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"  # Correct Tesseract path

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI",
    layout="wide",
    page_icon="🤖",
)

# Streamlit Layout Custom Styling
st.markdown(
    """
    <style>
     .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #555;
        margin-top: -20px;
     }
    .subtitle {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .feature-header {
        font-size: 24px;
        color: #333;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title"> 👁️‍🗨️Vision AI👁️‍🗨️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transforming Lives with AI: Real-Time Scene Understanding, Object Detection, Personalized Assistance, and Converting text into audio output. </div>', unsafe_allow_html=True)

# Load Object Detection Model
@st.cache_resource
def load_object_detection_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

object_detection_model = load_object_detection_model()

def detect_objects(image, threshold=0.3, iou_threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image)
    predictions = object_detection_model([img_tensor])[0]
    keep = torch.ops.torchvision.nms(predictions['boxes'], predictions['scores'], iou_threshold)

    filtered_predictions = {
        'boxes': predictions['boxes'][keep],
        'labels': predictions['labels'][keep],
        'scores': predictions['scores'][keep],
    }
    return filtered_predictions

def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for label, box, score in zip(predictions['labels'], predictions['boxes'], predictions['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=5)
    return image

# Text extraction from image
def extract_text_from_image(uploaded_file):
    img = Image.open(uploaded_file)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text.strip() or "No text found in the image."

# Converting text to speech
def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")

# Converting image to bytes
def image_to_bytes(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    return [{"mime_type": uploaded_file.type, "data": bytes_data}]

# Function to call the Gemini AI for personalized assistance
def get_assistance_response(input_prompt, image_data):
    system_prompt = """
    You are a specialized AI that provides accessibility assistance to visually impaired individuals. Your goal is to:
    1. Describing images in clear and simple language.
    2. Detecting objects and obstacles to help with navigation.
    3. Offering personalized suggestions based on the image content.
    4. Extracting and reading text from images clearly.
    """

    # Combine the system prompt with the user input prompt
    full_prompt = f"{system_prompt}\n{input_prompt}"
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([full_prompt, image_data[0]])
    return response.text

# UI Design
st.sidebar.image(r"/Users/charan/Desktop/ai.jpg", use_container_width=True)
st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Image:", type=['jpg', 'jpeg', 'png', 'webp'])

# Display uploaded image on the main page (right-hand side)
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# Declaring the features
st.markdown(""" 
### Features 
- 🏞️ **Scene Analysis**: Describe the content of an image in brief and detailed. 
- 🚧 **Object Detection**: Highlight objects and obstacles for navigation.
- 🤖 **Personalized Assistance**: Generate context-aware suggestions.
- 📝 **Text-to-Speech**: Convert text into audio outputs. 
""")

tab1, tab2, tab3, tab4 = st.tabs(["Scene Analysis", "Object Detection", "Assistance", "Text-to-Speech"])

# Scene Analysis Tab
with tab1:
    st.subheader("🏞️ Scene Analysis")
    if uploaded_file:
        with st.spinner("Analyzing Image..."):
            image_data = image_to_bytes(uploaded_file)
            user_prompt = "Describe this image in detail, briefly, with more text, clearly and concisely for visually impaired individuals."
            response = get_assistance_response(user_prompt, image_data)
            st.write(response)
            text_to_speech(response)

# Object Detection Tab
with tab2:
    st.subheader("🚧 Object Detection")
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            predictions = detect_objects(image)
            if predictions:
                image_with_boxes = draw_boxes(image.copy(), predictions)
                st.image(image_with_boxes, caption="Objects Highlighted", use_container_width=True)
            else:
                st.write("No objects detected in the image.")
        except Exception as e:
            st.error(f"Error processing the image: {e}")

# Assistance Tab
with tab3:
    st.subheader("🤖 Personalized Assistance")
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with st.spinner("Analyzing for personalized assistance..."):
            image_data = image_to_bytes(uploaded_file)
            user_prompt = "Provide detailed assistance based on the uploaded image."
            response = get_assistance_response(user_prompt, image_data)
            st.write(response)
            text_to_speech(response)

# Text-to-Speech Tab
with tab4:
    st.subheader("📝 Text Extraction and Speech")
    if uploaded_file:
        text = extract_text_from_image(uploaded_file)
        st.write(f"Extracted Text: {text}")
        if text:
            text_to_speech(text)
