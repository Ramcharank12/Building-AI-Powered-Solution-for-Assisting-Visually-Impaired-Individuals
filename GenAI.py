import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import pytesseract
from gtts import gTTS
import tempfile
import time

# ✅ Configure the Gemini API using Streamlit secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Set Tesseract command path for OCR (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"

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

# Function to call Gemini AI for assistance using image description and OCR
def get_assistance_response(input_prompt, uploaded_file):
    # Extract text from image (OCR)
    text_from_image = extract_text_from_image(uploaded_file)
    
    # Run object detection to describe objects
    image = Image.open(uploaded_file)
    predictions = detect_objects(image)
    object_desc = ""
    if predictions['labels'].numel() > 0:
        object_desc = f"Detected {len(predictions['labels'])} objects: " + ", ".join([str(l.item()) for l in predictions['labels']])
    else:
        object_desc = "No significant objects detected in the image."

    # Combine system prompt, user prompt, and image description
    system_prompt = """
    You are an AI specialized in assisting visually impaired users. Your goals:
    1. Describe images clearly and simply.
    2. Detect objects and obstacles.
    3. Give suggestions based on image content.
    4. Extract and read text from images.
    """

    full_prompt = (
        f"{system_prompt}\n"
        f"User Request: {input_prompt}\n"
        f"Image Description: {object_desc}\n"
        f"Extracted Text: {text_from_image}"
    )

    # Call Gemini text model
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([full_prompt])

    return response.text

# UI Design
st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Image:", type=['jpg', 'jpeg', 'png', 'webp'])

# Display uploaded image on the main page
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
            user_prompt = "Describe this image in detail, briefly, with more text, clearly and concisely for visually impaired individuals."
            response = get_assistance_response(user_prompt, uploaded_file)
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
            user_prompt = "Provide detailed assistance based on the uploaded image."
            response = get_assistance_response(user_prompt, uploaded_file)
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
