Generative AI-based Assistive Application

This project is an AI-powered assistive application designed to help visually impaired individuals. It uses Generative AI (Gemini) to understand scenes, summarize content, and provide speech output.

Features
	•	Object and obstacle detection
	•	Scene description using Generative AI
	•	Text summarization
	•	Text-to-speech conversion
	•	User-friendly Streamlit interface

Technologies Used
	•	Python
	•	Streamlit for web interface
	•	Google Gemini API (Generative AI)
	•	Text-to-Speech (gTTS)
	•	OpenCV for image/video processing

Setup
	1.	Clone the repository
     git clone https://github.com/Ramcharank12/Building-AI-Powered-Solution-for-Assisting-Visually-Impaired-Individuals.git
     cd Building-AI-Powered-Solution-for-Assisting-Visually-Impaired-Individuals
	2. Set up environment variable
     Create a file named GenAI.env and add your Gemini API key:
     GEMINI_API_KEY=your_real_api_key_here
  3. Run the app
     streamlit run GenAI.py
     
How it works
	•	Upload or capture an image or video
	•	The app analyzes it using Generative AI
	•	Provides description or summarized content
	•	Outputs audio feedback to help the user

