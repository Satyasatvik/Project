import os
import gradio as gr
from groq import Groq
import base64
from gtts import gTTS
from PIL import Image
import tempfile
from dotenv import load_dotenv

# Set GROQ API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ GROQ_API_KEY is missing! Did you set it in Hugging Face Secrets?")
else:
    print("✅ GROQ_API_KEY loaded successfully!")

# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return f"Image Encoding Error: {str(e)}"

# Function to talk to Groq LLM
def analyze_with_groq(query, encoded_image=None, model="llama-3.2-11b-vision-preview"):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [{"role": "user", "content": []}]
        messages[0]["content"].append({"type": "text", "text": query})

        if encoded_image:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
            })

        chat_completion = client.chat.completions.create(messages=messages, model=model)
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Analysis Error: {str(e)}"

# Text-to-speech using gTTS
def text_to_speech_with_gtts(input_text, output_filepath="final.mp3"):
    try:
        audioobj = gTTS(text=input_text, lang="en", slow=False)
        audioobj.save(output_filepath)
        return output_filepath
    except Exception as e:
        return f"Text-to-Speech Error: {str(e)}"

# Doctor-style prompt
system_prompt = (
    "You have to act as a professional doctor, I know you are not, but this is for learning purposes. "
    "With what I see, I think you have .... If you make a differential, suggest some remedies for them. "
    "Do not add any numbers or special characters in your response. Your response should be in one long paragraph. "
    "Always answer as if you are talking to a real person. Do not respond as an AI model in markdown; "
    "your answer should mimic that of an actual doctor, not an AI bot. Keep your answer concise (max 2 sentences). "
    "No preamble, start your answer right away please."
)

# Main logic
def process_inputs(text_input, image_file):
    if not GROQ_API_KEY:
        return "Error: Missing GROQ API Key", None

    if not text_input or text_input.strip() == "":
        return "Please describe your symptoms to proceed.", None

    query = system_prompt + " " + text_input

    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            image_file.save(temp_img.name)
            encoded_image = encode_image(temp_img.name)
        if "Image Encoding Error" in encoded_image:
            return encoded_image, None
        response = analyze_with_groq(query, encoded_image)
    else:
        response = analyze_with_groq(query)

    if "Analysis Error" in response:
        return response, None

    audio_path = text_to_speech_with_gtts(response)
    if "Text-to-Speech Error" in audio_path:
        return response, None

    return response, audio_path

# Gradio UI
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Textbox(label="Describe Your Symptoms", placeholder="e.g., I have a rash on my arm"),
        gr.Image(type="pil", label="Upload Image (Optional)")
    ],
    outputs=[
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Audio Response")
    ],
    title="🩺 AIcura"
)

# Launch the app with share=True
if __name__ == "__main__":
    iface.launch(share=True)  # Enable public sharing