import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline, DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import traceback

app = Flask(__name__)
CORS(app)

# Set the path to FFmpeg
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

@app.route('/uploads', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    # Transcribe the video
    transcription = transcribe_video(video_path)

    if isinstance(transcription, dict):  # If there's an error in transcription
        return jsonify(transcription), 500

    # Save transcription to a file
    transcription_file_path = os.path.join(UPLOAD_FOLDER, "extracted_text.txt")
    with open(transcription_file_path, 'w') as f:
        f.write(transcription)

    print(f"Transcription saved to: {transcription_file_path}")

    # Generate a summary
    summary = summarize_transcription(transcription)

    return jsonify({"transcription": transcription, "summary": summary})

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get('question', '')

    # Load the extracted text from file
    with open(os.path.join(UPLOAD_FOLDER, "extracted_text.txt"), 'r') as file:
        context = file.read()

    answer = answer_question(question, context)

    # Generate audio for the answer
    audio_file = text_to_speech(answer)

    return jsonify({'answer': answer, 'audio_file': audio_file})

def transcribe_video(video_path):
    audio_path = os.path.join(UPLOAD_FOLDER, "temp_audio.wav")
    extract_audio_from_video(video_path, audio_path)

    if not os.path.exists(audio_path):
        return {"error": "Audio extraction failed."}

    # Check if the audio file is empty or invalid
    if os.path.getsize(audio_path) == 0:
        print("Audio file is empty.")
        return {"error": "Extracted audio file is empty."}

    print(f"Transcribing audio from: {audio_path}")

    try:
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        traceback.print_exc()
        return {"error": f"Transcription failed: {str(e)}"}

def extract_audio_from_video(video_path, output_audio_path):
    try:
        print(f"Extracting audio from {video_path} to {output_audio_path}...")
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(output_audio_path)
        print("Audio extracted successfully.")
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        traceback.print_exc()

def summarize_transcription(transcription):
    try:
        summary = summarizer(transcription, max_length=50, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        traceback.print_exc()
        return "Error during summarization."

def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = qa_model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    answer_tokens = input_ids[0][start_index:end_index]
    answer = qa_tokenizer.decode(answer_tokens)

    return answer.strip()

def text_to_speech(text):
    from gtts import gTTS
    audio_file = os.path.join(UPLOAD_FOLDER, "answer.mp3")
    tts = gTTS(text, lang='en')
    tts.save(audio_file)
    return audio_file

@app.route('/uploads/<filename>', methods=['GET'])
def get_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
