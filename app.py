from flask import Flask, request, jsonify
from flask_cors import CORS
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline, DistilBertTokenizer, DistilBertForQuestionAnswering, MarianMTModel, MarianTokenizer
import os
import traceback
import uuid
import cv2
import re
from gtts import gTTS
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import send_from_directory

app = Flask(__name__)
CORS(app)

# Set the path to FFmpeg
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

UPLOAD_FOLDER = 'uploads'
FRAME_FOLDER = 'frames'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

# Load models
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
qa_model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

def load_translation_model(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

@app.route('/uploads', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    transcription = transcribe_video(video_path)

    if isinstance(transcription, dict):  # If there's an error in transcription
        return jsonify(transcription), 500

    transcription_file_path = os.path.join(UPLOAD_FOLDER, "extracted_text.txt")
    with open(transcription_file_path, 'w') as f:
        f.write(transcription)

    print(f"Transcription saved to: {transcription_file_path}")

    summary = summarize_transcription(transcription)
    top_keywords = extract_top_keywords(transcription, top_n=5)

    # Call key moments extraction here, limiting to top 10 frames
    keyword_frames = extract_frames(video_path, transcription, top_keywords, max_frames=10)

    return jsonify({
        "transcription": transcription,
        "summary": summary,
        "keywords": top_keywords,
        "keyMoments": keyword_frames  # Update key moments key
    })

@app.route('/qa', methods=['POST'])
def qa():
    data = request.json
    question = data.get('question', '')
    target_lang = data.get('target_lang', 'en')

    with open(os.path.join(UPLOAD_FOLDER, "extracted_text.txt"), 'r') as file:
        context = file.read()

    answer = answer_question(question, context)

    if target_lang != 'en':
        translation_model, translation_tokenizer = load_translation_model('en', target_lang)
        answer = translate_text(answer, translation_model, translation_tokenizer)

    audio_file = text_to_speech(answer, target_lang)

    return jsonify({'answer': answer, 'audio_file': audio_file})

def transcribe_video(video_path):
    audio_path = os.path.join(UPLOAD_FOLDER, "temp_audio.wav")
    extract_audio_from_video(video_path, audio_path)

    if not os.path.exists(audio_path):
        return {"error": "Audio extraction failed."}

    if os.path.getsize(audio_path) == 0:
        print("Audio file is empty.")
        return {"error": "Extracted audio file is empty."}

    print(f"Transcribing audio from: {audio_path}")

    try:
        result = whisper_model.transcribe(audio_path)
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

def extract_top_keywords(transcription, top_n=5):
    # Basic implementation using TF-IDF to extract top N keywords
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([transcription])
    indices = X[0].nonzero()[1]
    tfidf_scores = [(vectorizer.get_feature_names_out()[i], X[0, i]) for i in indices]
    sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [keyword for keyword, score in sorted_keywords]

def extract_frames(video_path, transcription, keywords, max_frames=10):
    keyword_frames = []
    # Use OpenCV to extract frames where keywords appear (this is a placeholder implementation)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Placeholder logic for selecting frames based on keywords (you'll need to implement actual logic here)
        if i % (frame_count // max_frames) == 0:
            frame_filename = os.path.join(FRAME_FOLDER, f'frame_{i}.jpg')
            cv2.imwrite(frame_filename, frame)
            keyword_frames.append(f'frame_{i}.jpg')  # Append the frame filename for later retrieval

    cap.release()
    return keyword_frames

def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = qa_model(input_ids, attention_mask=attention_mask)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.decode(input_ids[0][answer_start:answer_end], skip_special_tokens=True)

    return answer

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang)
    audio_file_name = f"answer_{uuid.uuid4()}.mp3"
    audio_file_path = os.path.join(UPLOAD_FOLDER, audio_file_name)
    tts.save(audio_file_path)
    return audio_file_name
@app.route('/uploads/<path:filename>', methods=['GET'])
def send_audio(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/frames/<path:filename>', methods=['GET'])
def send_frame(filename):
    return send_from_directory(FRAME_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)

