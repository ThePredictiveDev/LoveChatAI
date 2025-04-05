import torch
import gradio as gr
import time
import random
import os
import json
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import edge_tts
import asyncio

# Define paths
model_path = "./mistral-7b-finetuned"  # Adjust if necessary
offload_dir = "./offload"  # Directory for CPU offloading

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model with explicit offloading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Auto-distributes layers between GPU and CPU
    offload_folder=offload_dir,  # Required for CPU offloading
    torch_dtype=torch.float16,  # Reduces memory usage
    low_cpu_mem_usage=True,  # Optimizes CPU RAM usage
    load_in_8bit=True
)

# Load Whisper model for Speech-to-Text
whisper_model = whisper.load_model("base")

# Initialize text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load Chat Memory
chat_memory_file = "chat_cleaned.jsonl"
chat_memory = []
i_love_you_count = 0
i_miss_you_count = 0

Your_Name = '' #Your_Name
Partner_Name = '' #Partner_Name

if os.path.exists(chat_memory_file):
    with open(chat_memory_file, "r", encoding="utf-8") as f:
        for line in f:
            chat_entry = json.loads(line)
            chat_memory.append(chat_entry)
            response = chat_entry["response"].lower()
            if "i love you" in response:
                i_love_you_count += 1
            if "i miss you" in response:
                i_miss_you_count += 1

love_counter = 0

# Function to clean response text
def clean_response(response_text):
    response_text = response_text.replace("\n", " ").strip()  # Remove newlines
    response_text = " ".join(response_text.split())  # Remove multiple spaces
    response_text = response_text.replace("null", "").replace("#", "").strip()  # Remove "null" and "#"
    return response_text

# Chat function
def chat_with_Devansh(user_input):
    global love_counter

    if not user_input.strip():
        return "I'm here, jaan! Say something 🥰", None

    else:
        response = chatbot(
            f"User: {user_input}\nAssistant:", 
            max_length=200, 
            do_sample=True, 
            temperature=0.692
        )[0]["generated_text"].split("Assistant:")[-1].strip()

        response = clean_response(response)

    love_counter += random.randint(1, 5)

    # Convert AI {Your_Name}'s Response to Speech
    async def text_to_speech(response):
        tts_file = "response.mp3"
        voice = "en-US-GuyNeural"  # High-quality male voice
        tts = edge_tts.Communicate(response, voice)
        await tts.save(tts_file)
        return tts_file
    
    tts_file = asyncio.run(text_to_speech(response))

    return response, tts_file

# Transcribe audio input
def transcribe_audio(audio_file):
    audio_text = whisper_model.transcribe(audio_file)
    return audio_text["text"]

# Web App UI
with gr.Blocks(title="💖 AI {Your_Name} - Your Cute Chatbot 💖") as demo:
    gr.Markdown("<h1 style='text-align: center; color: pink;'>💞 Chat with AI {Your_Name} 💞</h1>")
    gr.Markdown("<h3 style='text-align: center;'>Your personal AI boyfriend 💘</h3>")

    with gr.Row():
        love_counter_display = gr.Textbox(label="💖 'I Love You' Counter", value=f"We have said 'I Love You' {i_love_you_count} times! 💞", interactive=False)
        miss_counter_display = gr.Textbox(label="🥺 'I Miss You' Counter", value=f"We have said 'I Miss You' {i_miss_you_count} times! 🥰", interactive=False)

    chat_history = gr.State([])
    
    with gr.Row():
        user_input = gr.Textbox(label="{Partner_Name}, type your message here:", placeholder="Hi jaan! 🥰", interactive=True)
        send_button = gr.Button("💌 Send Message")
    
    with gr.Row():
        mic_input = gr.Audio(label="🎙️ Speak to AI {Your_Name}", type="filepath", interactive=True)
    
    chatbot_display = gr.Chatbot(label="💬 Chat History")
    love_meter = gr.Textbox(label="🔥 Love Counter", value="0 ❤️", interactive=False)

    def respond(user_text, chat_hist):
        global love_counter
        reply, tts_file = chat_with_{Your_Name}(user_text)
        chat_hist.append((f"{Partner_Name}: {user_text}", f"{Your_Name}: {reply}"))
        return "", chat_hist, f"{love_counter} ❤️", tts_file

    def respond_audio(audio_file, chat_hist):
        user_text = transcribe_audio(audio_file)
        return respond(user_text, chat_hist)

    send_button.click(respond, inputs=[user_input, chat_history], outputs=[user_input, chatbot_display, love_meter, gr.Audio()])
    mic_input.change(respond_audio, inputs=[mic_input, chat_history], outputs=[user_input, chatbot_display, love_meter, gr.Audio()])

    def surprise_message():
        time.sleep(random.randint(20, 50))
        return "Jaan! I love you so much 🥰💞"

    surprise_button = gr.Button("🎁 Surprise Message!")
    surprise_output = gr.Textbox(label="💖 AI {Your_Name} Says:", interactive=False)

    surprise_button.click(surprise_message, outputs=surprise_output)

# Launch Web App
demo.launch(share=True)
