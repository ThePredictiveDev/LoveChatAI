LoveChatAI
==========

Export your WhatsApp CSV and use the simple plug-and-play scripts to create a web interface for your AI agent that talks just like your partner.

Introduction
------------

LoveChatAI is an open-source project that transforms your WhatsApp chat history into a personalized AI chatbot. By cleaning your exported chat data, fine-tuning a state-of-the-art Mistral-7B language model using LoRA, and deploying a web interface with Gradio, you can create an AI companion that mimics the conversational style of your partner.

Features
--------

-   Clean and structure WhatsApp chat logs for AI fine-tuning

-   Fine-tune the Mistral-7B model with LoRA using QLoRA techniques and 4-bit quantization for efficient training

-   Deploy a user-friendly Gradio web interface with text and voice interaction

-   Integrated speech-to-text (Whisper) and text-to-speech (Edge TTS) functionalities

-   Fun interactive elements including counters for phrases like "I love you" and "I miss you" and a surprise message button

Requirements
------------

-   Python 3.8 or higher

-   A GPU is recommended for model training (with CPU offloading options available)

-   Required libraries include torch, transformers, gradio, whisper, edge-tts, peft, datasets, and standard Python libraries such as re, json, and asyncio

Installation
------------

1.  Clone the repository using your preferred method.

2.  Install the required dependencies (for example, via pip) ensuring all the listed libraries are installed.

3.  Verify that your environment has the necessary resources (GPU or appropriate CPU offloading) for model fine-tuning.

Data Preparation
----------------

1.  Export your WhatsApp chat as a CSV or text file.

2.  In the **Chat_Data_Cleaning_Mistral.py** script, set the file path to your exported chat and fill in your user name and your partner's name.

3.  Run the script to clean the chat data. It will remove timestamps, system messages, and replace links with a placeholder. The output will be a structured JSON file (typically named structured_chat_data.json) formatted for Mistral-7B fine-tuning.

Model Fine-Tuning
-----------------

1.  Open **Model_Training_Mistral.py** and configure your Hugging Face token by setting the TOKEN variable.

2.  The script loads the base Mistral-7B model from Hugging Face ("mistralai/Mistral-7B-v0.1") and applies QLoRA with 4-bit quantization to optimize resource usage.

3.  The cleaned chat data is loaded from the JSON file and tokenized for training.

4.  LoRA is applied to the attention layers (targeting modules such as q_proj and v_proj) to fine-tune the model on your conversational data.

5.  The fine-tuned model is saved in the directory "mistral-7b-finetuned", ready for deployment.

Running the Web Interface
-------------------------

1.  In **Gradio_Web_App.py**, configure your settings by filling in your name (Your_Name) and your partner's name (Partner_Name).

2.  The script loads the fine-tuned model, integrates Whisper for transcribing audio input, and uses Edge TTS to convert AI responses into speech.

3.  The Gradio Blocks interface provides options for text input, audio input, and interactive features like the "Surprise Message" button and counters tracking "I love you" and "I miss you" phrases.

4.  Launch the Gradio app to interact with your personalized AI chatbot via a web interface that supports both text and voice communication.

Repository Structure
--------------------

-   .gitattributes -- Git configuration settings

-   Chat_Data_Cleaning_Mistral.py -- Cleans and structures WhatsApp chat logs for fine-tuning

-   Model_Training_Mistral.py -- Fine-tunes the Mistral-7B model using LoRA and QLoRA techniques

-   Gradio_Web_App.py -- Launches the Gradio web interface for chatting with your AI

-   README.md -- This documentation file

Contributing
------------

Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. Your contributions will help enhance the project and expand its capabilities.

License
-------

This project is open source. Covered under the MIT License

Conclusion
----------

LoveChatAI offers an innovative way to personalize AI interactions by replicating the unique conversational style of your partner. With straightforward scripts for data cleaning, model fine-tuning, and web interface deployment, you can easily transform your chat history into an engaging and interactive AI experience.

Enjoy building your AI companion and happy chatting!
