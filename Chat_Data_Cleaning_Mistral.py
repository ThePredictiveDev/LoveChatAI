import re
import json

# Define regex patterns
TIMESTAMP_PATTERN = r"^\d{1,2}/\d{1,2}/\d{2,4},? \d{1,2}:\d{2} - "
MEDIA_PATTERN = r"<Media omitted>"
SYSTEM_MESSAGES = [
    "Messages and calls are end-to-end encrypted.",
    "changed the group description",
    "changed the group name",
    "was added",
    "were added",
    "left",
    "removed",
    "created the group",
    "You deleted this message",
    "This message was deleted"
]

def clean_message(line):
    """Removes timestamps, system messages, and unnecessary elements."""
    line = re.sub(TIMESTAMP_PATTERN, "", line).strip()
    if any(msg in line for msg in SYSTEM_MESSAGES) or line in ["", MEDIA_PATTERN]:
        return None
    line = re.sub(r"http\S+", "<LINK>", line)  # Replace links with placeholder
    return line

def parse_whatsapp_chat(file_path, user_name, partner_name, output_file):
    """Parses WhatsApp chat and structures it for fine-tuning in Mistral-7B format."""
    conversations = {"messages": []}  # Format based on Mistral
    last_speaker = None
    current_message = None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_message(line)
            if not line:
                continue

            # Check for user message pattern
            match = re.match(r"^(.+?): (.+)$", line)
            if match:
                speaker, message = match.groups()
                role = "user" if speaker == partner_name else "assistant"
                message = message.replace("<Media omitted>", "").strip()

                if role == last_speaker:
                    # Append to last message if same speaker
                    current_message["content"] += f" {message}"
                else:
                    # Store previous message before adding new
                    if current_message and current_message["content"]:
                        conversations["messages"].append(current_message)

                    # Add new message
                    current_message = {"role": role, "content": message}
                    last_speaker = role
            else:
                # Append to the last message (continuation)
                if current_message:
                    current_message["content"] += f" {line}"

    # Append last message to conversations
    if current_message and current_message["content"]:
        conversations["messages"].append(current_message)

    # Save structured chat data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=4, ensure_ascii=False)

    print(f"✅ Processed WhatsApp chat and saved to {output_file}")

# Example Usage:
whatsapp_chat_file = ''# Replace with your actual file path

output_json = "structured_chat_data.json"
user_name = "Devansh Garg"  # Your name in chat
partner_name = ""  # Your partner's name in chat

parse_whatsapp_chat(whatsapp_chat_file, user_name, partner_name, output_json)
