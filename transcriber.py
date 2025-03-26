import os
import sys
import base64
import requests
import json
from pathlib import Path

from typing import Tuple

def transcribe_audio_from_path(audio_file_path, api_key, keep_transliterate_english=False) -> Tuple[str, str]:
    """
    Process audio file with Gemini Flash 2.0 - transcribe and translate to Thai in one call.

    Args:
        audio_file_path (str): Path to the audio file
        api_key (str): Gemini API key

    Returns:
        tuple: (transcription, thai_translation) or (None, None) if processing fails
    """
    try:
        # Load and encode the audio file
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()

        # Base64 encode the audio data
        audio_b64 = base64.b64encode(audio_data).decode()

        # Get the MIME type of the audio file
        mime_type = get_mime_type(audio_file_path)

        prompt = ''

        if not keep_transliterate_english:
            prompt = """
You are an expert audio transcriber and translator. Your task is to accurately and completely transcribe the provided audio file, no matter the length. Once you have the full transcription, translate it into Thai.

**Specific Instructions:**

1.  **Complete Transcription:** Transcribe the entire audio file without omissions. Do not provide partial transcripts.
2.  **Number Conversion:** Convert all numerical digits within the transcription into their word equivalents. Use the context of the audio to determine if the number should be read as a whole number (e.g., '123' -> 'one hundred twenty-three') or as individual digits (e.g., '123' -> 'one two three').
3.  **Thai Translation:** Translate the complete transcribed text into Thai.
4. **Absolutely NO usage of the Thai character 'ๆ':** DO NOT use 'ๆ' anywhere in the translation. If a word is repeated (e.g., 'จริงๆ', 'ต่างๆ', 'มากๆ'), **write the word twice explicitly**. Examples:
   - 'จริงๆ' → 'จริงจริง'
   - 'ต่างๆ' → 'ต่างต่าง'
   - 'มากๆ' → 'มากมาก'
   - 'สีดำๆ' → 'สีดำดำ'
   - 'ไฟไหม้ๆ' → 'ไฟไหม้ ไฟไหม้'
   This is extremely important — you must avoid 'ๆ' entirely and instead duplicate the word in full.
5. **English to Thai Sound-Alike:** If there is no direct Thai translation for an English word (like a name), use a Thai word that sounds similar. For example:
   - 'Edward' → 'เอ็ดเวิร์ด'
   - 'Maple' → 'เมเปิ้ล'
6.  **Natural and Smooth Translation:** Ensure each translation is natural and smooth, as if spoken by a native Thai speaker.
7.  **No English Words in Translation:** Absolutely no English words should remain in the Thai translation and have only thai character and blank space (no special character such as ```'"<>?/\{}[]()-=_+!@#$%^``` and etc).
8.  **Output Format:** Present your output in the following format:

    Original Transcription: [Full transcribed text with numbers converted to words]

    Thai Translation: [Full Thai translation]
9. **You MUST follow all these instructions PERFECTLY. Especially rule #4 about NEVER using the character 'ๆ'.**

**Example:**

**Example 1**

**Audio:** "I have one hundred twenty-three apples, and the fire was burning, burning. Edward was there too."

**Output:**

Original Transcription: I have one hundred twenty-three apples, and the fire was burning burning. Edward was there too.

Thai Translation: ฉันมีแอปเปิลหนึ่งร้อยยี่สิบสามผล และไฟไหม้ ไฟไหม้ เอ็ดเวิร์ดก็อยู่ที่นั่นด้วย

**Example 2**

**Audio:** "I have twenty-four black computers numbered one, two, three, which are with Edward, who has twelve computers, and he is the one shouting 'burning! burning!'"

**Output:**

Original Transcription: I have twenty-four black computers numbered one, two, three, which are with Edward, who has twelve computers, and he is the one shouting 'burning! burning! but actually he is not burning'

Thai Translation: ฉันมีคอมพิวเตอร์หมายเลขหนึ่งสองสามสีดำดำอยู่ยี่สิบสี่เครื่อง ซึ่งอยู่กับเอ็ดเวิร์ดสิบสองเครื่อง และเขาคือคนที่กำลังกรีดร้องว่า ไฟไหม้ ไฟไหม้ แต่จริงจริงแล้วเขาไม่ได้ไฟไหม้

Please process the provided audio file and follow all instructions strictly."""
        else:
            prompt = """
You are an expert audio transcriber and translator. Your task is to accurately and completely transcribe the provided audio file, no matter the length. Once you have the full transcription, translate it into Thai.

**Specific Instructions:**

1.  **Complete Transcription:** Transcribe the entire audio file without omissions. Do not provide partial transcripts.
2.  **Number Conversion:** Convert all numerical digits within the transcription into their word equivalents. Use the context of the audio to determine if the number should be read as a whole number (e.g., '123' -> 'one hundred twenty-three') or as individual digits (e.g., '123' -> 'one two three').
3.  **Thai Translation:** Translate the complete transcribed text into Thai.
4. **Absolutely NO usage of the Thai character 'ๆ':** DO NOT use 'ๆ' anywhere in the translation. If a word is repeated (e.g., 'จริงๆ', 'ต่างๆ', 'มากๆ'), **write the word twice explicitly**. Examples:
   - 'จริงๆ' → 'จริงจริง'
   - 'ต่างๆ' → 'ต่างต่าง'
   - 'มากๆ' → 'มากมาก'
   - 'สีดำๆ' → 'สีดำดำ'
   - 'ไฟไหม้ๆ' → 'ไฟไหม้ ไฟไหม้'
   This is extremely important — you must avoid 'ๆ' entirely and instead duplicate the word in full.
5. **English to Thai Sound-Alike:** If there is no direct Thai translation for an English word (like a name), use a Thai word that sounds similar. For example:
   - 'Edward' → 'เอ็ดเวิร์ด'
   - 'Maple' → 'เมเปิ้ล'
6.  **Natural and Smooth Translation:** Ensure each translation is natural and smooth, as if spoken by a native Thai speaker.
7.  **Output Format:** Present your output in the following format:

    Original Transcription: [Full transcribed text with numbers converted to words]

    Thai Translation: [Full Thai translation]
8. **You MUST follow all these instructions PERFECTLY. Especially rule #4 about NEVER using the character 'ๆ'.**

**Example:**

**Example 1**

**Audio:** "I have one hundred twenty-three apples, and the fire was burning, burning. Edward was there too."

**Output:**

Original Transcription: I have one hundred twenty-three apples, and the fire was burning burning. Edward was there too.

Thai Translation: ฉันมีแอปเปิลหนึ่งร้อยยี่สิบสามผล และไฟไหม้ ไฟไหม้ Edward ก็อยู่ที่นั่นด้วย

**Example 2**

**Audio:** "Snow white and the seven dwarfs'

**Output:**

Original Transcription: Snow white and the seven dwarfs'

Thai Translation: Snow white และคนแคระทั้งเจ็ดคน

**Example 3**

**Audio:** "I have twenty-four black computers numbered one, two, three, which are with Edward, who has twelve computers, and he is the one shouting 'burning! burning!'"

**Output:**

Original Transcription: I have twenty-four black computers numbered one, two, three, which are with Edward, who has twelve computers, and he is the one shouting 'burning! burning! but actually he is not burning'

Thai Translation: ฉันมีคอมพิวเตอร์หมายเลขหนึ่งสองสามสีดำดำอยู่ยี่สิบสี่เครื่อง ซึ่งอยู่กับ Edward สิบสองเครื่อง และเขาคือคนที่กำลังกรีดร้องว่า ไฟไหม้ ไฟไหม้ แต่จริงจริงแล้วเขาไม่ได้ไฟไหม้

Please process the provided audio file and follow all instructions strictly."""

        # Build the API request payload
        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": prompt,
                    },

                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": audio_b64
                        }
                    }
                ]
            }]
        }

        # API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        # Make the API call
        headers = {'Content-Type': 'application/json'}
        print("Sending request to Gemini API...")
        response = requests.post(url, headers=headers,
                                 data=json.dumps(payload))

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None, None

        # Parse the response
        response_data = response.json()

        # Extract the content from the response
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            content = response_data['candidates'][0]['content']
            full_text = "".join([part['text']
                                for part in content['parts'] if 'text' in part])

            # Split the response into transcription and translation
            # This assumes Gemini structures its response clearly with headings or sections
            sections = full_text.split("\n\n")

            # Initialize transcription and translation
            transcription = ""
            translation = ""

            # Extract transcription and translation based on the new format
            for section in sections:
                if section.startswith("Original Transcription:"):
                    transcription = section.replace(
                        "Original Transcription:", "").strip()
                elif section.startswith("Thai Translation:"):
                    translation = section.replace(
                        "Thai Translation:", "").strip()

            return transcription, translation
        else:
            print("Error: Unexpected response format from API")
            print(f"Response: {response_data}")
            return None, None

    except Exception as e:
        print(f"Error during processing: {e}")
        return None, None


def get_mime_type(file_path):
    """
    Get the MIME type of a file based on its extension.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MIME type of the file
    """
    extension = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/x-m4a",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm"
    }

    return mime_types.get(extension, "application/octet-stream")


def main():
    """
    Main function to run the audio transcription and translation.
    """
    print("Audio Transcription and Translation using Gemini Flash 2.0")
    print("=========================================================")
    print("Note: This script requires a valid Gemini API key.")

    # Check for command line arguments
    if len(sys.argv) < 2:
        # If no command line argument, prompt for file path
        audio_file_path = input("Enter the path to your audio file: ")
    else:
        audio_file_path = sys.argv[1]

    # Validate file existence
    if not os.path.exists(audio_file_path):
        print(f"Error: File '{audio_file_path}' not found.")
        sys.exit(1)

    # Check file size - Gemini may have limits on input size
    file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
    if file_size_mb > 10:
        print(
            f"Warning: The file size is {file_size_mb:.2f} MB which may exceed API limits.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(0)

    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Please enter your Gemini API key: ")
        # Optionally save to environment for future use
        os.environ["GEMINI_API_KEY"] = api_key

    print(f"\nProcessing audio file: {audio_file_path}")
    print("This may take a moment depending on the file size...")

    # Process audio - transcribe and translate in one call
    transcription, thai_translation = transcribe_audio_from_path(audio_file_path, api_key)
    
    if transcription and thai_translation:
        print("\nTranscription:")
        print("--------------")
        print(transcription)

        print("\nThai Translation:")
        print("-----------------")
        print(thai_translation)

        # Save results to a file
        output_file = "transcription_results.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Original Transcription:\n")
            f.write("----------------------\n")
            f.write(transcription)
            f.write("\n\nThai Translation:\n")
            f.write("----------------\n")
            f.write(thai_translation)

        print(f"\nResults saved to '{output_file}'")
    else:
        print("\nProcessing failed.")
        print("\nTroubleshooting tips:")
        print("1. Make sure your API key is correct")
        print("2. Check if your audio file format is supported")
        print("3. Gemini Flash 2.0 may have limitations for audio processing")
        print("4. The file size may be too large (try with a smaller file)")
        print("5. Check your internet connection")
        print("6. Inspect the error messages above for more details")


if __name__ == "__main__":
    main()
