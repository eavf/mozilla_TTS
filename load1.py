import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import soundfile as sf
import numpy as np

# Load the .env file
load_dotenv()

# Load the API key from the environment variable
api_key = os.getenv("VGK2")

# Log in using your Hugging Face token
if api_key:
    login(api_key)
else:
    raise EnvironmentError("API key is missing. Ensure the 'VGK2' environment variable is set.")

# Create output directory
output_dir = ".\\filtered_common_voice"
os.makedirs(output_dir, exist_ok=True)

# Load the Slovak dataset in streaming mode
dataset = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "sk",
    split="train",
    trust_remote_code=True,
    streaming=True  # Enable streaming mode to handle large datasets
)

# Metadata file
metadata_file = os.path.join(output_dir, "metadata.csv")
processed_count = 0

# Open the metadata file for writing
with open(metadata_file, "w", encoding="utf-8") as f:
    f.write("client_id|sentence|audio_path\n")  # Write the header

    for i, sample in enumerate(dataset):
        try:
            # Extract relevant fields
            client_id = sample.get("client_id", "unknown")
            sentence = sample.get("sentence", "").strip()
            audio_data = sample.get("audio", None)

            # Check if audio_data is valid
            if not audio_data or not isinstance(audio_data.get("array", None), np.ndarray):
                print(f"Skipping invalid audio data for sample {i}.")
                continue

            audio_array = audio_data["array"]
            sampling_rate = audio_data["sampling_rate"]

            # Check for valid audio array and sampling rate
            if audio_array.size == 0 or sampling_rate <= 0:
                print(f"Skipping empty or invalid audio sample {i}.")
                continue

            # Create a subdirectory based on the first 2 characters of client_id
            subdir = os.path.join(output_dir, client_id[:2])
            os.makedirs(subdir, exist_ok=True)

            # Generate a unique output file path
            audio_path = os.path.join(subdir, f"{client_id}_{i}.wav")

            # Save the audio file
            sf.write(audio_path, audio_array, sampling_rate)

            # Write metadata
            f.write(f"{client_id}|{sentence}|{audio_path}\n")
            processed_count += 1

            # Print progress every 100 files
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")

        except Exception as e:
            print(f"Error processing sample {i}: {e}")

print(f"Filtered dataset saved at {output_dir}. Metadata saved at {metadata_file}.")
print(f"Total processed files: {processed_count}")
