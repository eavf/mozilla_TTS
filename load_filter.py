from datasets import load_dataset, get_dataset_config_names, logging
import os
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
import soundfile as sf
import shutil


# Load the .env file
load_dotenv()

# Load the API key from the environment variable
api_key = os.getenv("VGK2")

# Log in using your Hugging Face token
if not api_key:
    raise ValueError("Error: VGK2 is not set or is invalid. Check your .env file.")
login(api_key)

logging.set_verbosity_debug()

# Check available configurations
configs = get_dataset_config_names("mozilla-foundation/common_voice_17_0", trust_remote_code=True)
print("Available configurations:", configs)

# Load the dataset for Slovak
dataset = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "sk",
    split="train",
    trust_remote_code=True
)

# Inspect the dataset fields
print ("dataset.column_names")
print(dataset.column_names)

# Display a sample of the dataset
print ("dataset[0]")
print(dataset[0])

# Filter Slovak entries (optional if you only loaded "sk")
slovak_dataset = dataset.filter(lambda x: x["locale"] == "sk")
print(f"Filtered Slovak dataset contains {len(slovak_dataset)} samples.")

# Debug output directory and free space
output_dir = ".\\filtered_common_voice"
os.makedirs(output_dir, exist_ok=True)
free_space = shutil.disk_usage(output_dir).free / (1024**2)
print(f"Free space in output directory: {free_space:.2f} MB")

# Save filtered dataset
metadata_path = os.path.join(output_dir, "metadata.csv")
processed_count = 0

with open(metadata_path, "w", encoding="utf-8") as metadata_file:
    for i, sample in enumerate(slovak_dataset):
        try:
            # Extract audio details
            audio_data = sample["audio"]
            sentence = sample["sentence"]
            client_id = sample["client_id"]

            # Generate unique output file path using index
            audio_path = os.path.join(output_dir, f"{client_id}_{i}.wav")

            # Check audio array and sampling rate
            if audio_data["array"] is None or len(audio_data["array"]) == 0:
                print(f"Sample {i} (client_id: {client_id}) has an empty audio array.")
                continue

            if audio_data["sampling_rate"] <= 0:
                print(f"Sample {i} (client_id: {client_id}) has an invalid sampling rate.")
                continue

            # Save the audio file
            sf.write(audio_path, audio_data["array"], audio_data["sampling_rate"])

            # Confirm file saved
            if os.path.exists(audio_path):
                print(f"Saved: {audio_path}")
                processed_count += 1
            else:
                print(f"Failed to save: {audio_path}")
                continue

            # Write metadata
            metadata_file.write(f"{client_id}.wav|{sentence}\n")

        except Exception as e:
            print(f"Error processing sample {i} (client_id: {client_id}): {e}")
            continue

print(f"Processed {processed_count} audio files out of {len(slovak_dataset)} samples.")
print(f"Metadata saved to: {metadata_path}")