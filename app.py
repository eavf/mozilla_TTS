from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from dotenv import load_dotenv
import os


# Load the .env file
load_dotenv()

# Load the API key from the environment variable
api_key = os.getenv("VGK2")

# Set Hugging Face token as an environment variable
os.environ["HF_HOME"] = "/HF_cache"  # Optional: Specify custom cache folder
os.environ["HF_TOKEN"] = api_key

cv_15 = load_dataset("mozilla-foundation/common_voice_15_0", "hi", split="train")
batch_sampler = BatchSampler(RandomSampler(cv_15), batch_size=32, drop_last=False)
dataloader = DataLoader(cv_15, batch_sampler=batch_sampler)
cv_15 = load_dataset("mozilla-foundation/common_voice_15_0", "hi", split="train", streaming=True)

print(next(iter(cv_15)))
