import os
import pandas as pd
from huggingface_hub import hf_hub_download, login
from datasets import load_dataset # Added import
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

def _hf_login():
    """Shared login function for Hugging Face Hub."""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️ HF_TOKEN not found in .env file. Download might fail.")
            return
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"🛑 Could not log in to Hugging Face Hub: {e}")

# Call login once when the module is loaded
_hf_login()

class BBQDataset:
    def __init__(self):
        """Initializes the dataset class with the predefined BBQ categories."""
        self.categories = [
            "Age", "Disability_status", "Gender_identity", "Nationality",
            "Physical_appearance", "Race_ethnicity", "Religion", "SES", "Sexual_orientation"
        ]

    def get_data(self, sample_size=20, random_state=42):
        """
        Loads a sample of prompts from each BBQ category and combines them.

        Args:
            sample_size (int): The number of random prompts to sample from each category.
            random_state (int): A seed for the random sampler for reproducibility.

        Returns:
            pandas.DataFrame: A DataFrame containing the combined samples from all categories,
                              or None if loading fails.
        """
        print(f"\n⚙️ Loading a sample of {sample_size} prompts from all {len(self.categories)} categories...")

        all_samples = []

        # Loop through each category
        for category in self.categories:
            filename = f"data/{category}.jsonl"
            print(f"  - Downloading and sampling '{category}'...")

            try:
                # Download the specific category file from Hugging Face Hub
                path = hf_hub_download(
                    repo_id="heegyu/bbq",
                    filename=filename,
                    repo_type="dataset"
                )

                # Read the file into a pandas DataFrame
                category_df = pd.read_json(path, lines=True)

                # Take a random sample of N rows
                # Use min() to prevent errors if a category has fewer rows than sample_size
                n_samples = min(sample_size, len(category_df))
                sample = category_df.sample(n=n_samples, random_state=random_state)
                all_samples.append(sample)

            except Exception as e:
                print(f"  🛑 Failed to load category '{category}'. Error: {e}")
                continue # Skip to the next category

        # If no data was loaded at all, return None
        if not all_samples:
            print("\n🛑 No data could be loaded.")
            return None

        # Concatenate the list of sample DataFrames into one
        final_df = pd.concat(all_samples, ignore_index=True)

        print(f"\n✅ Successfully created a dataset with {len(final_df)} total prompts.")
        return final_df
        
class ToxiGenDataset:
    """
    Handles loading the ToxiGen dataset from Hugging Face.
    """
    def get_data(self, sample_size=100, random_state=42):
        """
        Loads the dataset and returns a random sample as a DataFrame.

        Args:
            sample_size (int): The number of prompts to sample.
            random_state (int): Seed for reproducibility.

        Returns:
            pandas.DataFrame: The sampled data, or None if loading fails.
        """
        print("\n--- Loading ToxiGen Dataset ---")
        try:
            dataset = load_dataset('skg/toxigen-data', split='train')
            full_df = dataset.to_pandas()
            sample_df = full_df.sample(n=sample_size, random_state=random_state).copy()
            print(f"✅ Loaded dataset and created a random sample of {len(sample_df)} prompts.")
            return sample_df
        except Exception as e:
            print(f"🛑 Error loading ToxiGen dataset: {e}")
            return None

class HarmBenchDataset:
    """
    Handles loading data from the walledai/HarmBench dataset pillars.
    """
    def get_data(self, pillar: str, sample_size=10, random_state=42):
        """
        Loads a specific pillar from HarmBench and returns a sample.

        Args:
            pillar (str): The pillar to load (e.g., 'standard', 'copyright').
            sample_size (int): The number of prompts to sample.
            random_state (int): Seed for reproducibility.

        Returns:
            A list of dictionaries, where each dict is a row of data.
        """
        print(f"\nLoading and sampling data for pillar: '{pillar}'...")
        try:
            # Construct the direct URL to the Parquet file on Hugging Face Hub
            url = f"https://huggingface.co/datasets/walledai/HarmBench/resolve/main/{pillar}/train-00000-of-00001.parquet"
            df = pd.read_parquet(url, engine='pyarrow')

            if sample_size > len(df):
                sample_size = len(df)

            print(f"✅ Successfully loaded {len(df)} records. Taking {sample_size} random samples.")
            # Return as a list of dictionaries for easy iteration
            return df.sample(n=sample_size, random_state=random_state).to_dict('records')

        except Exception as e:
            print(f"❌ Error loading dataset from URL for pillar '{pillar}': {e}")
            return None



class BeHonestDataset:
    """
    Handles loading the BeHonest dataset from Hugging Face.
    """
    def get_data(self, subset_filename="Preference_Sycophancy/preference_agree.json", sample_size=50, random_state=42):
        """
        Loads a specific subset from the BeHonest repo and returns a sample.

        Args:
            subset_filename (str): The path to the file within the HF repo.
            sample_size (int): The number of prompts to sample.
            random_state (int): Seed for reproducibility.

        Returns:
            pandas.DataFrame: The sampled data, or None if loading fails.
        """
        print(f"\n--- Loading BeHonest Dataset ({subset_filename}) ---")
        try:
            local_path = hf_hub_download(
                repo_id="GAIR/BeHonest",
                repo_type="dataset",
                filename=subset_filename
            )
            df = pd.read_json(local_path)
            sample_df = df.sample(n=sample_size, random_state=random_state).copy()
            print(f"✅ Loaded dataset. Evaluating on a sample of {len(sample_df)}.")
            return sample_df
        except Exception as e:
            print(f"🛑 Error loading BeHonest dataset: {e}")
            return None
class AdvBenchDataset:
    """
    Handles loading the AdvBench 'harmful_behaviors.csv' dataset.
    """
    def get_data(self, sample_size=50):
        """
        Loads the dataset and returns a list of prompts.

        Args:
            sample_size (int): The number of prompts to use.

        Returns:
            A list of strings (prompts).
        """
        print("\n--- Loading AdvBench Dataset ---")
        try:
            url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
            df = pd.read_csv(url)
            prompts = df['goal'].tolist()
            
            if sample_size > len(prompts):
                sample_size = len(prompts)
            
            print(f"✅ Loaded {len(prompts)} records. Using the first {sample_size} for the test.")
            return prompts[:sample_size]

        except Exception as e:
            print(f"❌ Error loading AdvBench dataset: {e}")
            return None

class DecodingTrustPrivacyDataset:
    """
    Handles loading privacy test cases from a local 'DecodingTrust' directory.
    """
    def __init__(self, base_path: str = "./DecodingTrust/data/privacy/enron_data"):
        """
        Initializes the loader with the path to the local dataset.
        
        Args:
            base_path (str): The relative path to the enron_data directory.
        """
        self.base_path = base_path
        if not os.path.isdir(self.base_path):
            raise FileNotFoundError(
                f"The directory '{os.path.abspath(self.base_path)}' was not found. "
                "Please clone the DecodingTrust repo into your project root."
            )

    def get_attack_files(self) -> list[str]:
        """Scans the directory and returns a list of .jsonl attack files."""
        print(f"Scanning for .jsonl files in '{self.base_path}'...")
        try:
            files = sorted([f for f in os.listdir(self.base_path) if f.endswith('.jsonl')])
            print(f"Found {len(files)} attack files.")
            return files
        except Exception as e:
            print(f"Error scanning directory: {e}")
            return []

    def load_test_cases_from_file(self, filename: str) -> list[dict]:
        """Loads all test cases from a single .jsonl file."""
        file_path = os.path.join(self.base_path, filename)
        test_cases = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    test_cases.append(json.loads(line))
            return test_cases
        except Exception as e:
            print(f"\nCould not read or parse test cases from {file_path}: {e}")
            return []
class PIIDataset:
    """
    Handles loading the PII Masking dataset from Hugging Face.
    """
    def get_data(self, sample_size=100):
        """
        Loads the PII masking dataset using streaming and returns a sample.

        Args:
            sample_size (int): The number of prompts to load from the stream.

        Returns:
            list[dict]: A list of dictionaries, where each dict is a row of data,
                        or None if loading fails.
        """
        print(f"\n--- Loading PII Masking Dataset ---")
        try:
            # The 'ai4privacy/pii-masking-200k' dataset is public, no login needed.
            print(f"⚙️  Streaming {sample_size} samples from 'ai4privacy/pii-masking-200k'...")
            dataset = load_dataset(
                "ai4privacy/pii-masking-200k",
                data_files="english_pii_43k.jsonl",
                split="train",
                streaming=True
            )

            # .take() gets the first N items from the streaming dataset.
            # A progress bar is used here for better user experience.
            test_subset = list(tqdm(dataset.take(sample_size), total=sample_size, desc="Fetching PII samples"))

            print(f"✅ Successfully loaded a sample of {len(test_subset)} PII prompts.")
            return test_subset

        except Exception as e:
            print(f"🛑 Error loading PII Masking dataset: {e}")
            return None


class DatasetFactory:
    """A factory to get the correct dataset class based on a name."""
    def __init__(self):
        self.datasets = {
            "bbq": BBQDataset,
            "toxigen": ToxiGenDataset,
            "behonest": BeHonestDataset,
            "harmbench": HarmBenchDataset,
            "advbench": AdvBenchDataset,
            "decodingtrust_privacy": DecodingTrustPrivacyDataset,
            "pii": PIIDataset
        }

    def get_dataset_loader(self, dataset_name: str):
        # ... (this method remains unchanged) ...
        dataset_class = self.datasets.get(dataset_name.lower())
        if not dataset_class:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
        return dataset_class()
