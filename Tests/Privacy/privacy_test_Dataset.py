
import sys
import os
import time
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import LLMConfig
from dataset import DatasetFactory

# --- Test-Specific Helper Functions ---

def get_attack_config(filename: str) -> dict:
    """Determines the expected PII output based on the filename."""
    # ... (This function's logic is unchanged) ...
    pass

def get_judge_prompt(expected_output: str, generated_text: str) -> str:
    """Creates a tailored prompt for the judge LLM."""
    # ... (This function's logic is unchanged) ...
    pass

def format_prompt(test_case: dict) -> str:
    """Correctly formats a prompt from a test case dictionary."""
    # ... (This function's logic is unchanged) ...
    pass


# --- Main Test Orchestration ---

def run_privacy_test(test_config, judge_config, max_prompts_per_file):
    """Main function to execute the privacy test."""
    print(f"\n🔬 Starting Privacy Test for Target: {test_config['model']}")

    # 1. Initialize Clients
    try:
        llm_config = LLMConfig()
        test_client = llm_config.get_client(test_config['provider'], test_config['model'])
        judge_client = llm_config.get_client(judge_config['provider'], judge_config['model'])
        print(f"✅ Test LLM: {test_config['model']} | Judge LLM: {judge_config['model']}")
    except Exception as e:
        print(f"❌ Failed to initialize LLMs: {e}")
        return

    # 2. Initialize Dataset Loader
    try:
        dataset_factory = DatasetFactory()
        privacy_loader = dataset_factory.get_dataset_loader("decodingtrust_privacy")
        attack_files = privacy_loader.get_attack_files()
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ {e}")
        return

    # 3. Run evaluation loop for each attack file
    results = {}
    for attack_file in attack_files:
        config = get_attack_config(attack_file)
        expected_pii = config["expected_output"]
        print(f"\n--- Testing Attack File: {attack_file} (Expecting: {expected_pii.upper()}) ---")

        test_cases = privacy_loader.load_test_cases_from_file(attack_file)
        if not test_cases:
            print("  ⚠️ No test cases loaded. Skipping file.")
            continue

        sample_cases = test_cases[:max_prompts_per_file]
        leaks_found = 0

        for test_case in tqdm(sample_cases, desc=f"  Processing"):
            try:
                full_prompt = format_prompt(test_case)
                if not full_prompt: continue

                # Get response from test model with specific configs
                response_text = test_client.generate(
                    full_prompt,
                    temperature=0.7,
                    max_tokens=100,
                    safety_settings=test_config.get("safety_settings") # Pass safety settings
                )

                # Get judgment
                judge_prompt = get_judge_prompt(expected_pii, response_text)
                judge_response = judge_client.generate(judge_prompt)
                if "yes" in judge_response.strip().lower():
                    leaks_found += 1
                
                time.sleep(1) # Be respectful to APIs

            except Exception as e:
                print(f"\nAn API or other error occurred: {e}. Sleeping and continuing.")
                time.sleep(5)
                continue
        
        leakage_rate = (leaks_found / len(sample_cases)) * 100 if sample_cases else 0
        results[attack_file] = {"rate": leakage_rate, "leaks": leaks_found, "total": len(sample_cases), "type": expected_pii.upper()}
        print(f"  ✅ Finished. Leaks: {leaks_found}/{len(sample_cases)} ({leakage_rate:.2f}%)")

    # 4. Display Final Report
    print("\n" + "="*70 + "\n--- 📊 Privacy Test Final Report ---\n" + "="*70)
    if not results:
        print("No tests were successfully run.")
        return
        
    report_df = pd.DataFrame.from_dict(results, orient='index')
    report_df['Leakage'] = report_df.apply(lambda row: f"{row['rate']:.2f}% ({row['leaks']}/{row['total']})", axis=1)
    print(report_df[['type', 'Leakage']].rename(columns={'type': 'Expected PII'}).to_markdown())
    print("\n" + "="*70)


if __name__ == "__main__":
    # --- CONFIGURE YOUR TEST RUN HERE ---
    
    # Define models for test and judge roles
    TEST_CONFIG = {
        'provider': 'google',
        'model': 'gemini-1.5-flash-latest',
        'safety_settings': [
            {"category": c, "threshold": "BLOCK_NONE"} for c in 
            ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
             "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
    }
    JUDGE_CONFIG = {'provider': 'openai', 'model': 'gpt-4o'}
    
    # Set max number of prompts to test from each file
    MAX_PROMPTS_PER_FILE = 20

    run_privacy_test(TEST_CONFIG, JUDGE_CONFIG, MAX_PROMPTS_PER_FILE)
