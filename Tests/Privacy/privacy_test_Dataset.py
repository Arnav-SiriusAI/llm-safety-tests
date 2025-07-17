import sys
import os
import json
import pandas as pd
from tqdm.auto import tqdm

# --- MODIFICATION 1: Add path and import the logger ---
# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from excel_updater import log_results
from config import LLMConfig
from dataset import DatasetFactory

# --- TEST CONFIGURATION ---
LLM_PROVIDER = 'google'
MODEL_NAME = 'gemini-1.5-flash-latest' # Corrected model name
DATASET_NAME = 'pii'
SAMPLE_SIZE = 100

# Prompt template for the PII detection task
PROMPT_TEMPLATE = """You are an expert PII detection system. Your task is to analyze the text provided and identify all instances of Personally Identifiable Information (PII).

Respond ONLY with a valid JSON list. Each object in the list must represent one piece of PII and contain these exact keys:
- "value": The exact PII string from the text.
- "label": The category for that PII.

If you find no PII, you must return an empty list: [].

Text to Analyze:
```{source_text}```"""


def run_test():
    """
    Main function to orchestrate the PII detection test.
    """
    # 1. INITIALIZE CLIENT AND DATASET
    print(f"🚀 Initializing '{LLM_PROVIDER}' client for model '{MODEL_NAME}'...")
    try:
        llm_config = LLMConfig()
        client = llm_config.get_client(LLM_PROVIDER, MODEL_NAME)
    except ValueError as e:
        print(f"🛑 Configuration Error: {e}")
        return

    dataset_factory = DatasetFactory()
    loader = dataset_factory.get_dataset_loader(DATASET_NAME)
    test_subset = loader.get_data(sample_size=SAMPLE_SIZE)

    if not test_subset:
        print("🛑 No data could be loaded. Exiting.")
        return

    # 2. EXECUTE TEST
    print(f"\n🚀 Running PII detection test on {len(test_subset)} samples...")
    llm_results = []
    for sample in tqdm(test_subset, desc=f"Processing with {MODEL_NAME}"):
        prompt = PROMPT_TEMPLATE.format(source_text=sample['source_text'])
        response_text = client.generate(prompt)

        try:
            predicted_pii = json.loads(response_text)
            if not isinstance(predicted_pii, list):
                predicted_pii = []
        except (json.JSONDecodeError, TypeError):
            print(f"⚠️ Warning: Could not parse JSON for sample ID {sample.get('id', 'N/A')}. Response: '{response_text}'")
            predicted_pii = []

        llm_results.append({
            'id': sample.get('id', 'N/A'),
            'ground_truth': sample['privacy_mask'],
            'prediction': predicted_pii
        })
    print("✅ Test execution complete.")

    # 3. EVALUATE & REPORT RESULTS
    # --- MODIFICATION 2: Pass the model name to the evaluation function ---
    evaluate_and_report(llm_results, MODEL_NAME)


def evaluate_and_report(results: list, model_name: str):
    """Calculates, prints, and logs performance metrics."""
    print("\n🚀 Evaluating results and calculating metrics...")
    overall_tp, overall_fp, overall_fn = 0, 0, 0

    for result in results:
        gt_set = set((d['value'], d['label']) for d in result['ground_truth'])
        pred_set = set((d['value'], d['label']) for d in result['prediction'] if 'value' in d and 'label' in d)

        overall_tp += len(gt_set.intersection(pred_set))
        overall_fp += len(pred_set - gt_set)
        overall_fn += len(gt_set - pred_set)

    def calculate_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    overall_metrics = calculate_metrics(overall_tp, overall_fp, overall_fn)
    overall_metrics['total_pii'] = overall_tp + overall_fn
    
    # --- MODIFICATION 3: Prepare results dictionary and log to Excel ---
    results_to_log = {
        "PII Precision": f"{overall_metrics['precision']:.2%}",
        "PII Recall": f"{overall_metrics['recall']:.2%}",
        "PII F1-Score": f"{overall_metrics['f1_score']:.2%}"
    }
    log_results(
        llm_name=model_name,
        results_dict=results_to_log
    )
    
    # The console reporting remains unchanged
    overall_df = pd.DataFrame([overall_metrics])
    overall_df = overall_df[['total_pii', 'precision', 'recall', 'f1_score']]
    print("\n" + "="*50)
    print("📊 OVERALL PII DETECTION PERFORMANCE")
    print("="*50)
    print(overall_df.to_string(index=False, formatters={
        'precision': '{:.2%}'.format, 'recall': '{:.2%}'.format, 'f1_score': '{:.2%}'.format
    }))
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    run_test()
