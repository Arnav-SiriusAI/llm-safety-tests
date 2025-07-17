🔒 LLM Safety Tests
This repository provides a comprehensive test suite to evaluate Large Language Models (LLMs) across 7 critical dimensions of safety. Each test is grounded in leading academic benchmarks and datasets, allowing systematic, reproducible, and scalable LLM evaluations.

🧪 Included Safety Tests
Content Harms
Evaluates generation of explicitly harmful content including toxicity, violence, and inappropriate material.
Dataset used: ToxiGen

Discrimination and Bias
Measures the model's bias across demographics, such as race, gender, religion, etc.
Dataset used: BBQ (Bias Benchmark for QA)

Misinformation and Disinformation
Tests for hallucination and the generation of false or misleading facts.
Dataset used: BEHONEST

Malicious Use Enablement
Assesses how easily an LLM assists in harmful, illegal, or dangerous activities.
Dataset used: HarmBench

Privacy Violations
Detects leakage of personally identifiable information (PII) or other sensitive content.
Dataset used: DecodingTrust

Interaction and Psychological Harms
Captures subtle harms in human-LLM interaction such as manipulation, over-reliance, and deception.
Dataset used: BEHONEST (Sycophancy)

Robustness and Reliability
Evaluates the model’s stability under adversarial prompts and resistance to manipulation.
Dataset used: AdvBench

⚙️ Configuration & Customization
🔑 API Keys:
Add your model providers' API keys (e.g., OpenAI, Anthropic, Google) in the .env file. 

🧠 Add Custom LLMs:
Add or configure different LLMs in the config.yaml file. You can plug in models from OpenAI, Anthropic, HuggingFace, Google, etc.

📂 Datasets Loader:
All datasets used in these tests are modularized and managed via the datasets.py file. You can easily add new datasets or swap existing ones.

🚀 Getting Started
Clone the repo:

bash
Copy
Edit
git clone https://github.com/Arnav-SiriusAI/llm-safety-tests.git
cd llm-safety-tests
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set your .env with required API keys.

Run any test file to begin evaluation.

📌 Goals
This project aims to:

Promote transparency and safety in LLM deployment.

Provide reproducible, benchmark-based evaluations.

Support research on trustworthiness, robustness, and ethical AI.
