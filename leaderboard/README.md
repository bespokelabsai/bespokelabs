# Evaluation

Evaluate your LLM as a generator for context-based question answering. 

1. `generate_qac.py`: Use sample questions and corresponding contexts to query your LLMs to generate answer. The answer should be supported by the context.

Note: We use LiteLLM SDK to call LLM provider APIs. LiteLLM offers a variety of LLMs from multiple providers. To configure the API key for any LLM provider you wish to use, set the corresponding environment variables.

The source QAC file is expected to be a csv file with three headers: Quesion, Context, and Answer.

`OPENAI_API_KEY=<YOUR_API_KEY> python3 generate_qac.py --seed_data ./qac_dataset.csv --domain 'IT support'`

2. `eval_qac_agreement.py`: Evaluate generated answers on *Agreement with GOLD Answer* with LLM-as-a-judge. Create a leaderboard to rank generator models based on agreement.

The source QAC file is expected to be a csv file with three headers: Quesion, Context, and Answer.
The file containing LLM generated answers is expected to follow AlpacaEval's json format. Each sample is a dictionary with three keys: generator, instruction, and output.

`OPENAI_API_KEY=<YOUR_API_KEY> python3 eval_qac_agreement.py --source_path ./qac_dataset.csv --answers_path ./output/all_model_outputs.json`