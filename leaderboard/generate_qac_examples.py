# Given QAC samples, generate new answers using candidate LLMs,
# to evaluate models for question asnwerin.

import argparse
import copy
import json
import os
import pandas as pd
import textwrap
from tqdm import tqdm

from src.utils import prompt_utils as prompts
import src.utils.instruction_example as instruction_example
import src.utils.llm as llm_lib


parser = argparse.ArgumentParser(description="Eval QAC system.")
parser.add_argument(
    "--seed_data", type=str, required=True, help="Path to seed QAC w gold answers"
)

parser.add_argument(
    "--domain", type=str, required=True, help="Data domain. Example: IT support."
)

parser.add_argument(
    "--reference_model",
    default="gpt-4-1106-preview",
    type=str,
    help="Language model to use as reference for answer generation.",
)

parser.add_argument(
    "--generators",
    type=list,
    help="List of language models to evaluate",
    default=[
        "o1-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "together_ai/google/gemma-2-9b-it",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1",
        "together_ai/databricks/dbrx-instruct",
        "together_ai/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "together_ai/mistralai/Mistral-7B-Instruct-v0.2",
        "together_ai/google/gemma-7b-it",
    ],
)

parser.add_argument(
    "--output_dir", type=str, default="output", help="Output directory."
)

QAC = instruction_example.QAC


def get_examples_from_csv(file_path):
    examples = instruction_example.QACs([])
    data = pd.read_csv(file_path, header=0)
    data.dropna(inplace=True)

    for index, row in data.iterrows():
        sample = {
            "question": row.Question,
            "answer": str(row.Answer),
            "context": [row.Context],
        }
        qac_example = QAC.model_validate(sample)

        examples.qacs.append(qac_example)

    return examples


def write_answers_with_model(seed_data, generator_llm, domain):
    gen_data = copy.deepcopy(seed_data)
    print("Getting answers from generator LLM.")
    instructions = []
    for qac_example in tqdm(gen_data.qacs):
        concatenated_context = "\n".join(qac_example.context)
        current_context = textwrap.dedent(
            f"""
          Question: {qac_example.question}
          Context: {concatenated_context}
      """
        )
        requirements = prompts.REQUIREMENTS["short_outputs"]
        instruction = prompts.get_prompt(
            domain=domain,
            instruction=prompts.INSTRUCTIONS["answer_question_given_context"],
            current_context=current_context,
            requirements=requirements,
        )
        instructions.append(instruction)
        qac_example.answer = generator_llm(instruction)
    return gen_data, instructions


def write_json_output(output_file, gen_data_list, generators, instructions):
    data = []
    for gen_data, generator_name in zip(gen_data_list, generators):
        for example, instruction in zip(gen_data.qacs, instructions):
            data.append(
                {
                    "generator": generator_name,
                    "instruction": instruction,
                    "output": example.answer,
                }
            )

    with open(output_file, "w") as file:
        json.dump(data, file, indent=2)


if __name__ == "__main__":
    args = parser.parse_args()
    seed_data = get_examples_from_csv(args.seed_data)
    gen_data_list = []
    for generator_name in args.generators:
        generator_llm = llm_lib.LLM(generator_name)
        gen_data, instructions = write_answers_with_model(
            seed_data, generator_llm, args.domain
        )
        gen_data_list.append(gen_data)

    ref_llm = llm_lib.LLM(args.reference_model)

    reference_data, instructions = write_answers_with_model(
        seed_data, ref_llm, args.domain
    )

    # Save generated answers in alpaca_eval format
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "all_model_outputs.json")
    write_json_output(output_file, gen_data_list, args.generators, instructions)
    output_file = os.path.join(args.output_dir, "reference_outputs.json")
    write_json_output(
        output_file, [reference_data], [args.reference_model], instructions
    )
