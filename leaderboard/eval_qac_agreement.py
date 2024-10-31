"""Given QAC and answers generated from multiple models,
evaluate agreement with gold answer and create a leaderboard.
"""

import argparse
import csv
import json
import numpy as np
import os
import pandas as pd
from tabulate import tabulate
import time
from tqdm import tqdm
from typing import Tuple

from src.utils.prompt_utils import QAC_ANSWER_AGREEMENT_PROMPT
import src.utils.llm as llm_lib


class QACEvaluator:
    """Class for QAC evaluation."""

    def __init__(self, model="gpt-4o-2024-05-13"):
        self.claim_type_to_prompt = {
            "QAC_Eval_Answer_Agreement": QAC_ANSWER_AGREEMENT_PROMPT,
        }
        self.generator_llm = llm_lib.LLM(model)

    def eval_qac(
        self,
        question: str,
        context: str,
        golden_answer: str,
        answer: str,
        eval_type: str = "QAC_Eval_Answer_Agreement",
        max_retries: int = 3,
    ) -> Tuple[int, str]:

        prompt = self.claim_type_to_prompt[eval_type]
        prompt = prompt.replace("[QUESTION]", question)
        prompt = prompt.replace("[CONTEXT]", context)
        prompt = prompt.replace("[GOLDEN_ANSWER]", golden_answer)
        prompt = prompt.replace("[ACTUAL_ANSWER]", answer)

        retry, counter = True, 0
        while retry:
            try:
                response = self.generator_llm(message=prompt, with_json=True)
                response = json.loads(response, strict=False)
                retry = False

                if "agreement" not in response.keys():
                    response = {"agreement": "0", "reason": ""}

            except Exception as e:
                counter += 1
                if counter > max_retries:
                    response = {"agreement": "0", "reason": ""}
                    retry = False
                else:
                    time.sleep(10)

        return int(response["agreement"]), response["reason"]

    def eval_golden_answer_agreement(
        self,
        question: str,
        context: str,
        golden_answer: str,
        candidate_answer: str,
    ):
        agreement, agreement_reason = self.eval_qac(
            question,
            context,
            golden_answer,
            candidate_answer,
            eval_type="QAC_Eval_Answer_Agreement",
        )

        return int(agreement), agreement_reason


def print_leaderboard(results_dict, metric):
    sorted_keys = sorted(
        results_dict, key=lambda k: results_dict[k][metric], reverse=True
    )
    print(
        tabulate(
            [[key, "{:.2f}".format(results_dict[key][metric])] for key in sorted_keys],
            headers=["Model", "Agreement Rate"],
        )
    )


def read_alpacaeval_json(answers_path):
    # Loading the json list with all model answers into a dataframe
    with open(answers_path) as f:
        list_answers = json.load(f)

    generator_models = {item["generator"] for item in list_answers}
    model_count = len(generator_models)
    qac_count = int(len(list_answers) / model_count)

    generated_answers_df = [
        list_answers[qac_count * i : qac_count * (i + 1)] for i in range(model_count)
    ]
    generated_answers_df = pd.DataFrame(generated_answers_df).transpose()
    generated_answers_df.columns = list(generator_models)
    generated_answers_df.reset_index(inplace=True, drop=True)

    return generated_answers_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        required=True,
        type=str,
        help="Path pointing to seed QAC with gold answers",
    )

    parser.add_argument(
        "--answers_path",
        required=True,
        type=str,
        help=(
            "Path pointing to the json file with LLM answers "
            "(in alpacaeval format). You can generate this using generate_qac.py"
        ),
    )

    parser.add_argument(
        "--output_dir", type=str, default="./output/", help="Path to store results"
    )

    args = parser.parse_args()

    qac_data = pd.read_csv(args.source_path)
    qac_data.replace("", np.nan, inplace=True)
    qac_data.dropna(inplace=True)
    qac_data.reset_index(inplace=True, drop=True)

    generated_answers_df = read_alpacaeval_json(args.answers_path)
    # Creating output directory to save the results
    output_dir = os.path.join(f"{args.output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Evaluation pipeline
    pipeline = QACEvaluator()

    with open(f"{args.output_dir}/agreement_results.csv", "w") as output_file:
        csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)

        models_metric_dict = {}
        for key in generated_answers_df.columns:
            models_metric_dict[key] = {"agreement": []}

        for key in generated_answers_df.columns:
            for index, qac in tqdm(qac_data.iterrows(), total=qac_data.shape[0]):
                score, reason = pipeline.eval_golden_answer_agreement(
                    qac.Question,
                    qac.Context,
                    qac.Answer,
                    generated_answers_df[key][index]["output"],
                )
                models_metric_dict[key]["agreement"].append(score)
                csv_writer.writerow([str(index), key, str(score)])

    for key in generated_answers_df.columns:
        total_agreeing_samples = models_metric_dict[key]["agreement"].count(1)
        total_sample_count = len(models_metric_dict[key]["agreement"])
        models_metric_dict[key]["agreement_rate"] = (
            total_agreeing_samples / total_sample_count
        )

    print_leaderboard(models_metric_dict, metric="agreement_rate")
