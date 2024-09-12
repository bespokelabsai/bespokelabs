from string import Template
import random


QAC_ANSWER_AGREEMENT_PROMPT = """
You are tasked with evaluating a Retrieval-Augmented Generation (RAG) system. Given a context, query, and a "GOLDEN" answer, your job is to evaluate the actual answer . The actual answer should capture all key information in the *GOLDEN* answer.

Please:

1. Determine whether the answer contains all the information from the context required to thoroughly answer the question.
2. Check if the actual answer lacks any relevant or valuable information that is present in the GOLDEN answer.
3. Assess how closely the actual answer aligns with the *GOLDEN* answer in responding to the question. 
4. If the actual answer lacks some details found in the *GOLDEN* answer but still completely addresses the question, give full credit.

Question: <[QUESTION]>
Context: <[CONTEXT]>
GOLDEN Answer: <[GOLDEN_ANSWER]>
Actual Answer: <[ACTUAL_ANSWER]>

Your answer has to be one of the following three choices:
1. If the actual answer fully covers the NECESSARY information in the *GOLDEN* answer for responding to the question, return {'agreement': '1', 'reason':'<your reasoning>'}
2. If the actual answer misses some of the information in the *GOLDEN* answer for responding to the question, return {'agreement': '-1', 'reason':'<your reasoning>'}
3. If it is unclear whether the actual answer covers the information in the *GOLDEN* answer for responding to the question, return {'agreement': '0', 'reason':'<your reasoning>'}
"""

REQUIREMENTS = {
    "complex_question": "Do not generate simple lookup questions. The questions should require some reasoning to answer and should reflect understanding of the underlying context.",
    "avoid_phrases": "Avoid phrases like 'according to the context', 'based on the document', 'sure here is what you asked for', etc. Just focus on the required task.",
    "short_outputs": "Your output should be concise and to the point.",
    "reasoning": "The answer should include the reasoning behind the answer.",
    "json_output_with_keys": Template("Your output must be JSON, with keys: $keys_list in this exact order."),
    "json_list_output_with_keys": Template("Your output must be a list of JSONs, where each JSON has keys: $keys_list in this exact order."),
    "just_output": Template("Just output $what_to_output and nothing else.").substitute(what_to_output="what I asked you to output in the format I asked you and nothing else."),
    "many_different_options": Template("Before giving me your final output, create a list of $num_trials potential outputs.").substitute(num_trials=100),
    "interesting_chunk": "The part should be self-contained and should make sense on its own. The part can be anywhere in the document. You are allowed to mix and match sentences from different parts of the document. You can never write something that is not in the document.",
}


INSTRUCTIONS = {
    "rate_answer_given_context": "rate an answer to a question given a context",
    # singular
    "ask_question": "ask an interesting question given a context",
    "answer_question": "answer a question as precisely as possible",
    "answer_question_given_context": "answer a question given a context as precisely as possible",
    "ask_question_and_answer_given_context": "ask an interesting question given the context. Provide a concise answer to the question asked that is supported by the context. Make sure that both the question and the answer are exactly supported in the context.",    
}


def get_prompt(domain: str, instruction: str, current_context: str = None,
               requirements: str = None, examples: str = None, 
               num_distractions: int =1000, max_number: int = 100) -> str:
    expert_prompt = f"You are an expert in {domain}. \n <TASK> Your task is to {instruction}. \n </TASK> \n"
    if requirements:
        expert_prompt += f"\n <REQUIREMENTS>{requirements}\n </REQUIREMENTS>\n"
    if examples:
        expert_prompt += f"<EXAMPLES> Here are some examples:\n{examples}\n</EXAMPLES> \n"
    if current_context is not None:
        expert_prompt += f"""\n Please perform the assigned task for the following input (enclosed between <INPUT> and </INPUT> tags):
        <INPUT>
        {current_context}
        </INPUT> Make sure to follow the requirements."""
    else:
        expert_prompt += "\n Please perform the assigned task and remember the requirements."
    return expert_prompt


