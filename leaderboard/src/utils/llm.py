"""Define the LLMs to be used."""

# Built-in imports.
import json
import time
from typing_extensions import Annotated

# External imports.
from litellm import completion
import openai


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(
        self, message, with_json=False, max_failed_attempts=3, sleeping_time=10
    ):
        num_failed_attempts = 0
        retry = True
        while retry:
            if num_failed_attempts >= max_failed_attempts:
                print("Exception: ", e)
                print("Model failed to process query. Max retries reached.")
                return None
            try:
                response = (
                    completion(
                        model=self.model_name,
                        messages=[{"content": message, "role": "user"}],
                    )
                    .choices[0]
                    .message.content
                )
                retry = False
            except openai.AuthenticationError as e:
                print("Exception: ", e)
                return None
            except Exception as e:
                # sleep because maybe we are requesting too fast
                print("Exception: ", e)
                time.sleep(sleeping_time)
                num_failed_attempts += 1

        retry = True
        if with_json:
            num_failed_attempts = 0
            kwargs = dict()
            kwargs["response_format"] = {"type": "json_object"}
            while retry:
                if num_failed_attempts >= max_failed_attempts:
                    print("Exception: ", e)
                    potential_response = '{"question": "", "answer": "", "context": ""}'
                    retry = False
                json_message = f"You are a model that just returns valid json outputs. Rewrite the following as a proper JSON: {response}\nReturn only JSON and nothing else. Avoid phrases like 'sure here is your json' etc. Just return the JSON. If you can't parse this as a JSON, return an empty json. "
                potential_response = (
                    completion(
                        model=self.model_name,
                        messages=[{"content": json_message, "role": "user"}],
                        **kwargs,
                    )
                    .choices[0]
                    .message.content
                )
                # Often the LLM will return ```json for json output, which is not json loadable. Here we first remove markdown before trying to load as json.
                potential_response = instruction_example.remove_markdown(
                    potential_response
                )
                try:
                    json.loads(potential_response)
                    retry = False
                except openai.AuthenticationError as e:
                    print("Exception: ", e)
                    print("Model failed to process query. Max retries reached.")
                    return None
                except Exception as e:
                    # sleep because maybe we are requesting too fast
                    print("Exception: ", e)
                    time.sleep(sleeping_time)
                    num_failed_attempts += 1
                    continue
            response = potential_response
        return response
