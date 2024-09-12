"""Define the LLMs to be used."""

# Built-in imports.
from typing_extensions import Annotated
import json

# External imports.
from litellm import completion
import time
import openai


class LLM:
  def __init__(self, model_name):
    self.model_name = model_name

  def __call__(self, message, with_json=False, max_failed_attempts=3, sleeping_time=10):
    num_failed_attempts = 0
    while True:
      if num_failed_attempts >= max_failed_attempts:
        assert 1 == 0, "Model failed to process query."
      try:
        response = completion(model=self.model_name, messages=[{"content": message, "role": "user"}],).choices[0].message.content
        break
      except openai.AuthenticationError as e:
        print(e)
        break
      except Exception as e:
        # sleep because maybe we are requesting too fast
        time.sleep(sleeping_time)
        num_failed_attempts += 1

    if with_json:
      num_failed_attempts = 0
      kwargs = dict()
      kwargs["response_format"] = { "type": "json_object"}
      while True:
        if num_failed_attempts >= 3:
          potential_response = '{"question": "", "answer": "", "context": ""}'
          break
        json_message = f"You are a model that just returns valid json outputs. Rewrite the following as a proper JSON: {response}\nReturn only JSON and nothing else. Avoid phrases like 'sure here is your json' etc. Just return the JSON. If you can't parse this as a JSON, return an empty json. "
        potential_response = completion(model=self.model_name, messages=[{"content": json_message, "role": "user"}], **kwargs).choices[0].message.content
        try:
          json.loads(potential_response)
          break
        except openai.AuthenticationError as e:
          print(e)
          break
        except Exception as e:
          # sleep because maybe we are requesting too fast
          time.sleep(sleeping_time)
          num_failed_attempts += 1
          continue
      response = potential_response
    return response
