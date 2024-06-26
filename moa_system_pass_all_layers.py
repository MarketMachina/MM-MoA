import base64
from typing import List, Dict, Union
import requests
import json
import os
import logging
import textwrap


class MoASystem:
    def __init__(self):
        self.layers = [
            [self.claude_3_5_sonnet, self.gpt_4o],
            [self.claude_3_5_sonnet, self.gpt_4o],
            [self.claude_3_5_sonnet, self.gpt_4o],
            [self.claude_3_5_sonnet],  # Aggregation layer
        ]
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.anthropic_api_key or not self.openai_api_key:
            self.logger.warning(
                "Please set ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables."
            )

    def claude_3_5_sonnet(
        self, system_prompt: str, messages: List[Dict[str, str]]
    ) -> str:
        headers = {
            "content-type": "application/json",
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
        }

        data = {
            "model": "claude-3-5-sonnet-20240620",
            "system": system_prompt,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.2,
        }

        self.logger.debug("Request data:")
        self.logger.debug(json.dumps(data, indent=2))

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            )
            response.raise_for_status()
            response_data = response.json()

            self.logger.debug("Response data:")
            self.logger.debug(json.dumps(response_data, indent=2))

            if "content" in response_data and response_data["content"]:
                return response_data["content"][0]["text"]
            else:
                self.logger.error(f"Unexpected response structure: {response_data}")
                return "Error: Unexpected response structure from Claude API"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in Claude API сall: {e}")
            if hasattr(e, "response") and e.response is not None:
                self.logger.error(f"Response content: {e.response.text}")
            return f"Error: {e}"

    def gpt_4o(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        data = {
            "model": "gpt-4o",
            "messages": full_messages,
            "max_tokens": 1000,
            "temperature": 0.7,
        }

        self.logger.debug("Request data:")
        self.logger.debug(json.dumps(data, indent=2))

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            )
            response.raise_for_status()
            response_data = response.json()

            self.logger.debug("Response data:")
            self.logger.debug(json.dumps(response_data, indent=2))

            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            else:
                self.logger.error(f"Unexpected response structure: {response_data}")
                return "Error: Unexpected response structure from OpenAI API"
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error in OpenAI API сall: {e}")
            return f"Error: {e}"

    def process_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"Error processing image: {e}")
            return f"Error: {e}"

    def run(self, system_prompt: str, user_prompt: Union[str, Dict]) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    user_prompt
                    if isinstance(user_prompt, str)
                    else json.dumps(user_prompt)
                ),
            }
        ]

        for layer_index, layer in enumerate(self.layers):
            self.logger.info(f"Processing layer {layer_index + 1}")

            if layer_index < len(self.layers) - 1:  # for all layers except the last one
                layer_responses = []
                for model_index, model in enumerate(layer, start=1):
                    response = model(system_prompt, messages)
                    if not response.startswith("Error:"):
                        layer_responses.append(f"Answer{model_index}: {response}")
                    else:
                        self.logger.warning(
                            f"Skipping error response in layer {layer_index + 1}: {response}"
                        )

                # Combine all responses from this layer into one message
                if layer_responses:
                    combined_response = "\n\n".join(layer_responses)
                    messages.append({"role": "assistant", "content": combined_response})

                if (
                    layer_index != len(self.layers) - 2
                ):  # for all layers except the second-to-last one
                    messages.append(
                        {
                            "role": "user",
                            "content": textwrap.dedent(
                                """
                            Please review and analyze the previous answers. Your task is to:

                            1. Synthesize the information from all previous answers.
                            2. Identify and correct any errors or inconsistencies.
                            3. Provide additional insights or context that may have been overlooked.
                            4. Clarify any complex concepts or terminology.
                            5. If applicable, offer real-world applications or implications of the solution.

                            Based on this analysis, write a consolidated answer that:

                            - Is clear, concise, and accurate.
                            - Incorporates the best elements from each previous answer.
                            - Provides a comprehensive understanding of the problem and its solution.
                            - Uses a logical structure that's easy to follow.

                            Your consolidated answer should stand alone as a complete response to the original question, without needing to reference the previous answers directly.
                            """
                                # - Avoids redundancy and unnecessary details.
                            ),
                        }
                    )

            else:  # for the last (aggregation) layer
                final_prompt = textwrap.dedent(
                    f"""
                You are an AI assistant tasked with providing a final, authoritative answer to the following question:

                {user_prompt["text"]}

                Your task:
                1. Synthesize all the information from previous analyses without explicitly mentioning them.
                2. Provide a clear, concise, and direct answer to the question.
                3. Include the essential reasoning or formula used to arrive at the answer.
                4. If applicable, offer a brief explanation of the significance or real-world application of this result.
                5. Ensure your response is self-contained and can be understood without any additional context.

                Remember:
                - Be precise and factual.
                - Do not use phrases like "Based on previous responses" or "After careful consideration".
                - Your answer should appear as if it's the first and only response to the question.

                Provide your response now:
                """
                    #  - Avoid unnecessary elaboration or step-by-step problem-solving.
                ).strip()
                messages.append({"role": "user", "content": final_prompt})

                final_response = layer[0](system_prompt, messages)
                return final_response

        self.logger.error("No valid response generated")
        return "Error: No valid response generated"


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    moa = MoASystem()

    system_prompt = "You are a helpful AI assistant."
    user_prompt = {
        "text": "At the event, there were 66 handshakes. If everyone shook hands with each other, how many people were at the event in total?",
        # "image": moa.process_image("path/to/your/image.jpg")
    }

    final_response = moa.run(system_prompt, user_prompt)
    print("Final response:")
    print(final_response)
