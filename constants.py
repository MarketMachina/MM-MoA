import os
import textwrap
from dotenv import load_dotenv

load_dotenv()


class Prompts:
    def moa_system(self) -> str:
        return textwrap.dedent(
            """
        You are a helpful AI assistant.
        """
        )

    def moa_intermediate(self) -> str:
        return textwrap.dedent(
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
        )

    # - Avoids redundancy and unnecessary details.

    def moa_final(self, user_prompt: str) -> str:
        return textwrap.dedent(
            f"""
        Please, provide a final, authoritative answer to the following question:

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
        )

    # - Avoid unnecessary elaboration or step-by-step problem-solving.


class Config:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
