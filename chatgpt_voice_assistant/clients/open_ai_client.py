import json
import logging
from typing import List

import openai
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam


from chatgpt_voice_assistant.exceptions.text_generation_error import TextGenerationError
from chatgpt_voice_assistant.models.message import Message, MessageRole


class OpenAIClient:
    """Client to interact with the OpenAI API."""

    def __init__(self, api_key: str) -> None:
        self._client = openai.OpenAI(api_key=api_key)

    def get_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str = "gpt-4o-2024-05-13",
        max_tokens: int = 500,
        temperature: float = 0.2,
    ) -> Message:
        """
        Returns a completion (response) from the specified OpenAI GPT model.
        Reference: https://platform.openai.com/docs/api-reference/chat

        Args:
            messages: Messages to provide as context.
            prompt: The prompt to send to the model for completion.
            model: Optionally specify the model to use.
            max_tokens: The max number of tokens to use including the prompt and response.
            temperature: The temperature for the model to use.

        Returns:
            The response from the model.
        """
        if len(messages) == 0:
            raise ValueError(
                "The messages argument must be a list with at least one object."
            )

        logging.debug(
            f"Sending prompt to chat completion endpoint: {json.dumps(messages)}"
        )

        completion: ChatCompletion = self._client.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choices: List[Choice] = completion.choices
        tokens_used: int = completion.usage.total_tokens if completion.usage else 0
        was_cut_short = False

        if len(choices) == 0:
            raise TextGenerationError(
                "No choices returned from Open AI for provided context"
            )

        first_choice: Choice = choices[0]

        if first_choice.finish_reason == "length":
            logging.warning(
                "OpenAI stopped generating due to a limit on the max tokens. "
                f"Max tokens is set to: {max_tokens}. "
                f"Total tokens consumed with prompt were: {tokens_used}"
            )
            was_cut_short = True

        first_choice_text: str = first_choice.message.content or ""

        return Message(
            first_choice_text,
            MessageRole.ASSISTANT
            if first_choice.message.role == "assistant"
            else MessageRole.USER,
            was_cut_short,
        )
