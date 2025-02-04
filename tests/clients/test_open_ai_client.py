import pytest
from unittest import mock

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.completion_usage import CompletionUsage

from chatgpt_voice_assistant.clients.open_ai_client import OpenAIClient
from chatgpt_voice_assistant.exceptions.text_generation_error import TextGenerationError
from chatgpt_voice_assistant.models.message import Message

# Your mocked completion functions
MOCK_RESPONSE_STOP_DUE_TO_LENGTH = Choice(
    message=ChatCompletionMessage(
        content="Hello there! How are you?", role="assistant"
    ),
    index=0,
    finish_reason="length",
)

MOCK_RESPONSES = [
    Choice(
        message=ChatCompletionMessage(
            content="Hello there! How are you?", role="assistant"
        ),
        index=0,
        finish_reason="stop",
    ),
    Choice(
        message=ChatCompletionMessage(
            content="I'm doing well! How about you?", role="assistant"
        ),
        index=1,
        finish_reason="stop",
    ),
]


def mock_create_completion_no_responses(*args, **kwargs):
    return ChatCompletion(
        choices=[],
        usage=CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=kwargs["max_tokens"]
        ),
        model="gpt-3.5-turbo",
        created=0,
        id="chatcmpl-12345",
        object="chat.completion",
    )


def mock_create_completion_multiple_responses(*args, **kwargs):
    return ChatCompletion(
        choices=MOCK_RESPONSES,
        usage=CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=kwargs["max_tokens"]
        ),
        model="gpt-3.5-turbo",
        created=0,
        id="chatcmpl-12345",
        object="chat.completion",
    )


def mock_create_completion_stop_due_to_length(*args, **kwargs):
    return ChatCompletion(
        choices=[MOCK_RESPONSE_STOP_DUE_TO_LENGTH],
        usage=CompletionUsage(
            completion_tokens=0, prompt_tokens=0, total_tokens=kwargs["max_tokens"]
        ),
        model="gpt-3.5-turbo",
        created=0,
        id="chatcmpl-12345",
        object="chat.completion",
    )


# Fixture that patches out the OpenAI constructor and injects our nested mocks.
@pytest.fixture
def open_ai_client():
    with mock.patch("openai.OpenAI.__init__", return_value=None):
        client = OpenAIClient("fake-key")
        # Replace _client with a MagicMock so no real auth occurs.
        client._client = mock.MagicMock()
        # Set up the nested structure so our code can call:
        #   self._client.chat.completions.create(...)
        client._client.chat = mock.MagicMock()
        client._client.chat.completions = mock.MagicMock()
        yield client


def test_get_chat_completion_throws_exception_no_responses(open_ai_client):
    open_ai_client._client.chat.completions.create.side_effect = (
        mock_create_completion_no_responses
    )

    message = {"role": "user", "content": "Yeah do you have one in mind?"}

    with pytest.raises(TextGenerationError):
        open_ai_client.get_chat_completion([message], max_tokens=70)


def test_get_chat_completion_returns_first_response(open_ai_client):
    open_ai_client._client.chat.completions.create.side_effect = (
        mock_create_completion_multiple_responses
    )

    message = {"role": "user", "content": "Yeah do you have one in mind?"}
    response = open_ai_client.get_chat_completion([message], max_tokens=70)

    assert response.content == MOCK_RESPONSES[0].message.content
    assert not response.was_cut_short


def test_get_chat_completion_sets_was_cut_short_to_true(open_ai_client):
    open_ai_client._client.chat.completions.create.side_effect = (
        mock_create_completion_stop_due_to_length
    )

    message = {"role": "user", "content": "Yeah do you have one in mind?"}
    response: Message = open_ai_client.get_chat_completion([message], max_tokens=70)

    assert response.content == MOCK_RESPONSE_STOP_DUE_TO_LENGTH.message.content
    assert response.was_cut_short
