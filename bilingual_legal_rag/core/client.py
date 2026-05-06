import openai  # type: ignore[import]
from abc import ABC, abstractmethod
from linuxcompanion.core.schemas import ModelResponse
import itertools


class ClientInterface(ABC):
    @abstractmethod
    def send(self, messages, temp=0.6, tools=None) -> ModelResponse:
        pass


class OpenAIClient(ClientInterface):
    def __init__(self, port=8000):
        self.client = openai.OpenAI(
            base_url=f"http://localhost:{port}/v1", 
            api_key="sk-no-key-required" 
        )

    @staticmethod
    def _handle_tool_chunk(chunk, full_tool_calls):
        delta = chunk.choices[0].delta
        if delta.tool_calls: # this is a list of tool calls in the chunk
            for tc_delta in delta.tool_calls: # each tool call
                idx = tc_delta.index

                if len(full_tool_calls) <= idx: # new tool call
                    full_tool_calls.append({
                        "id": tc_delta.id,
                        "type": "function",
                        "function": {
                            "name": tc_delta.function.name, "arguments": ""
                        }
                    })

                # Append argument fragments
                if tc_delta.function.arguments:
                    full_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

    
    @staticmethod
    def _generate_content_stream(stream):
        return (
            chunk.choices[0].delta.content 
            for chunk in stream 
            if chunk.choices[0].delta.content
            )


    def send(self, messages, temp=0.6, tools=None) -> ModelResponse:
        try:
            stream = self.client.chat.completions.create(
                model="local-model",
                messages=messages,
                temperature=temp,
                stream=True,
                **({"tools": tools} if tools else {})
            )

            first_chunk = None
            for chunk in stream:
                if chunk.choices[0].delta.tool_calls or chunk.choices[0].delta.content:
                    first_chunk = chunk
                    break

            full_stream = itertools.chain([first_chunk], stream)

            if first_chunk.choices[0].delta.tool_calls: # this is a tool call response
                full_tool_calls = []
                for chunk in full_stream:
                    OpenAIClient._handle_tool_chunk(chunk, full_tool_calls)
                    
                return ModelResponse(tool_calls=full_tool_calls)

            elif first_chunk.choices[0].delta.content:
                return ModelResponse(content_stream=OpenAIClient._generate_content_stream(full_stream))

            else:
                raise RuntimeError("Unexpected empty response from model")
            

        except openai.APIConnectionError as e:
            raise ConnectionError("Llama server is not running") from e

