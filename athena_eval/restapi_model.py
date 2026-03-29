"""
OpenAI Proxy Model Interface

This module contains the model interface for connecting to OpenAI proxy endpoints
with Azure authentication and bus:snap model path support.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple
import requests
import urllib3
import tenacity
# Add src to Python path for synthetics imports

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)
class FatalOrangeModelError(Exception):
    pass

def oai_out_extract(
    messages: list[dict[str, Any]]
) -> str:
    def empty_message() -> str:
        return ''

    hidden_messages: list[dict[str, str | bool]] = []
    final_content = None

    for message in messages:
        if message["author"]["role"] != "assistant":
            raise RuntimeError(
                f"Expected assistant role messages only. If this happens often, something is wrong, as the model is claiming that the messages it produces are authored by someone else. Got: {message}"
            )
        final_content = "".join(message["content"]["parts"])
        if message["channel"] == "final":
            if message["content"]["content_type"] != "text":
                raise RuntimeError(
                    f"Expected 'text' content type for final message, got '{message['content_type']}'. If this happens often, something is wrong."
                )
                return empty_message()
            final_content = "".join(message["content"]["parts"])
            return final_content
    if final_content is not None:
        return final_content
    raise RuntimeError("No final message found in response")    
    
class OpenAIModelInterface():
    """
    Model interface for OpenAI proxy endpoints with Azure authentication.
    
    Supports bus:snap model paths with automatic renderer integration
    and simplified authentication (local datasets only).
    """
    
    def __init__(
        self,
        max_tokens: int = 16000,
        temperature: float = 1.0,
        endpoint_url: str = "",
        renderer_name: str = "harmony_v4.0.16_berry_v3_1mil_orion_lpe_no_budget_commentary_cs_cross_msg_msc_tools_v2",
        timeout: int = 900
    ):
        """
        Initialize OpenAI proxy model interface.
        
        Args:
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.endpoint_url = endpoint_url
        self._renderer_name = renderer_name
        self.timeout = timeout
        print(f"[OpenAIModelInterface] Initialized with endpoint {self.endpoint_url}, renderer {self._renderer_name}")
    
    def generate_response_for_fixed_qa(self, sample) -> Tuple[str, Optional[str], str]:
        """
        Generate response for a fixed Q&A sample.
        
        This interface is designed for prompt-based samples, but we implement this
        method to satisfy the BaseModelInterface abstract class requirements.
        """
        # Extract prompt from the sample object
        prompt = None
        
        # Try different attribute names that might contain the input text
        for attr in ['input_text', 'prompt', 'question', 'input', 'text']:
            if hasattr(sample, attr):
                prompt = getattr(sample, attr)
                break
        
        # If no prompt found, try to extract from sample dictionary if it's dict-like
        if prompt is None:
            if hasattr(sample, '__getitem__'):
                for key in ['input_text', 'prompt', 'question', 'input', 'text']:
                    try:
                        prompt = sample[key]
                        break
                    except (KeyError, TypeError):
                        continue
        
        # Final fallback: use a default prompt if still nothing found
        if prompt is None:
            logger.warning(f"Could not extract prompt from sample of type {type(sample)}, using fallback")
            prompt = "Hello, please respond with a test message."
        
        return self._generate_response_for_prompt_string(prompt)
    
    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=5),
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
        retry=tenacity.retry_if_not_exception_type(FatalOrangeModelError),
    )
    def generate_response_for_prompt(self, prompt: str) -> str:
        """
        Internal method to generate response for a prompt string.
        
        Args:
            prompt: Input prompt text as string
        
        Returns:
            Tuple of (model_output, thought_output, final_output)
        """
        try:
            # Get access token
            
            # Prepare headers
            headers = {}
            
            # Prepare payload
            payload = {
                "temperature": self.temperature,
                "renderer": self._renderer_name,
                "max_tokens": self.max_tokens,
                 "messages": [{
                    "role": "user",
                    "content": prompt
                 }],
            }
            #print(f"[generate_response_for_prompt] payload: {payload}")
            # Make request
            response = requests.post(
                self.endpoint_url,
                headers={},
                json=payload,
                verify=False,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                raise RuntimeError(f"OpenAI proxy request failed: {response.status_code}")
            
            # Parse response
            response_data = response.json()
            #print(f"[generate_response_for_prompt] response_data: {response_data}")            
            #content = oai_out_extract(response_data)
            content = response_data['choices'][0]['message']['content']
            if not content:
                raise RuntimeError("No content found in model response")

            # For this simplified version, we don't process thinking tags
            # The evaluation framework will handle that if needed
            return content
            
        except Exception as e:
            logger.error(f"Error generating response for prompt: {e}")
            raise
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"OpenAIProxyModelInterface(model_path='{self.model_path}', endpoint='{self.endpoint_url}')"
    
