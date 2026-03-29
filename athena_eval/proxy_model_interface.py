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
from typing import Optional, Tuple
import requests
import urllib3
from dotenv import load_dotenv

# Add src to Python path for synthetics imports
import os
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
bus_model = os.environ.get("BUS_MODEL")

logger = logging.getLogger(__name__)
load_dotenv()

class OpenAIProxyModelInterface():
    """
    Model interface for OpenAI proxy endpoints with Azure authentication.
    
    Supports bus:snap model paths with automatic renderer integration
    and simplified authentication (local datasets only).
    """
    
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 500,
        temperature: float = 0.1,
        timeout: int = 300
    ):
        """
        Initialize OpenAI proxy model interface.
        
        Args:
            model_path: Complete bus:snap model path (e.g., "bus:snap:orngcresco/models/user/model/policy:user:msft$renderer")
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            endpoint_url: OpenAI proxy endpoint URL
            resource_id: Azure resource ID for authentication
            service_url: Internal service URL header
            timeout: Request timeout in seconds
        """
        self._auth_token = os.environ.get("AUTH_TOKEN_OT")
        self.model_path = os.environ.get("BUS_MODEL")
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.timeout = timeout
        self._auth_token = os.environ.get("AUTH_TOKEN_OT")
        self._inference_endpoint = os.environ.get("INSPECTAI_OT_INFERENCE_ENDPOINT")
        logger.info(f"Initialized OpenAI proxy interface for model: {self.model_path}")
        
    def _validate_model_path(self, model_path: str) -> str:
        """Validate that model path is in the complete bus:snap format."""
        if not model_path.startswith("bus:snap:"):
            raise ValueError(
                f"Model path must be in complete bus:snap format. "
                f"Got: {model_path}. "
                f"Expected format: 'bus:snap:orngcresco/models/user/model/policy:user:msft$renderer'"
            )
        
        if ":user:" not in model_path or "$" not in model_path:
            raise ValueError(
                f"Model path must include both user and renderer components. "
                f"Got: {model_path}. "
                f"Expected format: 'bus:snap:orngcresco/models/user/model/policy:user:msft$renderer'"
            )
        
        return model_path
    
    def _get_azure_access_token(self) -> str:
        """Get Azure access token using Azure CLI."""
        '''
        try:
            cmd = [
                "az", "account", "get-access-token",
                "--resource", self.resource_id,
                "--query", "accessToken",
                "-o", "tsv"
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                check=True
            )
            
            token = result.stdout.strip()
            if not token:
                raise ValueError("Empty token received from Azure CLI")
            
            return token
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Azure CLI authentication failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Failed to authenticate with Azure CLI: {e}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Azure CLI authentication timed out after 30 seconds")
    '''
        return os.getenv("AUTH_TOKEN_OT")
    
    def generate_response_for_prompt(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate response for a prompt-based sample.
        
        Args:
            sample: PromptEvaluationSample object
            **kwargs: Additional generation parameters (unused)
        
        Returns:
            Tuple of (model_output, thought_output, final_output)
            
        Raises:
            RuntimeError: If API request fails or authentication issues
        """
        # Extract prompt from the sample object
        
        
        # Use the internal method with the extracted prompt
        return self._generate_response_for_prompt_string(prompt)
    
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
    
    def _generate_response_for_prompt_string(self, prompt: str) -> str:
        """
        Internal method to generate response for a prompt string.
        
        Args:
            prompt: Input prompt text as string
        
        Returns:
            Tuple of (model_output, thought_output, final_output)
        """
        while True:
            try:
                # Get access token
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "Cookie": f"_oauth2_proxy={self._auth_token}",
                }
                
                # Prepare payload
                payload = {
                    "model": self.model_path,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": 0.95,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "parallel_tool_calls": False,
                }
                inference_endpoint = f"https://openai-proxy.int.prod-southcentralus-hpe-2.dev.openai.org/v1/chat/completions"
                # Make request
                response = requests.post(
                    inference_endpoint,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=self.timeout
                )
                
                if response.status_code != 200:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    raise RuntimeError(f"OpenAI proxy request failed: {response.status_code}")
                
                # Parse response
                response_data = response.json()
                
                if 'choices' not in response_data or not response_data['choices']:
                    logger.error(f"Invalid response format: {response_data}")
                    raise RuntimeError("Invalid response format from OpenAI proxy")
                
                content = response_data['choices'][0]['message']['content']
                
                # For this simplified version, we don't process thinking tags
                # The evaluation framework will handle that if needed
                return content
                
            except Exception as e:
                print(f"Error generating response for prompt: {e}, brief pause 60s before retrying")
                time.sleep(60)  # brief pause before retrying
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"OpenAIProxyModelInterface(model_path='{self.model_path}', endpoint='{self.endpoint_url}')"
