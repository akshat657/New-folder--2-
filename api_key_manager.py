import os
from typing import Optional, Any
from groq import RateLimitError
from langchain_groq import ChatGroq
import streamlit as st
import re


class GroqKeyManager:
    """Manages multiple Groq API keys with automatic rotation on rate limits"""

    def __init__(self):
        self.keys = self._load_keys()
        self.current_index = 0

    def _load_keys(self) -> list[str]:
        """Load all available Groq API keys from environment"""
        keys = []

        # Primary key
        primary = os.getenv("GROQ_API_KEY")
        if primary:
            keys.append(primary)

        # Fallback keys
        for i in range(1, 10):  # Support up to GROQ_API_KEY_9
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key:
                keys.append(key)

        if not keys:
            raise ValueError("No Groq API keys found in environment variables")

        return keys

    def get_current_key(self) -> str:
        """Get the current active API key"""
        return self.keys[self.current_index]

    def rotate_key(self):
        """Rotate to the next available key"""
        self.current_index = (self.current_index + 1) % len(self.keys)

    def create_llm(self, model: str = "llama-3.3-70b-versatile",
                   temperature: float = 0.25,
                   max_retries: int = 2,
                   **kwargs) -> ChatGroq:
        """Create a ChatGroq instance with current key"""
        return ChatGroq(
            api_key=self.get_current_key(),
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            **kwargs
        )

    def safe_llm_call(self,
                     model: ChatGroq,
                     prompt: Any,
                     operation_name: str = "Processing",
                     max_key_attempts: Optional[int] = None) -> Optional[str]:
        """
        Execute LLM call with automatic key rotation on rate limits

        Args:
            model: ChatGroq instance (will be recreated with new keys)
            prompt: Prompt to send to the model
            operation_name: Description for error messages
            max_key_attempts: Maximum keys to try (None = try all keys)

        Returns:
            Response content string or None if all keys exhausted
        """
        max_attempts = max_key_attempts or len(self.keys)
        initial_index = self.current_index
        attempts = 0

        while attempts < max_attempts:
            try:
                # Create fresh model with current key
                current_model = self.create_llm(
                    model=model.model_name,
                    temperature=model.temperature,
                    max_retries=getattr(model, 'max_retries', 2)
                )

                response = current_model.invoke(prompt)
                return response.content.strip()

            except RateLimitError as e:
                attempts += 1

                # If we've tried all keys, show error and give up
                if attempts >= max_attempts:
                    error_msg = str(e)
                    wait_time_match = re.search(r'try again in ([\d]+[mh][\d]+\.[\d]+s)', error_msg)

                    if wait_time_match:
                        wait_time = wait_time_match.group(1)
                        st.error(f"""
⏰ **All API Keys Rate Limited**

All {len(self.keys)} Groq API keys have reached their rate limit.

**Please try again in: {wait_time}**

💡 **Tips:**
- Use "Quick Mode" for large PDFs
- Reduce number of pages/questions
- Rate limits reset every 24 hours

🔗 [Upgrade to Dev Tier](https://console.groq.com/settings/billing)
                        """)
                    else:
                        st.error(f"""
⏰ **All API Keys Rate Limited**

All {len(self.keys)} available API keys have been exhausted.
Please wait for rate limits to reset (usually within a few hours).
                        """)

                    return None

                # Rotate to next key and retry
                self.rotate_key()
                st.warning(f"🔄 Rate limit hit. Trying backup key {attempts + 1}/{max_attempts}...")

            except Exception as e:
                st.error(f"❌ {operation_name} failed: {str(e)}")
                return None

        return None


# Global singleton instance
_groq_manager = None

def get_groq_manager() -> GroqKeyManager:
    """Get or create the global Groq key manager"""
    global _groq_manager
    if _groq_manager is None:
        _groq_manager = GroqKeyManager()
    return _groq_manager
