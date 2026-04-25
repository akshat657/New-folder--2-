import os
import requests
from typing import List, Optional
import streamlit as st


class YouTubeTranscriptAPI:
    """Wrapper for RapidAPI YouTube Transcript API"""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.base_url = "https://youtube-transcript3.p.rapidapi.com/api/transcript"

    def _load_api_keys(self) -> List[str]:
        """Load RapidAPI keys from environment"""
        keys = []

        # Primary key
        primary = os.getenv("RAPIDAPI_KEY")
        if primary:
            keys.append(primary)

        # Fallback keys
        for i in range(1, 10):
            key = os.getenv(f"RAPIDAPI_KEY_{i}")
            if key:
                keys.append(key)

        if not keys:
            raise ValueError("No RapidAPI key found. Please set RAPIDAPI_KEY in .env file")

        return keys

    def _get_headers(self) -> dict:
        """Get headers with current API key"""
        return {
            "x-rapidapi-key": self.api_keys[self.current_key_index],
            "x-rapidapi-host": "youtube-transcript3.p.rapidapi.com"
        }

    def fetch_transcript(self, video_id: str, languages: Optional[List[str]] = None) -> str:
        """
        Fetch transcript for a YouTube video

        Args:
            video_id: YouTube video ID
            languages: List of language codes (e.g., ["en", "hi"])

        Returns:
            Transcript text as a single string

        Raises:
            Exception with user-friendly error message
        """
        if languages is None:
            languages = ["en", "hi"]

        # Try each language in order
        for lang in languages:
            try:
                transcript = self._fetch_with_retry(video_id, lang)
                if transcript:
                    return transcript
            except Exception:
                # Try next language
                continue

        # If all languages failed
        raise Exception(
            f"No transcript found for video {video_id}. "
            f"Tried languages: {', '.join(languages)}"
        )

    def _fetch_with_retry(self, video_id: str, lang: str) -> Optional[str]:
        """Fetch transcript with automatic key rotation"""
        max_attempts = len(self.api_keys)

        for attempt in range(max_attempts):
            try:
                querystring = {"videoId": video_id, "lang": lang}
                headers = self._get_headers()

                response = requests.get(
                    self.base_url,
                    headers=headers,
                    params=querystring,
                    timeout=10
                )

                # Check for rate limit (429) or other errors
                if response.status_code == 429:
                    # Rotate key and retry
                    if attempt < max_attempts - 1:
                        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                        st.warning(f"🔄 RapidAPI rate limit. Trying backup key {attempt + 2}/{max_attempts}...")
                        continue
                    else:
                        raise Exception("All RapidAPI keys have been rate limited. Please try again later.")

                response.raise_for_status()
                data = response.json()

                # Extract transcript text
                if "transcript" in data:
                    transcript_parts = data["transcript"]
                    if isinstance(transcript_parts, list):
                        return " ".join(part.get("text", "") for part in transcript_parts)
                    elif isinstance(transcript_parts, str):
                        return transcript_parts

                # If no transcript in response
                return None

            except requests.exceptions.Timeout:
                raise Exception("Request timed out. Please check your internet connection.")

            except requests.exceptions.RequestException as e:
                if attempt < max_attempts - 1:
                    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
                    continue
                else:
                    raise Exception(f"Failed to fetch transcript: {str(e)}")

        return None


# Global singleton
_youtube_api = None

def get_youtube_api() -> YouTubeTranscriptAPI:
    """Get or create the global YouTube API instance"""
    global _youtube_api
    if _youtube_api is None:
        _youtube_api = YouTubeTranscriptAPI()
    return _youtube_api
