import time
import megfile
from api import BaseAPIHandler
from typing import List
from utils.misc import *
import requests

class VLMAPIHandler(BaseAPIHandler):
    """
    Handler for VLM API batch and chat operations.
    Provides methods for file upload, batch job creation, polling, result/error download, and chat completion.
    """

    def __init__(self, api_key: Optional[str], api_url: Optional[str]):
        """
        Initialize VLMAPIHandler with API key and endpoint URL.
        """
        self.api_url = api_url
        self.max_retry = 64
        self.retry_delay = 5
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("VLM_API_KEY")
        assert self.api_key is not None, "Invalid VLM API key!"
        assert self.api_url, "Invalid VLM API url!"


    def submit_messages(self, messages: List, model_name: str = "Qwen3.5-397B-A17B", stream: bool = False) -> Dict:
        """
        Submit a chat completion request with messages and return the response.
        """
        headers = {
            'Authorization': self.api_key,
            'Content-Type': "application/json"
        }

        data = {
            "model": model_name,
            "messages": messages,
            "stream": stream}


        for attempt in range(1, self.max_retry + 1):
            try:
                response_api = requests.post(self.api_url, headers=headers, data=json.dumps(data))
                break
            except Exception as e:
                if attempt == self.max_retry:
                    raise RuntimeError(f"LLM score failed after {self.max_retry} attempt: {e}")
                time.sleep(self.retry_delay)

        #response_api = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        result = response_api.json()
        response_content = result['choices'][0]['message']['content']

        return {"response": response_content}

    @staticmethod
    def prepare_messages(image_links: Optional[List[Any]] = None, text_prompt: str = "", force_same_size: bool = False) -> List[Dict]:
        """
        Prepare messages for chat completion, encoding images as base64.
        """
        if image_links is not None:
            if not isinstance(image_links, list):
                image_links = [image_links]
            if force_same_size:
                image_links = [Image.open(megfile.smart_open(image, 'rb')).convert('RGB') for image in image_links]
                target_shape = image_links[-1].size
                image_links = [image.resize(target_shape) for image in image_links]

            image_links_base64 = [encode_image(img_link) for img_link in image_links]
            img_content = []
            for idx, img_link in enumerate(image_links_base64):
                #img_content.append({"type": "text", "text": f"Image {idx + 1}:"})
                img_content.append({"type": "image_url", "image_url": img_link})
            messages = [
                {
                    "role": "user",
                    "content": img_content + [{"type": "text", "text": text_prompt}]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text_prompt}]
                }
            ]
        return messages
