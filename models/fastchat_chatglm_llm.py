
from langchain.llms.utils import enforce_stop_tokens
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC
from langchain.llms.base import LLM
from typing import Optional, List
from models.base import (BaseAnswer,
                         AnswerResult)
import requests
import json

from models.loader import LoaderCheckPoint


class FastChatGLMLLM(BaseAnswer, LLM, ABC):
    api_base_url: str = "http://localhost:6006/"
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    checkPoint: LoaderCheckPoint = None
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # headers中添加上content-type这个参数，指定为json格式
        headers = {'Content-Type': 'application/json'}
        data=json.dumps({
          'prompt':prompt,
          'temperature':self.temperature,
          'history':self.history,
          'max_length':self.max_token
        })
        # print("ChatGLM prompt:",prompt)
        # 调用api
        response = requests.post(f"{self.api_base_url}/api",headers=headers,data=data)
        print("ChatGLM resp:",response)
        if response.status_code!=200:
          return "查询结果错误"
        resp = response.json()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history+[[None, resp['response']]]
        return resp['response']

    @property
    def _api_base_url(self) -> str:
        return self.api_base_url

    def set_api_base_url(self, api_base_url: str):
        self.api_base_url = api_base_url

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def generatorAnswer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False):
        # headers中添加上content-type这个参数，指定为json格式
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({
            'prompt': prompt,
            'temperature': self.temperature,
            'history': self.history,
            'max_length': self.max_token
        })
        print("ChatGLM prompt:",prompt)
        # 调用api
        response = requests.post(f"{self.api_base_url}", headers=headers, data=data)
        history += [[prompt, response.text]]
        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": response.text}
        yield answer_result