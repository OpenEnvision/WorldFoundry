# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from worldfoundry.base_models.llm_mllm_core.mllm.modelscope_swift.swift.llm import InferRequest
from worldfoundry.base_models.llm_mllm_core.mllm.modelscope_swift.swift.plugin import Metric

from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse, RequestConfig


class BaseInferEngine(ABC):
    @abstractmethod
    def infer(
        self,
        infer_requests: list[InferRequest],
        request_config: RequestConfig | None = None,
        metrics: list[Metric] | None = None,
        *,
        use_tqdm: bool | None = None,
        **kwargs,
    ) -> list[ChatCompletionResponse | Iterator[ChatCompletionStreamResponse]]:
        """
        This method performs inference on a list of inference requests.

        The method takes a list of inference requests and processes them according to the provided configuration.
        It can optionally use tqdm for progress visualization and accept additional keyword arguments.

        Args:
            infer_requests (List[InferRequest]): A list of inference requests to be processed.
            request_config (Optional[RequestConfig]): Configuration for the request, if any.
            metrics (Optional[List[Metric]]): A list of usage information to return.
            use_tqdm (Optional[bool]): Whether to use tqdm for progress visualization.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
                The result of the inference.
        """

    @abstractmethod
    async def infer_async(
        self, infer_request: InferRequest, request_config: RequestConfig | None = None, **kwargs
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionStreamResponse]:
        """
        This method performs asynchronous inference on a single inference request.

        The method takes an inference request and processes it according to the provided configuration.
        It can accept additional keyword arguments.

        Args:
            infer_request (InferRequest): An inference request to be processed.
            request_config (Optional[RequestConfig]): Configuration for the request, if any.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]: The result of
                the asynchronous inference.
        """
