# Inference-only Wan-VA source retained in-tree.
import logging
import os
import time
from typing import Dict  # noqa: UP035 - Re-exported to modules that star-import this namespace.

import websockets.sync.client
from typing_extensions import override

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        timeout = float(os.environ.get("WORLDFOUNDRY_LINGBOT_VA_SERVER_WAIT_TIMEOUT") or 0)
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                # Disable keepalive pings so long inference calls do not time out the connection.
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except Exception as e:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for server at {self._uri}") from e
                logging.info(f"Still waiting for server... (Error: {e})")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006 - Annotation keeps the re-exported typing.Dict name.
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self) -> None:
        self.infer({"reset": True})
