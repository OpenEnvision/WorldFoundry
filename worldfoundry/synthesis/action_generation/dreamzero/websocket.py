# Inference-only DreamZero runtime retained in-tree.
"""Server for serving a policy over websockets.

Adapted from https://github.com/robo-arena/roboarena/

"""

import asyncio
import dataclasses
import logging
import traceback

import websockets.asyncio.server
import websockets.frames

from . import protocol as msgpack_numpy
from .protocol import BasePolicy


@dataclasses.dataclass
class PolicyServerConfig:
    # Resolution that images get resized to client-side, None means no resizing.
    # It's beneficial to resize images to the desired resolution client-side for faster communication.
    image_resolution: tuple[int, int] | None = (224, 224)
    # Whether or not wrist camera image(s) should be sent.
    needs_wrist_camera: bool = True
    # Number of external cameras to send.
    n_external_cameras: int = 1  # can be in [0, 1, 2]
    # Whether or not stereo camera image(s) should be sent.
    needs_stereo_camera: bool = False
    # Whether or not the unique eval session id should be sent (e.g. for policies that want to keep track of history).
    needs_session_id: bool = False
    # Which action space to use.
    action_space: str = (
        "joint_position"  # can be in ["joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity"]
    )


class WebsocketPolicyServer:
    """
    Serves a policy using the websocket protocol.

    Interface:
      Observation:
        - observation/wrist_image_left: (H, W, 3) if needs_wrist_camera is True
        - observation/wrist_image_right: (H, W, 3) if needs_wrist_camera is True and needs_stereo_camera is True
        - observation/exterior_image_{i}_left: (H, W, 3) if n_external_cameras >= 1
        - observation/exterior_image_{i}_right: (H, W, 3) if needs_stereo_camera is True
        - session_id: (1,) if needs_session_id is True
        - observation/joint_position: (7,)
        - observation/cartesian_position: (6,)
        - observation/gripper_position: (1,)
        - prompt: str, the natural language task instruction for the policy

      Action:
        - action: (N, 8,) or (N, 7,): either 7 movement actions (for joint action spaces) or 6 (for cartesian) plus one dimension for gripper position
                           --> all N actions will get executed on the robot before the server is queried again

    """

    def __init__(
        self,
        policy: BasePolicy,
        server_config: PolicyServerConfig,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._policy = policy
        self._server_config = server_config
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send server config to client to configure what gets sent to server.
        await websocket.send(packer.pack(dataclasses.asdict(self._server_config)))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())

                endpoint = obs["endpoint"]
                del obs["endpoint"]
                if endpoint == "reset":
                    self._policy.reset(obs)
                    to_return = "reset successful"
                else:
                    action = self._policy.infer(obs)
                    to_return = packer.pack(action)
                await websocket.send(to_return)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


if __name__ == "__main__":
    import numpy as np

    class DummyPolicy(BasePolicy):
        def infer(self, obs):
            return np.zeros((1, 8), dtype=np.float32)

        def reset(self, reset_info):
            pass

    logging.basicConfig(level=logging.INFO)
    policy = DummyPolicy()
    server = WebsocketPolicyServer(policy, PolicyServerConfig())
    server.serve_forever()
