import numpy as np
from PIL import Image

from worldfoundry.synthesis.visual_generation.memory import VisualFrameMemory


def test_matrix_game_2_memory_records_numpy_video_chunk():
    memory = VisualFrameMemory(model_id="matrix-game-2")
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255

    memory.record(frames, type="video_chunk")

    selected = memory.select()
    assert isinstance(selected, Image.Image)
    assert selected.size == (8, 8)
    assert len(memory.all_frames) == 2
    assert np.asarray(selected)[0, 0, 0] == 255
