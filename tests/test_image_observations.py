from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from dino_rl.config import ImageConfig, SimulatorConfig, config_from_dict, load_config
from dino_rl.envs import DinoEnv, ImageObservationWrapper
from dino_rl.envs.simulator import Obstacle
from dino_rl.preprocessing.frame_stack import FrameStack
from dino_rl.preprocessing.image_preprocessing import ImagePreprocessor


def test_crop_excludes_outside_pixels():
    rgb = np.zeros((5, 8, 3), dtype=np.uint8)
    rgb[1:3, 2:6] = 255
    original = rgb.copy()
    processor = ImagePreprocessor(ImageConfig(crop=(2, 1, 6, 3), width=4, height=2))
    assert np.all(processor.crop_grayscale(rgb) == 255)
    frame = processor(rgb)
    assert frame.shape == (2, 4) and frame.dtype == np.float32
    assert np.all(frame == 1)
    np.testing.assert_array_equal(rgb, original)


def test_rgb_luminance_and_normalization():
    colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    processor = ImagePreprocessor(ImageConfig(crop=None, width=3, height=1))
    np.testing.assert_array_equal(processor.process_uint8(colors), [[76, 150, 29]])
    np.testing.assert_allclose(processor(colors), np.array([[76, 150, 29]]) / 255)


def test_default_resize_and_bilinear_intermediate_pixels():
    env = DinoEnv(render_mode="rgb_array")
    env.reset(seed=42)
    processor = ImagePreprocessor()
    result = processor(env.render())
    assert result.shape == (84, 84) and result.dtype == np.float32
    assert np.isfinite(result).all() and 0 <= result.min() < result.max() <= 1
    # Resampling object edges yields intensities beyond the source palette.
    assert np.unique(processor.process_uint8(env.render())).size > 3
    env.close()


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((240, 800, 3), dtype=np.float32),
        np.zeros((240, 800), dtype=np.uint8),
        np.zeros((240, 800, 4), dtype=np.uint8),
        np.zeros((0, 800, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
        None,
    ],
)
def test_bad_frame_input_fails_clearly(frame):
    with pytest.raises(ValueError):
        ImagePreprocessor()(frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop": [-1, 0, 10, 10]},
        {"crop": [0, 0, 0, 5]},
        {"crop": [0, 0, 5]},
        {"crop": [0, 0, 10.0, 10]},
        {"width": 0},
        {"height": False},
        {"stack_size": 0},
        {"stack_size": 1.5},
    ],
)
def test_invalid_image_config(kwargs):
    with pytest.raises(ValueError):
        ImageConfig(**kwargs)


def test_image_config_yaml_and_old_checkpoint_compatibility(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "image:\n  crop: [1, 2, 100, 120]\n  width: 64\n  height: 48\n  stack_size: 3\n"
    )
    config = load_config(path)
    assert config.image.crop == (1, 2, 100, 120)
    assert config.image.width == 64 and config.image.height == 48 and config.image.stack_size == 3
    assert config_from_dict(asdict(config)) == config
    assert config_from_dict({}).image == ImageConfig()
    path.write_text("image:\n  typo: 1\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_stack_order_padding_and_no_aliasing():
    stack = FrameStack(4, (2, 3))
    first = np.full((2, 3), 10, dtype=np.uint8)
    initial = stack.reset(first)
    first[:] = 250
    np.testing.assert_allclose(initial, 10 / 255)
    for value in (20, 30, 40, 50):
        result = stack.append(np.full((2, 3), value, dtype=np.uint8))
    np.testing.assert_array_equal(stack.as_uint8()[:, 0, 0], [20, 30, 40, 50])
    assert result.flags.c_contiguous and result.dtype == np.float32
    result[:] = 0
    compact = stack.as_uint8()
    compact[:] = 0
    assert stack.as_uint8()[0, 0, 0] == 20
    np.testing.assert_allclose(initial, 10 / 255)
    reset = stack.reset(np.full((2, 3), 255, dtype=np.uint8))
    assert np.all(reset == 1)
    assert torch.from_numpy(reset).unsqueeze(0).shape == (1, 4, 2, 3)


def test_stack_lifecycle_and_invalid_frame():
    stack = FrameStack()
    with pytest.raises(RuntimeError, match="reset"):
        stack.append(np.zeros((84, 84), dtype=np.uint8))
    with pytest.raises(RuntimeError):
        stack.observation()
    with pytest.raises(ValueError):
        stack.reset(np.zeros((84, 84), dtype=np.float32))
    with pytest.raises(ValueError):
        stack.reset(np.zeros((83, 84), dtype=np.uint8))
    with pytest.raises(ValueError):
        FrameStack(0)
    stack.reset(np.zeros((84, 84), dtype=np.uint8))
    stack.clear()
    with pytest.raises(RuntimeError):
        stack.as_uint8()


def test_visual_wrapper_gym_contract():
    env = ImageObservationWrapper(DinoEnv(render_mode="rgb_array"))
    # Gymnasium advises checking unwrapped environments; here the wrapper API is the subject.
    with pytest.warns(UserWarning, match="different from the unwrapped"):
        check_env(env, skip_render_check=True)
    env.close()


def test_visual_observations_shift_with_one_physics_tick():
    env = ImageObservationWrapper(DinoEnv(render_mode="rgb_array"))
    previous, _ = env.reset(seed=42)
    for _ in range(110):
        current, _, terminated, truncated, info = env.step(0)
        assert not terminated and not truncated
        np.testing.assert_array_equal(current[:-1], previous[1:])
        assert env.observation_space.contains(current)
        previous = current
    assert info["episode_steps"] == 110
    assert not np.array_equal(current[0], current[-1])
    assert torch.from_numpy(current).unsqueeze(0).shape == (1, 4, 84, 84)
    env.close()


def test_same_actions_preserve_rewards_and_episode_endings():
    physics = SimulatorConfig(max_episode_steps=170)
    raw = DinoEnv(physics)
    visual = ImageObservationWrapper(DinoEnv(physics, render_mode="rgb_array"))
    rng = np.random.default_rng(4)
    raw.reset(seed=42)
    visual.reset(seed=42)
    for _ in range(500):
        action = int(rng.integers(3))
        raw_result, image_result = raw.step(action), visual.step(action)
        assert raw_result[1:] == image_result[1:]
        assert image_result[0].shape == (4, 84, 84)
        if raw_result[2] or raw_result[3]:
            raw.reset()
            visual.reset()
    raw.close()
    visual.close()


@pytest.mark.parametrize("crash", [True, False])
def test_final_frame_preserved_and_new_episode_clears_history(crash):
    physics = SimulatorConfig(max_episode_steps=1 if not crash else 6000)
    env = ImageObservationWrapper(DinoEnv(physics, render_mode="rgb_array"))
    start, _ = env.reset(seed=42)
    if crash:
        env.env.simulator.obstacles = [Obstacle(115, 22, 48, 0, 1)]
    final, _, terminated, truncated, _ = env.step(0)
    assert terminated is crash and truncated is not crash
    np.testing.assert_array_equal(final[-1], env.preprocessor(env.render()))
    with pytest.raises(RuntimeError):
        env.step(0)
    fresh, _ = env.reset(seed=42)
    np.testing.assert_array_equal(start, fresh)
    for channel in fresh:
        np.testing.assert_array_equal(channel, fresh[0])
    env.close()


def test_configurable_shapes_and_rgb_renderer_requirement():
    with pytest.raises(ValueError, match="rgb_array"):
        ImageObservationWrapper(DinoEnv())
    config = replace(ImageConfig(), crop=None, width=60, height=40, stack_size=2)
    env = ImageObservationWrapper(DinoEnv(render_mode="rgb_array"), config)
    observation, _ = env.reset(seed=42)
    assert observation.shape == (2, 40, 60) and env.observation_space.contains(observation)
    env.close()
