"""Automated Unit & Empirical Integration Tests for BIB-2 Neuro-Vision System."""

import os
import sys
import tempfile
import numpy as np
import pytest
from PIL import Image

# Ensure project and BIB-2 paths are discoverable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BIB2_PATH = os.path.join(_ROOT, "BIB-2")
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bib2.brain import BIB2NervousSystem
from neuro_vision.retina import RetinalProcessor, gabor_kernel
from neuro_vision.dataset import generate_benchmark_dataset, render_object_image, CLASSES
from neuro_vision.adapter import VisionAdapter
from neuro_vision.hud import VisionHUD
from neuro_vision.run_vision import run_benchmark


class TestRetinalVentralStream:
    """Tests for the biological ventral visual processing pipeline."""

    def test_gabor_kernel_generation(self):
        kernel = gabor_kernel(size=7, theta=0.0)
        assert kernel.shape == (7, 7)
        assert kernel.dtype == np.float32
        assert not np.isnan(kernel).any()

    def test_retina_processing_pipeline(self):
        rp = RetinalProcessor(target_size=(64, 64))

        # Test float numpy array input [0, 1]
        dummy_img = np.zeros((64, 64), dtype=np.float32)
        dummy_img[20:44, 20:44] = 0.9  # Central high-contrast square

        sensory = rp.process_image(dummy_img)

        assert "raw" in sensory
        assert "lgn" in sensory
        assert "foveated" in sensory
        assert "v1" in sensory
        assert "it_vector" in sensory
        assert "dominant_orientation" in sensory

        assert sensory["raw"].shape == (64, 64)
        assert sensory["lgn"].shape == (64, 64)
        assert sensory["foveated"].shape == (64, 64)
        assert sensory["v1"].shape == (64, 64)
        assert sensory["it_vector"].shape == (64,)
        assert sensory["it_vector"].dtype == np.float32

        # Verify unit normalization of IT population code
        norm = np.linalg.norm(sensory["it_vector"])
        assert np.isclose(norm, 1.0, atol=1e-3)

    def test_input_type_flexibility(self):
        rp = RetinalProcessor()

        # PIL Image
        pil_img = Image.new("L", (80, 80), color=128)
        res_pil = rp.process_image(pil_img)
        assert res_pil["it_vector"].shape == (64,)

        # RGB numpy array (uint8)
        rgb_arr = np.ones((70, 70, 3), dtype=np.uint8) * 150
        res_rgb = rp.process_image(rgb_arr)
        assert res_rgb["it_vector"].shape == (64,)

    def test_raw_ventral_stream_ablation(self):
        """Verifies that use_preprocessing=False skips foveation and size constancy."""
        rp = RetinalProcessor()
        dummy_img = np.zeros((64, 64), dtype=np.float32)
        dummy_img[5:15, 5:15] = 0.9  # Off-center square

        res_preprocessed = rp.process_image(dummy_img, use_preprocessing=True)
        res_raw = rp.process_image(dummy_img, use_preprocessing=False)

        assert res_preprocessed["use_preprocessing"] is True
        assert res_raw["use_preprocessing"] is False

        # In raw mode, foveated field is identical to unshifted lgn
        assert np.allclose(res_raw["foveated"], res_raw["lgn"])
        # IT vectors should differ because foveated was centered while raw was off-center
        assert not np.allclose(res_raw["it_vector"], res_preprocessed["it_vector"])


class TestDatasetGenerator:
    """Tests for procedural object drawing and variation generation."""

    def test_classes_and_counts(self):
        ds = generate_benchmark_dataset()

        assert ds["classes"] == CLASSES
        assert len(ds["classes"]) == 6
        assert len(ds["prototypes"]) == 6

        # 6 classes x 5 variations = 30 test samples
        assert len(ds["test_samples"]) == 30

    def test_variation_types(self):
        ds = generate_benchmark_dataset()
        variations = {s["variation"] for s in ds["test_samples"]}
        expected = {"rotated", "scaled", "translated", "noisy", "alternate"}
        assert variations == expected

    def test_image_value_ranges(self):
        ds = generate_benchmark_dataset()
        for cls_name, proto in ds["prototypes"].items():
            assert proto.shape == (64, 64)
            assert 0.0 <= proto.min() <= proto.max() <= 1.0

        for s in ds["test_samples"]:
            img = s["image"]
            assert img.shape == (64, 64)
            assert 0.0 <= img.min() <= img.max() <= 1.0


class TestVisionAdapter:
    """Tests for BIB-2 VisionAdapter, 1-shot learning, and cognitive telemetry."""

    @pytest.fixture
    def setup_system(self):
        brain = BIB2NervousSystem(feature_dim=64, seed=42)
        adapter = VisionAdapter(brain)
        dataset = generate_benchmark_dataset()
        return brain, adapter, dataset

    def test_1shot_imprinting_and_sleep(self, setup_system):
        brain, adapter, dataset = setup_system

        train_stats = adapter.train_prototypes(dataset["prototypes"], include_saccades=True)

        assert train_stats["total_imprinted"] == 18  # 6 classes x (1 canonical + 2 saccades)
        assert len(brain.limbic.hippocampus.ca3.stored_patterns) == 18
        assert brain.tick_count >= 6
        assert train_stats["sleep_stage"] == "SleepStage.WAKE"

        # Check Basal Ganglia D1 reinforcement
        assert (brain.basal_ganglia.d1_weights[:6] > 1.0).all()

    def test_prediction_telemetry(self, setup_system):
        brain, adapter, dataset = setup_system
        adapter.train_prototypes(dataset["prototypes"], include_saccades=True)

        sample = dataset["test_samples"][0]
        res = adapter.predict(sample["image"])

        assert "predicted_class" in res
        assert res["predicted_class"] in CLASSES
        assert "confidence" in res
        assert 0.0 <= res["confidence"] <= 1.0
        assert "winner_idx" in res
        assert "disinhibition" in res
        assert "proposals" in res
        assert len(res["proposals"]) == 6

        telemetry = res["telemetry"]
        assert "heart_rate" in telemetry
        assert 50 <= telemetry["heart_rate"] <= 150
        assert "dopamine" in telemetry
        assert "cortisol" in telemetry

        activations = telemetry["brain_activations"]
        assert "v1" in activations
        assert "ventral_stream" in activations
        assert "dlpfc" in activations
        assert "hippocampus" in activations
        assert "basal_ganglia" in activations
        assert "amygdala" in activations
        assert "thalamus" in activations
        assert "brainstem" in activations

    def test_generalization_benchmark_accuracy(self, setup_system):
        brain, adapter, dataset = setup_system
        adapter.train_prototypes(dataset["prototypes"], include_saccades=True)

        # 1. Raw Ventral Stream (Zero algorithmic preprocessing / pure CA3 learning >= 40%)
        results_raw = run_benchmark(adapter, dataset, verbose=False, use_preprocessing=False)
        assert results_raw["accuracy_pct"] >= 40.0, f"Expected raw CA3 >=40%, got {results_raw['accuracy_pct']}%"

        # 2. Full Pipeline (With saccadic foveation & size constancy >= 80%)
        results_full = run_benchmark(adapter, dataset, verbose=False, use_preprocessing=True)
        assert results_full["accuracy_pct"] >= 80.0, f"Expected full pipeline >=80%, got {results_full['accuracy_pct']}%"

        by_var = results_full["by_variation"]
        trans_acc = (by_var["translated"][0] / by_var["translated"][1]) * 100.0
        scale_acc = (by_var["scaled"][0] / by_var["scaled"][1]) * 100.0
        assert trans_acc >= 80.0, f"Translation invariance {trans_acc}% < 80%"
        assert scale_acc >= 80.0, f"Scale invariance {scale_acc}% < 80%"

    def test_checkpoint_save_and_load(self, setup_system):
        brain, adapter, dataset = setup_system
        adapter.train_prototypes(dataset["prototypes"], include_saccades=True)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            adapter.save_checkpoint(tmp_path)
            assert os.path.exists(tmp_path)

            # Create fresh uninitialized brain
            fresh_brain = BIB2NervousSystem(feature_dim=64, seed=99)
            fresh_adapter = VisionAdapter(fresh_brain)
            fresh_adapter.load_checkpoint(tmp_path)

            # Verify identical predictions
            sample = dataset["test_samples"][0]
            pred_orig = adapter.predict(sample["image"])
            pred_fresh = fresh_adapter.predict(sample["image"])

            assert pred_orig["predicted_class"] == pred_fresh["predicted_class"]
            assert np.isclose(pred_orig["confidence"], pred_fresh["confidence"], atol=1e-3)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestVisionHUDHeadless:
    """Tests for GUI rendering in headless environment."""

    def test_hud_render_without_crash(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        brain = BIB2NervousSystem(feature_dim=64, seed=42)
        adapter = VisionAdapter(brain)
        hud = VisionHUD()
        dataset = generate_benchmark_dataset()
        adapter.train_prototypes(dataset["prototypes"], include_saccades=True)

        sample = dataset["test_samples"][0]
        pred = adapter.predict(sample["image"])

        # Render frame
        hud.render(
            current_sample=sample,
            prediction_result=pred,
            benchmark_stats={"correct": 24, "total": 30, "by_variation": {"rotated": [4, 6]}},
            sample_idx=0,
            total_samples=30,
        )
        assert True
