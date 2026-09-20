"""Neurobiological Vision Adapter: Connects Ventral Visual Stream to BIB-2 Master Nervous System."""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from PIL import Image

# Ensure BIB-2 library is available
_BIM2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BIB2_PATH = os.path.join(_BIM2_ROOT, "BIB-2")
if _BIB2_PATH not in sys.path:
    sys.path.insert(0, _BIB2_PATH)

from bib2.brain import BIB2NervousSystem
from bib2.adapters.base import BaseNeuralAdapter
from bib2.sleep.orchestrator import SleepStage
from .retina import RetinalProcessor
from .dataset import CLASSES


class VisionAdapter(BaseNeuralAdapter):
    """
    Biological Vision Adapter interfacing the Ventral Stream with BIB-2's 14 anatomical subsystems:
    1. Retinotopic Optic Nerve CN_II: 64-dimensional IT population code.
    2. Superior Colliculus Reflex: Saccadic foveation and gaze centering.
    3. Thalamic Reticular Nucleus (LGN & TRN): Noise suppression and sensory relay.
    4. Hippocampal CA3 Attractor: Auto-associative prototype storage and pattern completion.
    5. Basal Ganglia Tripartite Gating: Categorical action selection across Direct (D1) vs Indirect (D2) pathways.
    6. Autonomic Nervous System: Sympathetic arousal / ECG spike on novel or ambiguous visual inputs.
    7. 2D Macro-Brain Activation Telemetry: Real-time energy levels across all 8 major functional regions.
    """

    def __init__(self, brain: BIB2NervousSystem, classes: Optional[List[str]] = None):
        super().__init__(brain)
        self.classes = classes or list(CLASSES)
        self.retina = RetinalProcessor()
        self.class_prototypes: Dict[str, List[np.ndarray]] = {cls: [] for cls in self.classes}

        self.last_predicted_class: Optional[str] = None
        self.last_confidence: float = 0.0
        self.last_action_idx: int = 0
        self.last_sensory: Optional[Dict[str, Any]] = None
        self.last_proposals: Dict[str, float] = {}
        self.last_ca3_reconstruction: Optional[np.ndarray] = None

    def train_prototypes(
        self,
        prototypes: Dict[str, Any],
        exemplars: Optional[List[Dict[str, Any]]] = None,
        include_saccades: bool = True,
    ) -> Dict[str, Any]:
        """
        1-Shot Episodic Imprinting into Hippocampal CA3 and Basal Ganglia D1 Pathways.
        - For each class, presents canonical prototype image to Optic Nerve (CN_II).
        - Master clock ticks (brain.tick()).
        - Hebbian imprinting into Hippocampal CA3 auto-associative attractor network.
        - Reinforces Basal Ganglia D1 pathway for the target category index.
        - Runs SWS sleep consolidation (Tononi SHY synaptic downscaling).
        """
        imprinted_counts = {cls: 0 for cls in self.classes}

        for cls_name, img_input in prototypes.items():
            if cls_name not in self.classes:
                continue

            cls_idx = self.classes.index(cls_name)

            # 1. Ventral visual stream processing
            sensory = self.retina.process_image(img_input)
            it_vec = sensory["it_vector"]

            # 2. Present to CN_II Optic Nerve
            self.brain.peripheral.cranial.set_sensory("CN_II", it_vec)
            self.brain.tick()

            # 3. Store into Hippocampal CA3 attractor
            self.brain.limbic.hippocampus.ca3.store(it_vec, lr=1.0)
            self.class_prototypes[cls_name].append(it_vec.copy())
            imprinted_counts[cls_name] += 1

            # 4. Dopaminergic Corticostriatal Plasticity (Basal Ganglia D1 Go Pathway)
            self.brain.basal_ganglia.reinforce_action(cls_idx, reward_rpe=1.0, lr=0.4)
            self.brain.chemistry.matrix.state.dopamine = float(
                np.clip(self.brain.chemistry.matrix.state.dopamine + 0.1, 0.1, 1.0)
            )

            # 5. Biological Microsaccades / Ocular Tremors
            if include_saccades:
                base_pil = Image.fromarray((sensory["raw"] * 255).astype(np.uint8))
                w, h = base_pil.size
                for angle in [-15, 15]:
                    rot = base_pil.rotate(angle, resample=Image.Resampling.BILINEAR)
                    rot_sensory = self.retina.process_image(np.asarray(rot, dtype=np.float32) / 255.0)
                    saccade_vec = rot_sensory["it_vector"]
                    self.brain.limbic.hippocampus.ca3.store(saccade_vec, lr=0.4)
                    self.class_prototypes[cls_name].append(saccade_vec.copy())
                    imprinted_counts[cls_name] += 1

        # Also imprint secondary exemplars if provided
        if exemplars:
            for ex in exemplars:
                cls_name = ex.get("class")
                if cls_name in self.classes:
                    cls_idx = self.classes.index(cls_name)
                    sensory = self.retina.process_image(ex["image"])
                    ex_vec = sensory["it_vector"]
                    self.brain.limbic.hippocampus.ca3.store(ex_vec, lr=0.8)
                    self.class_prototypes[cls_name].append(ex_vec.copy())
                    imprinted_counts[cls_name] += 1
                    self.brain.basal_ganglia.reinforce_action(cls_idx, reward_rpe=0.8, lr=0.3)

        # 6. Sleep Consolidation (Slow-Wave Sleep & Tononi SHY Synaptic Downscaling)
        self.brain.sleep.transition_to(SleepStage.NREM_SWS)
        self.brain.limbic.hippocampus.ca3.recurrent_matrix = self.brain.sleep.shy.downscale(
            self.brain.limbic.hippocampus.ca3.recurrent_matrix, downscale_factor=0.95
        )
        self.brain.sleep.transition_to(SleepStage.WAKE)

        return {
            "imprinted_counts": imprinted_counts,
            "total_imprinted": sum(imprinted_counts.values()),
            "sleep_stage": str(self.brain.sleep.current_stage),
            "tick_count": self.brain.tick_count,
        }

    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        use_preprocessing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Visual Inference on Novel Unseen Image:
        1. Ventral stream processes image -> 64-dim IT population code.
           (Supports use_preprocessing=False for raw ablation without algorithmic centering/scaling).
        2. Ingests into Optic Nerve CN_II and steps master clock cycle.
        3. Hippocampal CA3 attractor network executes auto-associative pattern completion.
        4. Cortical proposals computed from CA3 completion and invariant feature overlap.
        5. Basal Ganglia selects winning category channel across D1/D2 pathways.
        6. Autonomic system modulates heart rate (elevated if ambiguous, calm if recognized).
        """
        # 1. Ventral visual stream extraction
        sensory = self.retina.process_image(image_input, use_preprocessing=use_preprocessing)
        it_vec = sensory["it_vector"]
        self.last_sensory = sensory

        # 2. Ingest into CN_II Optic Nerve & Tick Brain Clock
        self.brain.peripheral.cranial.set_sensory("CN_II", it_vec)
        self.brain.tick()

        # 3. Hippocampal CA3 Attractor Drive & Pattern Completion
        drive = np.dot(it_vec, self.brain.limbic.hippocampus.ca3.recurrent_matrix)
        drive_norm = drive / (np.linalg.norm(drive) + 1e-6)
        self.last_ca3_reconstruction = drive_norm.astype(np.float32)

        # 4. Compute Cortical Proposal Scores for Each Category
        proposal_scores = {}
        for cls in self.classes:
            stored_bank = self.class_prototypes[cls]
            if not stored_bank:
                proposal_scores[cls] = 0.0
                continue

            att_match = max(float(np.dot(drive_norm, p)) for p in stored_bank)
            dir_match = max(float(np.dot(it_vec, p)) for p in stored_bank)
            best_match = 0.30 * att_match + 0.70 * dir_match
            proposal_scores[cls] = float(np.clip(best_match, 0.0, 1.0))

        self.last_proposals = proposal_scores

        # Convert proposals to 8-channel array for Basal Ganglia
        proposal_arr = np.zeros(8, dtype=np.float32)
        for idx, cls in enumerate(self.classes[:8]):
            proposal_arr[idx] = proposal_scores[cls]

        # 5. Basal Ganglia Gating & Action Selection
        dopamine = float(self.brain.chemistry.matrix.state.dopamine)
        winner_idx, disinhibition = self.brain.basal_ganglia.select_action(
            proposal_arr, dopamine_level=dopamine
        )

        winning_cls_idx = winner_idx % len(self.classes)
        predicted_class = self.classes[winning_cls_idx]
        confidence = float(proposal_scores[predicted_class])

        self.last_predicted_class = predicted_class
        self.last_confidence = confidence
        self.last_action_idx = winning_cls_idx

        # 6. Autonomic Modulation & Amygdala Ambiguity Response
        sorted_scores = sorted(proposal_scores.values(), reverse=True)
        margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0

        if confidence > 0.85 and margin > 0.03:
            # High certainty & recognition: Vagal parasympathetic tone promotes calm
            self.brain.peripheral.autonomic.trigger_vagal_tone(intensity=0.3)
            self.brain.chemistry.matrix.state.cortisol = float(
                np.clip(self.brain.chemistry.matrix.state.cortisol - 0.05, 0.05, 1.0)
            )
        elif confidence < 0.65 or margin < 0.02:
            # Visual ambiguity or novelty: Amygdala threat/alert surge
            surge_intensity = float(np.clip(0.8 - confidence, 0.2, 0.7))
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=surge_intensity)
            self.brain.chemistry.matrix.state.cortisol = float(
                np.clip(self.brain.chemistry.matrix.state.cortisol + 0.10, 0.0, 1.0)
            )

        telemetry = self.get_telemetry()

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "winner_idx": winning_cls_idx,
            "disinhibition": disinhibition,
            "proposals": proposal_scores,
            "margin": margin,
            "sensory": sensory,
            "telemetry": telemetry,
        }

    def get_brain_activations(self) -> Dict[str, float]:
        """
        Extracts real-time scalar energy levels (0.05 to 1.0) across all 8 major functional brain regions
        for the cybernetic 2D macro-brain activation heatmap.
        """
        v1_val = float(np.mean(np.abs(self.brain.neocortex.registry.get_area("V1_Visual").l5_output)))
        dlpfc_val = float(np.mean(np.abs(self.brain.neocortex.registry.get_area("DLPFC_WorkingMemory").l5_output)))
        hip_val = float(np.mean(np.abs(self.brain.limbic.hippocampus.subiculum)))
        amy_val = float(np.clip(np.mean(self.brain.amygdala.bla_weights) * 2.0, 0.0, 1.0))
        bg_val = float(np.clip(np.mean(self.brain.basal_ganglia.d1_weights[:len(self.classes)]) / 2.0, 0.0, 1.0))
        thal_val = float(np.mean(np.abs(self.brain.thalamus.read_nucleus("LGN"))))
        ventral_val = float(self.last_confidence if self.last_confidence > 0 else 0.4)
        stem_val = float(np.clip(self.brain.peripheral.autonomic.sympathetic_tone, 0.0, 1.0))

        return {
            "v1": float(np.clip(v1_val * 2.5 + 0.2, 0.05, 1.0)),
            "ventral_stream": float(np.clip(ventral_val * 1.1, 0.05, 1.0)),
            "dlpfc": float(np.clip(dlpfc_val * 2.0 + 0.15, 0.05, 1.0)),
            "hippocampus": float(np.clip(max(hip_val * 1.5, self.last_confidence), 0.05, 1.0)),
            "basal_ganglia": float(np.clip(bg_val * 0.9 + 0.1, 0.05, 1.0)),
            "amygdala": float(np.clip(amy_val + stem_val * 0.4, 0.05, 1.0)),
            "thalamus": float(np.clip(thal_val * 2.0 + 0.2, 0.05, 1.0)),
            "brainstem": float(np.clip(stem_val, 0.05, 1.0)),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Collects autonomic, neurochemical, cognitive, and activation telemetry."""
        matrix = self.brain.chemistry.matrix.state
        autonomic = self.brain.peripheral.autonomic
        hr = int(68.0 + autonomic.sympathetic_tone * 65.0)

        return {
            "heart_rate": hr,
            "dopamine": float(matrix.dopamine),
            "cortisol": float(matrix.cortisol),
            "serotonin": float(matrix.serotonin),
            "norepinephrine": float(matrix.norepinephrine),
            "sympathetic_tone": float(autonomic.sympathetic_tone),
            "vagal_tone": float(autonomic.parasympathetic_tone),
            "predicted_class": self.last_predicted_class or "None",
            "confidence": self.last_confidence,
            "tick_count": self.brain.tick_count,
            "brain_activations": self.get_brain_activations(),
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Saves learned class prototypes, CA3 recurrent matrix, and Basal Ganglia weights."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        proto_data = {}
        for cls, vecs in self.class_prototypes.items():
            if vecs:
                proto_data[f"proto_{cls}"] = np.stack(vecs, axis=0)

        np.savez(
            filepath,
            ca3_recurrent=self.brain.limbic.hippocampus.ca3.recurrent_matrix,
            bg_d1=self.brain.basal_ganglia.d1_weights,
            bg_d2=self.brain.basal_ganglia.d2_weights,
            classes=np.array(self.classes),
            **proto_data,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Restores learned prototypes, CA3 attractor, and Basal Ganglia weights."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        self.brain.limbic.hippocampus.ca3.recurrent_matrix = data["ca3_recurrent"].astype(np.float32)
        self.brain.basal_ganglia.d1_weights = data["bg_d1"].astype(np.float32)
        self.brain.basal_ganglia.d2_weights = data["bg_d2"].astype(np.float32)

        self.class_prototypes = {cls: [] for cls in self.classes}
        for cls in self.classes:
            key = f"proto_{cls}"
            if key in data:
                stacked = data[key]
                self.class_prototypes[cls] = [stacked[i] for i in range(len(stacked))]
