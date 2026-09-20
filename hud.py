"""Interactive Cybernetic Vision HUD with Real-Time Ventral Stream Decomposition, Macro-Brain Heatmap, and ECG."""

import math
import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pygame
from PIL import Image

from .dataset import CLASSES

# Color Palette
BG_DARK = (10, 14, 22)
PANEL_BG = (14, 20, 32)
PANEL_BORDER = (28, 44, 68)
GRID_LINE = (20, 30, 48)

TEXT_WHITE = (235, 242, 255)
TEXT_MUTED = (140, 160, 185)
TEXT_GOLD = (250, 204, 21)
TEXT_CYAN = (56, 189, 248)
TEXT_GREEN = (34, 197, 94)
TEXT_RED = (239, 68, 68)

ECG_LINE = (34, 211, 153)
ECG_BG = (8, 16, 24)

# Brain Anatomical Coordinates (Normalized within Center Panel 0.0 to 1.0)
BRAIN_NODES = {
    "v1": {"pos": (0.80, 0.40), "name": "V1 (Visual)", "tracts": ["ventral_stream", "thalamus"]},
    "ventral_stream": {"pos": (0.68, 0.58), "name": "Ventral IT", "tracts": ["hippocampus", "dlpfc"]},
    "thalamus": {"pos": (0.50, 0.40), "name": "Thalamus (LGN)", "tracts": ["v1", "basal_ganglia"]},
    "hippocampus": {"pos": (0.42, 0.60), "name": "Hippocampus CA3", "tracts": ["basal_ganglia", "amygdala"]},
    "basal_ganglia": {"pos": (0.36, 0.40), "name": "Basal Ganglia", "tracts": ["dlpfc", "amygdala"]},
    "dlpfc": {"pos": (0.24, 0.26), "name": "DLPFC (Prefrontal)", "tracts": ["basal_ganglia"]},
    "amygdala": {"pos": (0.28, 0.64), "name": "Amygdala", "tracts": ["brainstem"]},
    "brainstem": {"pos": (0.50, 0.82), "name": "Brainstem (ANS)", "tracts": []},
}


class VisionHUD:
    """
    1360x820 Cybernetic Visual Recognition HUD:
    - Ventral Visual Stream Decomposition (Raw -> LGN DoG -> Foveated -> V1 Gabor -> IT 64-Dim).
    - Real-Time 2D Macro-Brain Activation Heatmap & Synaptic Routing.
    - Basal Ganglia Categorical Action Proposals Bar Chart.
    - Real-Time Autonomic ECG Oscilloscope & Neurochemistry Telemetry.
    - Interactive Test Explorer and Automated Benchmark Mode.
    """

    def __init__(self, width: int = 1360, height: int = 820):
        pygame.init()
        pygame.display.set_caption("BIB-2 Neuro-Vision: Biomimetic Object Recognition & Generalization")
        self.width = width
        self.height = height

        # Headless safety
        try:
            self.screen = pygame.display.set_mode((width, height))
        except pygame.error:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.display.set_mode((width, height))

        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Consolas", 17, bold=True)
        self.font_body = pygame.font.SysFont("Consolas", 12)
        self.font_small = pygame.font.SysFont("Consolas", 10)
        self.font_large = pygame.font.SysFont("Consolas", 22, bold=True)

        # ECG State
        self.ecg_points: List[float] = [0.0] * 120
        self.ecg_phase: float = 0.0

    def _generate_ecg_sample(self, phase: float) -> float:
        """P-Q-R-S-T cardiac voltage waveform."""
        p = phase % (2.0 * math.pi)
        val = 0.0
        if 0.5 <= p < 1.0:
            val += 0.15 * math.sin((p - 0.5) * 2.0 * math.pi)
        elif 1.4 <= p < 1.6:
            val -= 0.20 * math.sin((p - 1.4) * 5.0 * math.pi)
        elif 1.6 <= p < 1.9:
            val += 1.30 * math.sin((p - 1.6) * 3.33 * math.pi)
        elif 1.9 <= p < 2.1:
            val -= 0.35 * math.sin((p - 1.9) * 5.0 * math.pi)
        elif 2.6 <= p < 3.4:
            val += 0.28 * math.sin((p - 2.6) * 1.25 * math.pi)
        return val

    def _update_ecg(self, heart_rate: float, dt: float = 0.033) -> None:
        """Advance cardiac rhythm oscilloscope based on heart rate."""
        freq = heart_rate / 60.0
        self.ecg_phase += 2.0 * math.pi * freq * dt
        sample = self._generate_ecg_sample(self.ecg_phase)
        self.ecg_points.pop(0)
        self.ecg_points.append(sample)

    def _array_to_surface(self, arr: np.ndarray, target_size: Tuple[int, int]) -> pygame.Surface:
        """Converts normalized float array [0, 1] to scaled Pygame Surface."""
        norm = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([norm, norm, norm], axis=-1)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        return pygame.transform.scale(surf, target_size)

    def render(
        self,
        current_sample: Dict[str, Any],
        prediction_result: Dict[str, Any],
        benchmark_stats: Optional[Dict[str, Any]] = None,
        sample_idx: int = 0,
        total_samples: int = 30,
    ) -> None:
        """Renders the full 1360x820 Cybernetic Vision Interface."""
        self.screen.fill(BG_DARK)
        telemetry = prediction_result.get("telemetry", {})
        sensory = prediction_result.get("sensory", {})
        hr = float(telemetry.get("heart_rate", 70.0))
        self._update_ecg(hr)

        # 1. Top Header Banner
        self._render_header(telemetry, sample_idx, total_samples)

        # 2. Left Panel: Visual Ventral Stream (x=20, y=65, w=380, h=700)
        self._render_ventral_stream_panel(20, 65, 380, 700, current_sample, sensory)

        # 3. Center Panel: 2D Macro-Brain Heatmap (x=415, y=65, w=530, h=700)
        self._render_macro_brain_panel(415, 65, 530, 700, telemetry)

        # 4. Right Panel: Categorical Decision & Telemetry (x=960, y=65, w=380, h=700)
        self._render_decision_panel(960, 65, 380, 700, current_sample, prediction_result, benchmark_stats)

        # 5. Bottom Instructions Bar
        self._render_footer()

        pygame.display.flip()
        self.clock.tick(30)

    def _render_header(self, telemetry: Dict[str, Any], sample_idx: int, total_samples: int) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (20, 15, 1320, 42), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (20, 15, 1320, 42), width=1, border_radius=4)

        title = self.font_large.render("BIB-2 NEURO-VISION: VENTRAL STREAM RECOGNITION", True, TEXT_GOLD)
        self.screen.blit(title, (35, 23))

        tick_str = f"CLOCK TICKS: {telemetry.get('tick_count', 0):,}"
        tick_txt = self.font_body.render(tick_str, True, TEXT_CYAN)
        self.screen.blit(tick_txt, (820, 27))

        test_str = f"SAMPLE {sample_idx + 1}/{total_samples}"
        test_txt = self.font_body.render(test_str, True, TEXT_WHITE)
        self.screen.blit(test_txt, (1080, 27))

        stat_dot = (34, 197, 94)
        pygame.draw.circle(self.screen, stat_dot, (1260, 35), 5)
        live_txt = self.font_small.render("ONLINE", True, stat_dot)
        self.screen.blit(live_txt, (1275, 29))

    def _render_ventral_stream_panel(
        self, x: int, y: int, w: int, h: int, sample: Dict[str, Any], sensory: Dict[str, Any]
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), width=1, border_radius=6)

        hdr = self.font_title.render("VENTRAL VISUAL STREAM", True, TEXT_CYAN)
        self.screen.blit(hdr, (x + 18, y + 16))

        # Main Input Image Card
        raw_img_arr = sample.get("image")
        var_name = sample.get("variation", "prototype").upper()
        true_cls = sample.get("class", "Unknown")

        # Frame for input image
        img_x, img_y = x + (w - 200) // 2, y + 48
        pygame.draw.rect(self.screen, (10, 16, 26), (img_x - 4, img_y - 4, 208, 208), border_radius=6)
        pygame.draw.rect(self.screen, TEXT_CYAN, (img_x - 4, img_y - 4, 208, 208), width=1, border_radius=6)

        if raw_img_arr is not None:
            surf_img = self._array_to_surface(raw_img_arr, (200, 200))
            self.screen.blit(surf_img, (img_x, img_y))

        # Metadata Badge
        var_color = TEXT_GOLD if var_name != "PROTOTYPE" else TEXT_GREEN
        var_txt = self.font_body.render(f"INPUT: {true_cls} [{var_name}]", True, var_color)
        self.screen.blit(var_txt, (x + 20, y + 266))

        # Cortical Decompositions (4 stages in a 2x2 grid)
        stage_y = y + 295
        lbl_stages = self.font_small.render("VENTRAL DECOMPOSITION STACK", True, TEXT_MUTED)
        self.screen.blit(lbl_stages, (x + 20, stage_y))

        stages = [
            ("LGN Center-Surround (DoG)", sensory.get("lgn")),
            ("Foveated / Size Norm", sensory.get("foveated")),
            ("V1 Gabor Wavelet Energy", sensory.get("v1")),
            ("Raw Retinal Array", sensory.get("raw")),
        ]

        grid_coords = [
            (x + 25, stage_y + 20),
            (x + 200, stage_y + 20),
            (x + 25, stage_y + 160),
            (x + 200, stage_y + 160),
        ]

        for idx, (label, arr) in enumerate(stages):
            gx, gy = grid_coords[idx]
            pygame.draw.rect(self.screen, (10, 16, 26), (gx - 2, gy - 2, 114, 114), border_radius=4)
            pygame.draw.rect(self.screen, PANEL_BORDER, (gx - 2, gy - 2, 114, 114), width=1, border_radius=4)
            if arr is not None:
                s_surf = self._array_to_surface(arr, (110, 110))
                self.screen.blit(s_surf, (gx, gy))
            lbl = self.font_small.render(label, True, TEXT_MUTED)
            self.screen.blit(lbl, (gx - 5, gy + 115))

        # CN_II Optic Nerve 64-Dim Heatmap Bar
        cn2_y = y + 590
        lbl_cn2 = self.font_small.render("OPTIC NERVE CN_II (64-DIM IT VECTOR)", True, TEXT_CYAN)
        self.screen.blit(lbl_cn2, (x + 20, cn2_y))

        it_vec = sensory.get("it_vector")
        bar_x, bar_y = x + 20, cn2_y + 20
        cell_w = 5
        cell_h = 24
        pygame.draw.rect(self.screen, (8, 12, 18), (bar_x - 2, bar_y - 2, 64 * cell_w + 4, cell_h + 4), border_radius=3)

        if it_vec is not None and len(it_vec) >= 64:
            for i in range(64):
                val = float(it_vec[i])
                # Map -0.3..0.3 to 0..255 cyan/blue/gold intensity
                norm_v = np.clip((val + 0.3) / 0.6, 0.0, 1.0)
                col = (
                    int(20 + norm_v * 180),
                    int(40 + norm_v * 180),
                    int(100 + norm_v * 155),
                )
                pygame.draw.rect(self.screen, col, (bar_x + i * cell_w, bar_y, cell_w - 1, cell_h))

    def _render_macro_brain_panel(self, x: int, y: int, w: int, h: int, telemetry: Dict[str, Any]) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), width=1, border_radius=6)

        hdr = self.font_title.render("2D MACRO-BRAIN ACTIVATION HEATMAP", True, TEXT_GOLD)
        self.screen.blit(hdr, (x + 22, y + 16))

        activations = telemetry.get("brain_activations", {})

        # Brain Panel Coordinates
        bx_origin = x + 25
        by_origin = y + 50
        bw = w - 50
        bh = 380

        # Background Brain Silhouette Oval
        pygame.draw.ellipse(self.screen, (16, 24, 40), (bx_origin, by_origin, bw, bh))
        pygame.draw.ellipse(self.screen, (28, 44, 72), (bx_origin, by_origin, bw, bh), width=2)

        # Draw Neural Tracts / Connections
        for node_id, data in BRAIN_NODES.items():
            n_pos = (bx_origin + int(data["pos"][0] * bw), by_origin + int(data["pos"][1] * bh))
            for target_id in data["tracts"]:
                if target_id in BRAIN_NODES:
                    t_pos = (
                        bx_origin + int(BRAIN_NODES[target_id]["pos"][0] * bw),
                        by_origin + int(BRAIN_NODES[target_id]["pos"][1] * bh),
                    )
                    act = max(activations.get(node_id, 0.2), activations.get(target_id, 0.2))
                    col = (
                        int(25 + act * 150),
                        int(45 + act * 170),
                        int(80 + act * 175),
                    )
                    pygame.draw.line(self.screen, col, n_pos, t_pos, width=max(1, int(act * 3)))

        # Draw Brain Nodes with Pulsating Energy Glow
        t_sec = time.time()
        for node_id, data in BRAIN_NODES.items():
            pos = (bx_origin + int(data["pos"][0] * bw), by_origin + int(data["pos"][1] * bh))
            energy = float(activations.get(node_id, 0.3))
            pulse = math.sin(t_sec * 4.0 + data["pos"][0] * 10) * 0.15
            radius = int(14 + (energy + pulse) * 12)

            # Outer glow
            glow_surf = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
            glow_col = (
                int(min(255, 30 + energy * 225)),
                int(min(255, 100 + energy * 155)),
                int(min(255, 200 - energy * 100)),
                int(min(255, 50 + energy * 90)),
            )
            pygame.draw.circle(glow_surf, glow_col, (radius * 2, radius * 2), radius)
            self.screen.blit(glow_surf, (pos[0] - radius * 2, pos[1] - radius * 2))

            # Core Node
            core_col = (
                int(min(255, 50 + energy * 205)),
                int(min(255, 150 + energy * 105)),
                int(min(255, 250 - energy * 80)),
            )
            pygame.draw.circle(self.screen, core_col, pos, 8)
            pygame.draw.circle(self.screen, TEXT_WHITE, pos, 8, width=1)

            # Node Label
            lbl = self.font_small.render(f"{data['name']}: {int(energy * 100)}%", True, TEXT_WHITE)
            self.screen.blit(lbl, (pos[0] - 30, pos[1] + 12))

        # Bottom Sub-Panel: Neurochemical Meters
        chem_y = y + 450
        pygame.draw.rect(self.screen, (12, 18, 28), (x + 20, chem_y, w - 40, 225), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x + 20, chem_y, w - 40, 225), width=1, border_radius=4)

        lbl_chem = self.font_title.render("NEUROTRANSMITTERS & HPA AXIS", True, TEXT_CYAN)
        self.screen.blit(lbl_chem, (x + 35, chem_y + 14))

        meters = [
            ("Dopamine (DA - Striatal Reward)", telemetry.get("dopamine", 0.5), (250, 204, 21)),
            ("Cortisol (Stress / Visual Ambiguity)", telemetry.get("cortisol", 0.2), (239, 68, 68)),
            ("Serotonin (5-HT - Cognitive Patience)", telemetry.get("serotonin", 0.6), (56, 189, 248)),
            ("Norepinephrine (NE - Visual Alertness)", telemetry.get("norepinephrine", 0.4), (168, 85, 247)),
            ("Sympathetic Tone (Fight-or-Flight)", telemetry.get("sympathetic_tone", 0.3), (249, 115, 22)),
            ("Vagal Tone (Parasympathetic Rest)", telemetry.get("vagal_tone", 0.7), (34, 197, 94)),
        ]

        for idx, (m_label, m_val, m_col) in enumerate(meters):
            m_y = chem_y + 44 + idx * 28
            lbl_m = self.font_small.render(m_label, True, TEXT_MUTED)
            self.screen.blit(lbl_m, (x + 35, m_y))

            # Progress Bar
            val_f = float(np.clip(m_val, 0.0, 1.0))
            bar_w = 180
            pygame.draw.rect(self.screen, (20, 30, 45), (x + 310, m_y + 2, bar_w, 10), border_radius=3)
            pygame.draw.rect(self.screen, m_col, (x + 310, m_y + 2, int(bar_w * val_f), 10), border_radius=3)

            pct_txt = self.font_small.render(f"{val_f:.2f}", True, TEXT_WHITE)
            self.screen.blit(pct_txt, (x + 500, m_y))

    def _render_decision_panel(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        sample: Dict[str, Any],
        pred: Dict[str, Any],
        benchmark: Optional[Dict[str, Any]],
    ) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x, y, w, h), width=1, border_radius=6)

        hdr = self.font_title.render("COGNITIVE DECISION & ECG", True, TEXT_GOLD)
        self.screen.blit(hdr, (x + 18, y + 16))

        pred_cls = pred.get("predicted_class", "None")
        conf = float(pred.get("confidence", 0.0))
        true_cls = sample.get("class", "None")
        is_correct = (pred_cls == true_cls)

        # Recognition Decision Card
        card_y = y + 46
        box_col = (16, 32, 24) if is_correct else (36, 18, 20)
        border_col = TEXT_GREEN if is_correct else TEXT_RED
        pygame.draw.rect(self.screen, box_col, (x + 18, card_y, w - 36, 95), border_radius=6)
        pygame.draw.rect(self.screen, border_col, (x + 18, card_y, w - 36, 95), width=2, border_radius=6)

        verdict = "MATCH [RECOGNIZED]" if is_correct else "MISMATCH"
        verdict_col = TEXT_GREEN if is_correct else TEXT_RED
        v_txt = self.font_body.render(verdict, True, verdict_col)
        self.screen.blit(v_txt, (x + 30, card_y + 10))

        pred_txt = self.font_large.render(f"{pred_cls}", True, TEXT_WHITE)
        self.screen.blit(pred_txt, (x + 30, card_y + 32))

        conf_txt = self.font_body.render(f"CONFIDENCE: {conf * 100:.1f}%  |  GROUND TRUTH: {true_cls}", True, TEXT_GOLD)
        self.screen.blit(conf_txt, (x + 30, card_y + 65))

        # Basal Ganglia Action Proposals Bar Chart
        bg_y = y + 155
        lbl_bg = self.font_small.render("BASAL GANGLIA CATEGORICAL PROPOSALS (D1/D2)", True, TEXT_CYAN)
        self.screen.blit(lbl_bg, (x + 18, bg_y))

        proposals = pred.get("proposals", {})
        classes_order = list(CLASSES)
        chart_y = bg_y + 20

        for idx, c_name in enumerate(classes_order):
            c_score = float(proposals.get(c_name, 0.0))
            c_y = chart_y + idx * 24
            is_win = (c_name == pred_cls)
            t_col = TEXT_GOLD if is_win else TEXT_MUTED

            c_lbl = self.font_small.render(f"{c_name:<11}", True, t_col)
            self.screen.blit(c_lbl, (x + 18, c_y))

            b_len = int(c_score * 200)
            bar_color = TEXT_GOLD if is_win else (56, 189, 248)
            pygame.draw.rect(self.screen, (20, 30, 45), (x + 115, c_y + 2, 200, 11), border_radius=2)
            pygame.draw.rect(self.screen, bar_color, (x + 115, c_y + 2, b_len, 11), border_radius=2)

            score_txt = self.font_small.render(f"{c_score:.2f}", True, TEXT_WHITE)
            self.screen.blit(score_txt, (x + 325, c_y))

        # Real-time ECG Oscilloscope
        ecg_y = y + 335
        telemetry = pred.get("telemetry", {})
        hr_val = telemetry.get("heart_rate", 70)
        lbl_ecg = self.font_title.render(f"AUTONOMIC ECG: {hr_val} BPM", True, TEXT_GREEN)
        self.screen.blit(lbl_ecg, (x + 18, ecg_y))

        ecg_w, ecg_h = w - 36, 115
        box_ecg_y = ecg_y + 26
        pygame.draw.rect(self.screen, ECG_BG, (x + 18, box_ecg_y, ecg_w, ecg_h), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x + 18, box_ecg_y, ecg_w, ecg_h), width=1, border_radius=4)

        # Oscilloscope grid
        for gy in range(box_ecg_y + 20, box_ecg_y + ecg_h, 25):
            pygame.draw.line(self.screen, (16, 28, 38), (x + 18, gy), (x + 18 + ecg_w, gy))

        # Draw waveform
        if len(self.ecg_points) > 1:
            step = ecg_w / float(len(self.ecg_points) - 1)
            mid_y = box_ecg_y + ecg_h // 2
            pts = []
            for i, val in enumerate(self.ecg_points):
                px = int(x + 18 + i * step)
                py = int(mid_y - val * 32.0)
                pts.append((px, py))
            pygame.draw.lines(self.screen, ECG_LINE, False, pts, width=2)

        # Benchmark Statistics Summary Card
        stat_y = y + 495
        pygame.draw.rect(self.screen, (12, 18, 28), (x + 18, stat_y, w - 36, 180), border_radius=4)
        pygame.draw.rect(self.screen, PANEL_BORDER, (x + 18, stat_y, w - 36, 180), width=1, border_radius=4)

        lbl_bench = self.font_title.render("EMPIRICAL BENCHMARK STATS", True, TEXT_CYAN)
        self.screen.blit(lbl_bench, (x + 30, stat_y + 12))

        if benchmark:
            tot_corr = benchmark.get("correct", 0)
            tot_cnt = benchmark.get("total", 30)
            acc = (tot_corr / tot_cnt * 100) if tot_cnt > 0 else 0.0
            acc_str = f"OVERALL ACCURACY: {tot_corr}/{tot_cnt} ({acc:.1f}%)"
            acc_txt = self.font_body.render(acc_str, True, TEXT_GOLD)
            self.screen.blit(acc_txt, (x + 30, stat_y + 38))

            by_var = benchmark.get("by_variation", {})
            v_idx = 0
            for v_name, (c_cnt, t_cnt) in by_var.items():
                v_y = stat_y + 64 + v_idx * 20
                v_acc = (c_cnt / t_cnt * 100) if t_cnt > 0 else 0.0
                line = f"{v_name.capitalize():<12}: {c_cnt}/{t_cnt} ({v_acc:.0f}%)"
                col = TEXT_GREEN if v_acc >= 80.0 else TEXT_MUTED
                self.screen.blit(self.font_small.render(line, True, col), (x + 30, v_y))
                v_idx += 1
        else:
            txt = self.font_body.render("Press [A] to run automated benchmark", True, TEXT_MUTED)
            self.screen.blit(txt, (x + 30, stat_y + 45))

    def _render_footer(self) -> None:
        foot_y = 780
        pygame.draw.rect(self.screen, PANEL_BG, (20, foot_y, 1320, 30), border_radius=3)
        pygame.draw.rect(self.screen, PANEL_BORDER, (20, foot_y, 1320, 30), width=1, border_radius=3)

        ctrl_text = (
            "[SPACE] Next Test Sample  |  [P] Previous Sample  |  "
            "[T] Re-Train 1-Shot  |  [A] Run Benchmark  |  [S] Save Checkpoint  |  [Q] Exit"
        )
        txt = self.font_body.render(ctrl_text, True, TEXT_WHITE)
        self.screen.blit(txt, (40, foot_y + 8))
