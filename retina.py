"""Ventral Visual Stream: Retina -> LGN -> Superior Colliculus -> V1 -> V4 -> IT Invariant 64-Dim Bus Mapping."""

import math
from typing import Tuple, Dict, Any, Union
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


def gabor_kernel(size: int = 7, theta: float = 0.0, sigma: float = 2.0, lambd: float = 4.0, gamma: float = 0.5) -> np.ndarray:
    """Generates a 2D Gabor wavelet filter matching V1 simple cell receptive fields."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]

    # In-plane rotation
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)

    gaussian = np.exp(-0.5 * (x_theta**2 + (gamma * y_theta)**2) / (sigma**2))
    sinusoid = np.cos(2.0 * math.pi * x_theta / lambd)
    kernel = gaussian * sinusoid
    return kernel.astype(np.float32)


class RetinalProcessor:
    """
    Biological visual ventral stream pipeline:
    1. Photoreceptor Ingestion: 64x64 luminance array.
    2. LGN Center-Surround: Difference of Gaussians (DoG) contrast filtering with thalamic noise suppression.
    3. Superior Colliculus Reflex: Saccadic foveation (mass-centroid centering for translation invariance).
    4. Size Constancy Normalization: Cortical scale invariance via radius of gyration matching.
    5. Primary Visual Cortex (V1): 4-orientation Gabor filter bank (0, 45, 90, 135 deg).
    6. Inferotemporal Cortex (IT): 64-dimensional invariant population code mapped to Optic Nerve (CN_II).
    """

    def __init__(self, target_size: Tuple[int, int] = (64, 64), use_preprocessing: bool = True):
        self.target_size = target_size
        self.use_preprocessing = use_preprocessing
        self.orientations = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
        self.gabor_filters = [gabor_kernel(size=7, theta=th) for th in self.orientations]

    def _difference_of_gaussians(self, img: np.ndarray) -> np.ndarray:
        """Simulates retinal ganglion & LGN center-surround contrast filtering with thalamic gating."""
        dog = gaussian_filter(img, sigma=1.0) - gaussian_filter(img, sigma=2.5)
        dog = np.clip(dog, 0.0, None)
        max_val = np.max(dog)
        if max_val > 1e-6:
            dog /= max_val
        # Thalamic TRN inhibitory gating to eliminate low-amplitude background noise
        dog[dog < 0.15] = 0.0
        max_val = np.max(dog)
        if max_val > 1e-6:
            dog /= max_val
        return dog.astype(np.float32)

    def _superior_colliculus_foveation(self, dog: np.ndarray) -> np.ndarray:
        """Saccadic foveation: shifts gaze centroid to center of visual field for translation invariance."""
        h, w = dog.shape
        mass = dog**2
        tot_mass = np.sum(mass)
        if tot_mass > 1e-4:
            y_coords, x_coords = np.indices((h, w))
            cy = np.sum(y_coords * mass) / tot_mass
            cx = np.sum(x_coords * mass) / tot_mass
            shift_y = int(round(h / 2.0 - cy))
            shift_x = int(round(w / 2.0 - cx))
            centered = np.roll(dog, shift_y, axis=0)
            centered = np.roll(centered, shift_x, axis=1)
            return centered.astype(np.float32)
        return dog

    def _size_constancy(self, centered: np.ndarray) -> np.ndarray:
        """Normalizes object scale via radius of gyration to canonical scale (16.0 px)."""
        h, w = centered.shape
        y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
        r_dist = np.sqrt(x**2 + y**2)
        sig_mask = centered > 0.15
        rg = float(np.mean(r_dist[sig_mask])) if np.any(sig_mask) else 16.0

        if rg > 4.0:
            target_rg = 16.0
            scale = float(np.clip(target_rg / rg, 0.65, 1.5))
            scaled = cv2.resize(centered, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            sh, sw = scaled.shape
            norm_img = np.zeros((h, w), dtype=np.float32)
            h_s = min(h, sh)
            w_s = min(w, sw)
            y1_src = (sh - h_s) // 2
            x1_src = (sw - w_s) // 2
            y1_dst = (h - h_s) // 2
            x1_dst = (w - w_s) // 2
            norm_img[y1_dst:y1_dst+h_s, x1_dst:x1_dst+w_s] = scaled[y1_src:y1_src+h_s, x1_src:x1_src+w_s]
            return norm_img.astype(np.float32)
        return centered

    def _v1_gabor_filter(self, lgn_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies V1 oriented simple-cell filters. Returns (energy_map, orientation_responses)."""
        responses = []
        for kernel in self.gabor_filters:
            resp = convolve2d(lgn_img, kernel, mode="same", boundary="symm")
            energy = np.abs(resp)
            responses.append(energy)

        responses_stack = np.stack(responses, axis=0)  # (4, 64, 64)
        v1_energy = np.mean(responses_stack, axis=0)
        max_val = np.max(v1_energy)
        if max_val > 1e-6:
            v1_energy /= max_val
        return v1_energy.astype(np.float32), responses_stack.astype(np.float32)

    def _it_multiscale_pooling(
        self, norm_dog: np.ndarray, v1_energy: np.ndarray, ori_stack: np.ndarray
    ) -> np.ndarray:
        """
        Compresses visual activity into a biologically invariant 64-dimensional IT population code:
        - 16 Concentric Radial Rings (Rotation Invariance)
        - 16 Orientation Distributions across concentric zones
        - 12 Radial Fourier Power Spectrum Harmonics
        - 8 Circular Angular Harmonics (Symmetry profile)
        - 8 Structural & Topographic Shape Moments
        - 4 Global Orientation Energy Shares
        """
        h, w = v1_energy.shape
        y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
        r_dist = np.sqrt(x**2 + y**2)
        max_r = math.sqrt((h//2)**2 + (w//2)**2)

        features = []

        # A. 16 Concentric Radial Rings
        ring_w = max_r / 16.0
        for r_i in range(16):
            mask = (r_dist >= r_i * ring_w) & (r_dist < (r_i + 1) * ring_w)
            val = np.mean(v1_energy[mask]) if np.any(mask) else 0.0
            features.append(float(val))

        # B. Orientation Distribution across Rings (4 orientations x 4 zones = 16 dims)
        zone_w = max_r / 4.0
        for z_i in range(4):
            mask_z = (r_dist >= z_i * zone_w) & (r_dist < (z_i + 1) * zone_w)
            for o_i in range(4):
                val = np.mean(ori_stack[o_i][mask_z]) if np.any(mask_z) else 0.0
                features.append(float(val))

        # C. Fourier Radial Power Spectrum (12 dims)
        fft = np.abs(np.fft.fftshift(np.fft.fft2(norm_dog)))
        fft_norm = fft / (np.sum(fft) + 1e-6)
        f_ring_w = max_r / 12.0
        for f_i in range(12):
            mask_f = (r_dist >= f_i * f_ring_w) & (r_dist < (f_i + 1) * f_ring_w)
            val_f = np.sum(fft_norm[mask_f]) if np.any(mask_f) else 0.0
            features.append(float(val_f))

        # D. Circular Harmonic Spectrum (8 dims: 0 to 7 harmonics)
        angles = np.arctan2(y, x) + math.pi
        sector_w = (2 * math.pi) / 16.0
        sector_energies = []
        for s_i in range(16):
            m_s = (angles >= s_i * sector_w) & (angles < (s_i + 1) * sector_w)
            sector_energies.append(float(np.mean(v1_energy[m_s]) if np.any(m_s) else 0.0))
        s_arr = np.array(sector_energies, dtype=np.float32)
        s_fft = np.abs(np.fft.fft(s_arr))
        s_fft_norm = s_fft / (np.sum(s_fft) + 1e-6)
        for k in range(8):
            features.append(float(s_fft_norm[k]))

        # E. Structural & Topographic Shape Moments (8 dims)
        top_energy = np.sum(v1_energy[:h//2, :])
        bot_energy = np.sum(v1_energy[h//2:, :])
        vert_asym = (top_energy - bot_energy) / (top_energy + bot_energy + 1e-6)

        left_energy = np.sum(v1_energy[:, :w//2])
        right_energy = np.sum(v1_energy[:, w//2:])
        horiz_asym = (left_energy - right_energy) / (left_energy + right_energy + 1e-6)

        tot_e = np.sum(v1_energy) + 1e-6
        mu20 = np.sum((x**2) * v1_energy) / tot_e
        mu02 = np.sum((y**2) * v1_energy) / tot_e
        aspect = (mu20 + 1e-6) / (mu02 + 1e-6)
        log_aspect = math.log(aspect + 1e-3)

        density = np.mean(v1_energy > 0.15)
        contrast = np.std(norm_dog)
        sig_mask = norm_dog > 0.15
        rg = float(np.mean(r_dist[sig_mask])) if np.any(sig_mask) else 16.0

        features.extend([
            float(vert_asym),
            float(horiz_asym),
            float(np.clip(log_aspect, -2.0, 2.0) / 2.0),
            float(density),
            float(contrast),
            float(np.mean(v1_energy)),
            float(np.max(v1_energy)),
            float(rg / (h / 2.0)),
        ])

        # F. 4 Global Orientation Shares (4 dims)
        tot_ori = np.sum(ori_stack) + 1e-6
        for o_i in range(4):
            features.append(float(np.sum(ori_stack[o_i]) / tot_ori))

        # Enforce exact 64 dimensions
        feat_arr = np.array(features[:64], dtype=np.float32)

        # Sparse IT Cortical normalization: zero mean & unit norm
        feat_arr = feat_arr - np.mean(feat_arr)
        feat_norm = feat_arr / (np.linalg.norm(feat_arr) + 1e-6)
        return feat_norm.astype(np.float32)

    def process_image(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        use_preprocessing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Processes an input image through the biological ventral visual stream.
        When use_preprocessing is False, operates in raw mode without algorithmic
        saccadic recentering or radius-of-gyration scale normalization.
        Returns visual telemetry and 64-dim CN_II optic vector.
        """
        preprocess = self.use_preprocessing if use_preprocessing is None else use_preprocessing

        if isinstance(image_input, str):
            img = Image.open(image_input).convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if np.issubdtype(arr.dtype, np.floating):
                if np.max(arr) <= 1.05:
                    arr = arr * 255.0
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            if arr.ndim == 3:
                # RGB to Grayscale
                img_gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
                img = Image.fromarray(img_gray.astype(np.uint8))
            else:
                img = Image.fromarray(arr)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # 1. Retina: Resize & Normalize to [0, 1]
        img_resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        raw_arr = np.asarray(img_resized, dtype=np.float32) / 255.0

        # 2. LGN: Center-Surround DoG Filter with noise gating
        lgn_arr = self._difference_of_gaussians(raw_arr)

        if preprocess:
            # 3. Superior Colliculus Reflex: Saccadic Foveation
            centered_lgn = self._superior_colliculus_foveation(lgn_arr)

            # 4. Size Constancy: Scale Normalization
            norm_lgn = self._size_constancy(centered_lgn)
        else:
            # Raw Ventral Stream mode: Direct pass without algorithmic centering/scaling
            norm_lgn = lgn_arr

        # 5. V1: Gabor Wavelet Edge Filter Bank
        v1_energy, ori_stack = self._v1_gabor_filter(norm_lgn)

        # 6. IT: Invariant 64-Dimensional Population Code
        it_vec = self._it_multiscale_pooling(norm_lgn, v1_energy, ori_stack)

        return {
            "raw": raw_arr,
            "lgn": lgn_arr,
            "foveated": norm_lgn,
            "v1": v1_energy,
            "it_vector": it_vec,
            "dominant_orientation": float(np.argmax(np.mean(ori_stack, axis=(1, 2))) * 45.0),
            "use_preprocessing": preprocess,
        }
