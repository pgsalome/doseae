"""
Clinical-plan-aware dose perturbation utilities.

This module builds on top of :class:`OpenTPSDoseAugmentor` to generate
clinically meaningful synthetic dose volumes by perturbing the beam geometry,
isocenter position, delivery fluence, and (optionally) robustness/objective
settings extracted from a clinical RTPLAN.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

from .opentps_generator import BeamConfig, OpenTPSDoseAugmentor

logger = logging.getLogger(__name__)


@dataclass
class BeamMetadata:
    gantry_angles: np.ndarray
    couch_angles: np.ndarray
    beam_names: List[str]
    isocenter_mm: np.ndarray
    energies_kev: List[float]
    cumulative_meterset: List[float]


class ClinicalPlanPerturbator:
    """
    Generate clinically-relevant synthetic dose volumes by perturbing a
    baseline RT plan in combination with the OpenTPS optimiser.
    """

    def __init__(self, augmentation_cfg: Dict[str, Any], project_root: Optional[Path] = None):
        self.config = dict(augmentation_cfg or {})
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
        self.operations_cfg = (
            self.config.get("clinical_perturbation", {}).get("operations", {}) or {}
        )
        perturb_cfg = self.config.get("clinical_perturbation", {}) or {}
        self.target_threshold_range = perturb_cfg.get("target_threshold_range", [0.78, 0.92])
        self.random_seed = perturb_cfg.get("random_seed", self.config.get("random_seed"))
        self.rng = np.random.default_rng(self.random_seed)

        # Instantiate the base OpenTPS augmentor for beam delivery/dose computation.
        self.augmentor = OpenTPSDoseAugmentor(self.config, project_root=self.project_root)
        if not self.augmentor.is_available():
            raise RuntimeError("OpenTPS augmentor unavailable; cannot perform clinical perturbations.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_from_paths(
        self,
        *,
        patient_id: str,
        ct_path: Path,
        dose_path: Path,
        rtplan_path: Path,
        n_samples: int,
        output_dir: Optional[Path] = None,
        retain_reference: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convenience helper that loads imaging data from disk, generates
        synthetic variants, and optionally writes them to ``output_dir``.
        """
        ct_path = Path(ct_path).expanduser().resolve()
        dose_path = Path(dose_path).expanduser().resolve()
        rtplan_path = Path(rtplan_path).expanduser().resolve()
        if output_dir:
            output_dir = Path(output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)

        ct_image = sitk.ReadImage(str(ct_path))
        dose_image = sitk.ReadImage(str(dose_path))
        dose_image = self._resample_dose_to_ct(ct_image, dose_image)
        norm_dose, scale_info = self._normalize_dose(dose_image)

        variants = self.generate_variants(
            patient_id=patient_id,
            ct_image_hu=ct_image,
            normalized_dose_image=norm_dose,
            rtplan_path=rtplan_path,
            base_prescription=float(scale_info.get("scale", 1.0)),
            n_samples=n_samples,
        )

        if retain_reference and output_dir:
            reference_dir = output_dir / f"{patient_id}_reference"
            reference_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(ct_image, str(reference_dir / f"{patient_id}_ct.nrrd"))
            sitk.WriteImage(norm_dose, str(reference_dir / f"{patient_id}_dose_normalized.nrrd"))
            with (reference_dir / f"{patient_id}_norm_scale.json").open("w") as handle:
                json.dump(scale_info, handle, indent=2)

        if output_dir:
            for item in variants:
                variant_dir = output_dir / item["patient_id"]
                variant_dir.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(
                    item["dose_image"],
                    str(variant_dir / f"{item['patient_id']}_dose.nrrd"),
                )
                meta = item.get("metadata", {})
                meta_path = variant_dir / f"{item['patient_id']}_metadata.json"
                with meta_path.open("w") as handle:
                    json.dump(meta, handle, indent=2)

        return variants

    def generate_variants(
        self,
        *,
        patient_id: str,
        ct_image_hu: sitk.Image,
        normalized_dose_image: sitk.Image,
        rtplan_path: Path,
        base_prescription: float,
        n_samples: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic dose volumes by perturbing the RT plan defined in
        ``rtplan_path``.
        """
        if n_samples <= 0:
            return []

        plan_metadata = self._extract_plan_metadata(rtplan_path)
        norm_dose_array = sitk.GetArrayFromImage(normalized_dose_image).astype(np.float32)
        target_mask = self.augmentor._derive_target_mask(norm_dose_array)
        if target_mask is None:
            raise RuntimeError("Unable to derive optimisation target mask from dose volume.")

        ct_array_hu = sitk.GetArrayFromImage(ct_image_hu).astype(np.float32)
        spacing = ct_image_hu.GetSpacing()
        origin = ct_image_hu.GetOrigin()
        direction = ct_image_hu.GetDirection()

        results: List[Dict[str, Any]] = []
        for sample_idx in range(n_samples):
            params = self._sample_parameters(plan_metadata)
            beam_config = BeamConfig(
                gantry_angles=params["gantry_angles"],
                couch_angles=params["couch_angles"],
                beam_names=plan_metadata.beam_names,
                prescription_gy=base_prescription,
                target_threshold=params["target_threshold"],
                random_seed=params["rng_seed"],
            )

            iso_override = plan_metadata.isocenter_mm + params["isocenter_shift_mm"]
            dose_image = self.augmentor._compute_opentps_plan(
                patient_id=patient_id,
                synthetic_idx=sample_idx,
                ct_array_hu=ct_array_hu,
                target_mask=target_mask,
                spacing=spacing,
                origin=origin,
                direction=direction,
                beam_config=beam_config,
                isocenter_override=tuple(float(v) for v in iso_override),
                prescription_scale=params["mu_scale"],
                objective_variation=params.get("objective_variation"),
                robustness=params.get("robustness"),
            )

            variant_id = f"{patient_id}_variant_{sample_idx + 1:02d}"
            metadata = {
                "base_plan": {
                    "beam_names": plan_metadata.beam_names,
                    "gantry_angles_deg": plan_metadata.gantry_angles.tolist(),
                    "couch_angles_deg": plan_metadata.couch_angles.tolist(),
                    "isocenter_mm": plan_metadata.isocenter_mm.tolist(),
                    "energies_mev": plan_metadata.energies_kev,
                    "cumulative_meterset": plan_metadata.cumulative_meterset,
                },
                "perturbation": {
                    "gantry_jitter_pct": params["gantry_jitter_pct"].tolist(),
                    "couch_jitter_pct": params["couch_jitter_pct"].tolist(),
                    "isocenter_shift_mm": params["isocenter_shift_mm"].tolist(),
                    "mu_scale": params["mu_scale"],
                    "target_threshold": params["target_threshold"],
                    "objective_variation": params.get("objective_variation"),
                    "robustness": params.get("robustness"),
                    "rng_seed": params["rng_seed"],
                },
            }

            results.append(
                {
                    "patient_id": variant_id,
                    "dose_image": dose_image,
                    "metadata": metadata,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_plan_metadata(self, rtplan_path: Path) -> BeamMetadata:
        ds = pydicom.dcmread(str(rtplan_path))
        gantry_angles: List[float] = []
        couch_angles: List[float] = []
        beam_names: List[str] = []
        energies: List[float] = []
        meterset_weights: List[float] = []
        isocenters: List[np.ndarray] = []

        for beam_index, beam in enumerate(ds.BeamSequence):
            cp0 = beam.ControlPointSequence[0]
            gantry_angles.append(float(getattr(cp0, "GantryAngle", 0.0)))
            couch_angles.append(float(getattr(cp0, "PatientSupportAngle", 0.0)))
            beam_names.append(getattr(beam, "BeamName", f"B{beam_index + 1}"))
            energies.append(float(getattr(cp0, "NominalBeamEnergy", 0.0)))
            meterset_weights.append(
                float(getattr(beam.ControlPointSequence[-1], "CumulativeMetersetWeight", 1.0))
            )
            iso = np.array(getattr(cp0, "IsocenterPosition", [0.0, 0.0, 0.0]), dtype=np.float32)
            isocenters.append(iso)

        if not isocenters:
            raise RuntimeError("RTPLAN does not contain any beam definitions.")
        iso_mean = np.mean(np.stack(isocenters, axis=0), axis=0)

        return BeamMetadata(
            gantry_angles=np.asarray(gantry_angles, dtype=np.float32),
            couch_angles=np.asarray(couch_angles, dtype=np.float32),
            beam_names=beam_names,
            isocenter_mm=iso_mean.astype(np.float32),
            energies_kev=energies,
            cumulative_meterset=meterset_weights,
        )

    def _sample_parameters(self, metadata: BeamMetadata) -> Dict[str, Any]:
        n_beams = len(metadata.gantry_angles)
        operations = self.operations_cfg

        gantry_jitter_cfg = operations.get("beam_jitter", {})
        gantry_jitter_pct = self._sample_beta_array(
            size=n_beams,
            cfg=gantry_jitter_cfg,
            default_max=0.05,
        )
        couch_jitter_cfg = gantry_jitter_cfg.get("couch", operations.get("couch_jitter", gantry_jitter_cfg))
        couch_jitter_pct = self._sample_beta_array(
            size=n_beams,
            cfg=couch_jitter_cfg,
            default_max=0.03,
        )

        mu_cfg = operations.get("mu_scaling", {})
        mu_scale = self._sample_beta_scalar(mu_cfg, default_max=0.05)

        iso_cfg = operations.get("isocenter_shift", {})
        iso_shift = self._sample_isocenter_shift(iso_cfg)

        objective_cfg = operations.get("objective_variation", {})
        objective_variation = self._sample_objective_variation(objective_cfg)

        robustness_cfg = operations.get("robustness", {})
        robustness = self._sample_robustness(robustness_cfg)

        target_threshold = self._sample_uniform(self.target_threshold_range, default=0.85)
        rng_seed = int(self.rng.integers(0, 2**31 - 1))

        gantry_angles = self._apply_jitter(metadata.gantry_angles, gantry_jitter_pct)
        couch_angles = self._apply_jitter(metadata.couch_angles, couch_jitter_pct)

        return {
            "gantry_angles": gantry_angles,
            "couch_angles": couch_angles,
            "gantry_jitter_pct": gantry_jitter_pct,
            "couch_jitter_pct": couch_jitter_pct,
            "isocenter_shift_mm": iso_shift,
            "mu_scale": mu_scale,
            "target_threshold": target_threshold,
            "objective_variation": objective_variation,
            "robustness": robustness,
            "rng_seed": rng_seed,
        }

    def _apply_jitter(self, base_angles: np.ndarray, jitter_pct: np.ndarray) -> List[float]:
        if jitter_pct is None or not jitter_pct.any():
            return base_angles.tolist()
        jittered = base_angles * (1.0 + jitter_pct)
        jittered = np.mod(jittered, 360.0)
        return jittered.tolist()

    def _sample_beta_array(self, size: int, cfg: Dict[str, Any], default_max: float) -> np.ndarray:
        if not cfg.get("enabled", True):
            return np.zeros(size, dtype=np.float32)
        alpha = float(cfg.get("beta_alpha", 2.0))
        beta = float(cfg.get("beta_beta", 2.0))
        max_pct = float(cfg.get("max_pct", default_max))
        draws = self.rng.beta(alpha, beta, size=size)
        jitter = (draws - 0.5) * 2.0 * max_pct
        return jitter.astype(np.float32)

    def _sample_beta_scalar(self, cfg: Dict[str, Any], default_max: float) -> float:
        if not cfg.get("enabled", True):
            return 1.0
        alpha = float(cfg.get("beta_alpha", 2.5))
        beta = float(cfg.get("beta_beta", 2.5))
        max_pct = float(cfg.get("max_pct", default_max))
        draw = self.rng.beta(alpha, beta)
        delta = (draw - 0.5) * 2.0 * max_pct
        return float(max(0.01, 1.0 + delta))

    def _sample_isocenter_shift(self, cfg: Dict[str, Any]) -> np.ndarray:
        if not cfg.get("enabled", True):
            return np.zeros(3, dtype=np.float32)
        sigma = np.asarray(cfg.get("sigma_mm", [1.0, 1.0, 1.0]), dtype=np.float32)
        trunc = float(cfg.get("max_mm", 5.0))
        shift = self.rng.normal(0.0, sigma, size=3)
        shift = np.clip(shift, -trunc, trunc)
        return shift.astype(np.float32)

    def _sample_objective_variation(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not cfg.get("enabled", False):
            return None
        dmin_margin = self._sample_uniform(cfg.get("dmin_margin_pct", [0.02, 0.05]), default=0.03)
        dmax_margin = self._sample_uniform(cfg.get("dmax_margin_pct", [0.04, 0.08]), default=0.05)
        weight_range = cfg.get("weight_range", [0.8, 1.2])
        dmin_weight = self._sample_uniform(weight_range, default=1.0)
        dmax_weight = self._sample_uniform(weight_range, default=1.0)
        return {
            "dmin_margin_pct": dmin_margin,
            "dmax_margin_pct": dmax_margin,
            "dmin_weight": dmin_weight,
            "dmax_weight": dmax_weight,
        }

    def _sample_robustness(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not cfg.get("enabled", False):
            return None
        return {
            "enabled": True,
            "setup_systematic_mm": cfg.get("setup_systematic_mm", [2.0, 2.0, 2.0]),
            "setup_random_mm": cfg.get("setup_random_mm", [1.0, 1.0, 1.0]),
            "num_scenarios": cfg.get("num_scenarios", 6),
            "strategy": cfg.get("strategy", "REDUCED_SET"),
        }

    def _sample_uniform(self, value_range: Any, default: float) -> float:
        if isinstance(value_range, (list, tuple)) and len(value_range) == 2:
            low, high = float(value_range[0]), float(value_range[1])
            if low == high:
                return low
            return float(self.rng.uniform(low, high))
        return float(default)

    @staticmethod
    def _resample_dose_to_ct(ct_image: sitk.Image, dose_image: sitk.Image) -> sitk.Image:
        ct_spacing = ct_image.GetSpacing()
        dose_spacing = dose_image.GetSpacing()
        ct_origin = ct_image.GetOrigin()
        dose_origin = dose_image.GetOrigin()
        ct_direction = ct_image.GetDirection()
        dose_direction = dose_image.GetDirection()
        ct_size = ct_image.GetSize()
        dose_size = dose_image.GetSize()

        tolerance = 1e-6
        spacing_match = all(abs(ct_spacing[i] - dose_spacing[i]) < tolerance for i in range(3))
        origin_match = all(abs(ct_origin[i] - dose_origin[i]) < tolerance for i in range(3))
        direction_match = all(abs(ct_direction[i] - dose_direction[i]) < tolerance for i in range(9))
        size_match = ct_size == dose_size

        if spacing_match and origin_match and direction_match and size_match:
            return dose_image

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(ct_spacing)
        resampler.SetOutputOrigin(ct_origin)
        resampler.SetOutputDirection(ct_direction)
        resampler.SetSize(ct_size)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0.0)
        return resampler.Execute(dose_image)

    @staticmethod
    def _normalize_dose(
        dose_image: sitk.Image,
        prescribed_dose: Optional[float] = None,
        percentile: float = 99.0,
    ) -> Tuple[sitk.Image, Dict[str, Any]]:
        array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
        if prescribed_dose and prescribed_dose > 0:
            scale = float(prescribed_dose)
            normalized = array / scale
            method = "prescribed"
        else:
            pxx = np.percentile(array, percentile)
            scale = float(pxx if pxx > 0 else 1.0)
            normalized = array / scale
            method = f"p{percentile:g}"
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
        norm_image = sitk.GetImageFromArray(normalized)
        norm_image.CopyInformation(dose_image)
        return norm_image, {"method": method, "scale": scale}
