"""
Utilities for generating synthetic dose distributions with OpenTPS.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

try:
    import importlib
except ImportError:  # pragma: no cover - fallback for very old Python
    import importlib as importlib

logger = logging.getLogger(__name__)


@dataclass
class BeamConfig:
    gantry_angles: List[float]
    couch_angles: List[float]
    beam_names: List[str]
    prescription_gy: float
    target_threshold: float
    random_seed: int


class OpenTPSDoseAugmentor:
    """
    Generate synthetic dose plans with OpenTPS for data augmentation.
    """

    def __init__(self, config: Dict[str, Any], project_root: Optional[Path] = None):
        self.config = dict(config or {})
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
        self.enabled = bool(self.config.get("enable_synthetic_dose", False))
        synthetic_count = int(self.config.get("synthetic_doses_per_patient", 0) or 0)
        self.synthetic_count = max(0, synthetic_count)
        self.random_seed = self.config.get("random_seed")
        self.rng = np.random.default_rng(self.random_seed)

        opentps_cfg = dict(self.config.get("opentps", {}))
        self.min_beams = max(1, int(opentps_cfg.get("min_beams", 3)))
        self.max_beams = max(self.min_beams, int(opentps_cfg.get("max_beams", max(4, self.min_beams))))
        gantry_range = opentps_cfg.get("gantry_range_deg", [0.0, 360.0])
        self.gantry_range = (float(gantry_range[0]), float(gantry_range[1])) if len(gantry_range) >= 2 else (0.0, 360.0)
        couch_range = opentps_cfg.get("couch_range_deg", [-5.0, 5.0])
        self.couch_range = (float(couch_range[0]), float(couch_range[1])) if len(couch_range) >= 2 else (-5.0, 5.0)
        self.beamlet_spacing = float(opentps_cfg.get("beamlet_spacing_mm", 5.0))
        self.target_margin = float(opentps_cfg.get("target_margin_mm", 5.0))
        self.batch_size = max(1, int(opentps_cfg.get("ccc_batch_size", 24)))
        self.optimizer_maxiter = max(1, int(opentps_cfg.get("optimizer_max_iterations", 150)))
        self.prescription_gy = opentps_cfg.get("prescription_gy")
        scale_range = opentps_cfg.get("prescription_scale_range", [0.9, 1.1])
        self.prescription_scale_range = (
            float(scale_range[0]),
            float(scale_range[1] if len(scale_range) > 1 else scale_range[0]),
        )
        threshold_range = opentps_cfg.get("target_isodose_range", [0.75, 0.95])
        self.target_threshold_range = (
            float(threshold_range[0]),
            float(threshold_range[1] if len(threshold_range) > 1 else threshold_range[0]),
        )
        self.min_target_voxels = int(opentps_cfg.get("min_target_voxels", 750))
        self.mask_dilation_mm = float(opentps_cfg.get("target_mask_dilation_mm", 2.0))
        core_path_cfg = opentps_cfg.get("core_path")
        default_core = self.project_root / "external" / "OpenTPS" / "opentps_core"
        self.core_path = Path(core_path_cfg).resolve() if core_path_cfg else default_core.resolve()
        self.custom_workspace = opentps_cfg.get("workspace_dir")

        self.available = False
        self._ct_calibration = None
        self._bootstrap_failure_reason: Optional[str] = None

        if self.enabled and self.synthetic_count > 0:
            self._bootstrap_opentps()
        else:
            if not self.enabled:
                logger.info("OpenTPS dose augmentation disabled in configuration.")
            elif self.synthetic_count == 0:
                logger.info("OpenTPS dose augmentation requested but synthetic_doses_per_patient=0.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self.available

    def generate(
        self,
        *,
        patient_id: str,
        ct_image_hu: sitk.Image,
        normalized_dose_image: sitk.Image,
        normalization_fn: Callable[[sitk.Image, Optional[float]], Tuple[sitk.Image, Dict[str, Any]]],
        base_prescription: Optional[float],
    ) -> List[Dict[str, Any]]:
        """
        Generate additional dose volumes using OpenTPS.

        Returns:
            List of dictionaries with keys:
                - patient_id: synthetic identifier
                - dose_image: normalized SimpleITK image (0-1 scale)
                - scale_info: normalization metadata
                - metadata: diagnostic information about augmentation
        """
        if not self.available:
            if self._bootstrap_failure_reason:
                logger.warning("OpenTPS augmentation unavailable: %s", self._bootstrap_failure_reason)
            return []

        if self.synthetic_count <= 0:
            return []

        ct_hu_array = sitk.GetArrayFromImage(ct_image_hu).astype(np.float32)
        norm_dose_array = sitk.GetArrayFromImage(normalized_dose_image).astype(np.float32)

        target_mask = self._derive_target_mask(norm_dose_array)
        if target_mask is None:
            logger.warning(
                "Patient %s: unable to derive target mask from normalized dose distribution; skipping OpenTPS augmentation.",
                patient_id,
            )
            return []

        ct_spacing = ct_image_hu.GetSpacing()
        ct_origin = ct_image_hu.GetOrigin()
        ct_direction = ct_image_hu.GetDirection()

        patient_results: List[Dict[str, Any]] = []
        for synthetic_idx in range(self.synthetic_count):
            seed = int(self.rng.integers(0, 2**31 - 1))
            beam_config = self._sample_beam_config(seed, base_prescription)
            try:
                synthetic_dose = self._compute_opentps_plan(
                    patient_id=patient_id,
                    synthetic_idx=synthetic_idx,
                    ct_array_hu=ct_hu_array,
                    target_mask=target_mask,
                    spacing=ct_spacing,
                    origin=ct_origin,
                    direction=ct_direction,
                    beam_config=beam_config,
                )
                if synthetic_dose is None:
                    continue
                normalized_synthetic_dose, scale_info = normalization_fn(synthetic_dose, None)
                synthetic_id = f"{patient_id}_synthetic_{synthetic_idx + 1:02d}"
                patient_results.append(
                    {
                        "patient_id": synthetic_id,
                        "dose_image": normalized_synthetic_dose,
                        "scale_info": scale_info,
                        "metadata": {
                            "beam_config": {
                                "gantry_angles_deg": beam_config.gantry_angles,
                                "couch_angles_deg": beam_config.couch_angles,
                                "beam_names": beam_config.beam_names,
                                "prescription_gy": beam_config.prescription_gy,
                                "target_threshold": beam_config.target_threshold,
                            },
                            "rng_seed": beam_config.random_seed,
                        },
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.exception(
                    "OpenTPS augmentation failed for patient %s (synthetic idx %d): %s",
                    patient_id,
                    synthetic_idx,
                    exc,
                )
                continue

        return patient_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bootstrap_opentps(self) -> None:
        if not self.core_path.exists():
            self._bootstrap_failure_reason = f"OpenTPS core path not found: {self.core_path}"
            logger.error(self._bootstrap_failure_reason)
            return

        if str(self.core_path) not in sys.path:
            sys.path.insert(0, str(self.core_path))

        try:
            self.Patient = importlib.import_module("opentps.core.data").Patient
            images_mod = importlib.import_module("opentps.core.data.images")
            self.CTImage = images_mod.CTImage
            self.ROIMask = images_mod.ROIMask
            plan_mod = importlib.import_module("opentps.core.data.plan")
            self.PhotonPlanDesign = plan_mod.PhotonPlanDesign
            self.FidObjective = plan_mod.FidObjective
            dose_mod = importlib.import_module("opentps.core.processing.doseCalculation.photons.cccDoseCalculator")
            self.CCCDoseCalculator = dose_mod.CCCDoseCalculator
            opti_mod = importlib.import_module("opentps.core.processing.planOptimization.planOptimization")
            self.IntensityModulationOptimizer = opti_mod.IntensityModulationOptimizer
            io_mod = importlib.import_module("opentps.core.io.scannerReader")
            self.readScanner = io_mod.readScanner
            config_mod = importlib.import_module("opentps.core.processing.doseCalculation.doseCalculationConfig")
            self.DoseCalculationConfig = config_mod.DoseCalculationConfig
        except ImportError as exc:  # pragma: no cover - depends on environment
            self._bootstrap_failure_reason = f"Failed to import OpenTPS modules: {exc}"
            logger.error(self._bootstrap_failure_reason)
            return

        if self.custom_workspace:
            workspace = Path(self.custom_workspace).expanduser().resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            import os

            os.environ.setdefault("OpenTPSWorkspace", str(workspace))

        self.available = True
        logger.info("OpenTPS dose augmentation ready (core path: %s).", self.core_path)

    def _get_ct_calibration(self):
        if self._ct_calibration is None:
            cfg = self.DoseCalculationConfig()
            self._ct_calibration = self.readScanner(cfg.scannerFolder)
        return self._ct_calibration

    def _sample_beam_config(self, seed: int, base_prescription: Optional[float]) -> BeamConfig:
        local_rng = np.random.default_rng(seed)
        beam_count = int(local_rng.integers(self.min_beams, self.max_beams + 1))
        gantry_angles = local_rng.uniform(self.gantry_range[0], self.gantry_range[1], beam_count)
        gantry_angles = np.mod(gantry_angles, 360.0).tolist()
        couch_angles = local_rng.uniform(self.couch_range[0], self.couch_range[1], beam_count).tolist()
        beam_names = [f"B{i+1}" for i in range(beam_count)]

        base_prescription = float(base_prescription) if base_prescription and base_prescription > 0 else (
            float(self.prescription_gy) if self.prescription_gy else 66.0
        )
        scale = local_rng.uniform(self.prescription_scale_range[0], self.prescription_scale_range[1])
        prescription = float(max(0.1, base_prescription * scale))

        threshold = float(
            local_rng.uniform(self.target_threshold_range[0], self.target_threshold_range[1])
        )

        return BeamConfig(
            gantry_angles=list(gantry_angles),
            couch_angles=list(couch_angles),
            beam_names=beam_names,
            prescription_gy=prescription,
            target_threshold=threshold,
            random_seed=seed,
        )

    def _derive_target_mask(self, norm_dose_array: np.ndarray) -> Optional[np.ndarray]:
        threshold_values = np.linspace(
            self.target_threshold_range[1],
            self.target_threshold_range[0],
            num=5,
        )
        structure = ndimage.generate_binary_structure(3, 2)
        for threshold in threshold_values:
            mask = norm_dose_array >= float(threshold)
            if not np.any(mask):
                continue
            mask = ndimage.binary_closing(mask, structure=structure, iterations=1)
            mask = ndimage.binary_fill_holes(mask)
            mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
            labeled, n_labels = ndimage.label(mask, structure=structure)
            if n_labels == 0:
                continue
            counts = ndimage.sum(mask, labeled, index=range(1, n_labels + 1))
            max_idx = int(np.argmax(counts)) + 1
            largest = labeled == max_idx
            voxel_count = int(counts[max_idx - 1])
            if voxel_count < self.min_target_voxels:
                continue
            if self.mask_dilation_mm > 0:
                iterations = max(1, int(round(self.mask_dilation_mm / 2.0)))
                largest = ndimage.binary_dilation(largest, structure=structure, iterations=iterations)
            return largest.astype(bool)
        return None

    def _compute_opentps_plan(
        self,
        *,
        patient_id: str,
        synthetic_idx: int,
        ct_array_hu: np.ndarray,
        target_mask: np.ndarray,
        spacing: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        direction: Tuple[float, ...],
        beam_config: BeamConfig,
        isocenter_override: Optional[Tuple[float, float, float]] = None,
        prescription_scale: float = 1.0,
        objective_variation: Optional[Dict[str, Any]] = None,
        robustness: Optional[Dict[str, Any]] = None,
    ) -> Optional[sitk.Image]:
        patient = self.Patient()
        patient.name = f"{patient_id}_{synthetic_idx:02d}"

        ct_array_xyz = np.transpose(ct_array_hu, axes=(2, 1, 0))
        ct_image = self.CTImage(
            imageArray=ct_array_xyz.copy(),
            spacing=tuple(spacing),
            origin=tuple(origin),
            patient=patient,
            name=f"{patient_id}_ct",
        )

        mask_xyz = np.transpose(target_mask.astype(bool), axes=(2, 1, 0))
        roi = self.ROIMask(
            imageArray=mask_xyz,
            spacing=tuple(spacing),
            origin=tuple(origin),
            patient=patient,
            name="synthetic_target",
        )

        plan_design = self.PhotonPlanDesign()
        plan_design.ct = ct_image
        plan_design.targetMask = roi
        plan_design.calibration = self._get_ct_calibration()
        plan_design.gantryAngles = beam_config.gantry_angles
        plan_design.couchAngles = beam_config.couch_angles
        plan_design.beamNames = beam_config.beam_names
        plan_design.xBeamletSpacing_mm = self.beamlet_spacing
        plan_design.yBeamletSpacing_mm = self.beamlet_spacing
        plan_design.targetMargin = self.target_margin
        if isocenter_override is not None:
            plan_design.isocenterPosition_mm = tuple(float(v) for v in isocenter_override)
        effective_prescription = float(
            max(0.1, beam_config.prescription_gy * max(prescription_scale, 1e-3))
        )
        plan_design.defineTargetMaskAndPrescription(target=roi, targetPrescription=effective_prescription)
        plan_design.ROI_cropping = False

        plan = plan_design.buildPlan()

        dose_calculator = self.CCCDoseCalculator(batchSize=self.batch_size)
        dose_calculator.ctCalibration = self._get_ct_calibration()
        beamlets = dose_calculator.computeBeamlets(ct_image, plan)
        plan.planDesign.beamlets = beamlets

        fid_objectives = plan.planDesign.objectives
        try:
            fid_objectives.fidObjList.clear()
        except AttributeError:
            plan.planDesign.objectives.fidObjList = []
        if objective_variation:
            dmin_margin = float(objective_variation.get('dmin_margin_pct', 0.03))
            dmax_margin = float(objective_variation.get('dmax_margin_pct', 0.05))
            dmin_weight = float(objective_variation.get('dmin_weight', 1.0))
            dmax_weight = float(objective_variation.get('dmax_weight', 1.0))
        else:
            dmin_margin = 0.03
            dmax_margin = 0.05
            dmin_weight = 1.0
            dmax_weight = 1.0
        lower_bound = effective_prescription * max(0.0, 1.0 - dmin_margin)
        upper_bound = effective_prescription * (1.0 + dmax_margin)
        fid_objectives.addFidObjective(roi, self.FidObjective.Metrics.DMIN, lower_bound, dmin_weight)
        fid_objectives.addFidObjective(roi, self.FidObjective.Metrics.DMAX, upper_bound, dmax_weight)

        if robustness and robustness.get('enabled'):
            rob_cfg = robustness
            rob_settings = plan.planDesign.robustness
            strategy = str(rob_cfg.get('strategy', 'REDUCED_SET')).upper()
            try:
                rob_settings.selectionStrategy = rob_settings.Strategies[strategy]
            except KeyError:
                rob_settings.selectionStrategy = rob_settings.Strategies.REDUCED_SET
            setup_sys = rob_cfg.get('setup_systematic_mm')
            if setup_sys is not None:
                rob_settings.setupSystematicError = [float(v) for v in setup_sys]
            setup_rand = rob_cfg.get('setup_random_mm')
            if setup_rand is not None:
                rob_settings.setupRandomError = [float(v) for v in setup_rand]
            if 'num_scenarios' in rob_cfg:
                rob_settings.numScenarios = int(rob_cfg['num_scenarios'])

        optimizer = self.IntensityModulationOptimizer(
            method="Scipy_L-BFGS-B",
            plan=plan,
            maxiter=self.optimizer_maxiter,
        )
        dose_image, _ = optimizer.optimize()

        dose_array_xyz = np.asarray(dose_image.imageArray, dtype=np.float32)
        dose_array_zyx = np.transpose(dose_array_xyz, axes=(2, 1, 0))
        synthetic_dose = sitk.GetImageFromArray(dose_array_zyx)
        synthetic_dose.SetSpacing(spacing)
        synthetic_dose.SetOrigin(origin)
        synthetic_dose.SetDirection(direction)
        return synthetic_dose
