"""
Heat Stress Index computation module.

Computes a Heat Stress Index (HSI) based on Land Surface Temperature
relative to the regional mean and historical baseline.

HSI = (LST - LST_mean) / LST_std

Interpretation:
  HSI > 2.0:  Extreme heat stress
  HSI 1-2:    High heat stress
  HSI 0-1:    Moderate heat stress
  HSI < 0:    Below average temperature
"""
import ee
from loguru import logger
from typing import Dict, Optional


class HeatStressProcessor:
    """Compute Heat Stress Index from Land Surface Temperature."""

    def __init__(self):
        """Initialize the Heat Stress processor."""
        logger.debug("Initialized HeatStressProcessor.")

    def compute_hsi(self, lst_image: ee.Image, aoi: ee.Geometry, scale: int = 30) -> ee.Image:
        """
        Compute Heat Stress Index as z-score of LST.

        Formula: HSI = (LST - LST_mean) / LST_std

        Args:
            lst_image: ee.Image with LST band (in Celsius).
            aoi: Area of interest for computing regional stats.
            scale: Scale for computing mean and std.

        Returns:
            ee.Image with 'heat_stress_index' band.
        """
        try:
            logger.info("Computing Heat Stress Index.")
            stats = lst_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=aoi,
                scale=scale,
                maxPixels=1e13,
                bestEffort=True
            )
            
            lst_mean = ee.Number(stats.get('LST_mean'))
            lst_std = ee.Number(stats.get('LST_stdDev'))
            
            # Avoid division by zero
            lst_std_safe = ee.Number(ee.Algorithms.If(lst_std.eq(0), 1, lst_std))
            
            hsi = lst_image.select('LST').subtract(lst_mean).divide(lst_std_safe).rename('heat_stress_index')
            return hsi
        except ee.EEException as e:
            logger.error(f"Failed to compute Heat Stress Index: {e}")
            raise

    def compute_hsi_from_baseline(self, lst_current: ee.Image, lst_baseline: ee.Image,
                                   aoi: ee.Geometry, scale: int = 30) -> ee.Image:
        """
        Compute Heat Stress Index relative to a historical baseline.

        Formula: HSI = (LST_current - LST_baseline_mean) / LST_baseline_std

        Args:
            lst_current: Current period LST image.
            lst_baseline: Historical baseline LST image (e.g., multi-year average).
            aoi: Area of interest.
            scale: Scale for computing statistics.

        Returns:
            ee.Image with 'heat_stress_index' band.
        """
        try:
            logger.info("Computing HSI relative to historical baseline.")
            baseline_stats = lst_baseline.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=aoi, scale=scale, maxPixels=1e13, bestEffort=True
            )
            
            baseline_mean = ee.Number(baseline_stats.get('LST_mean'))
            baseline_std = ee.Number(baseline_stats.get('LST_stdDev'))
            baseline_std_safe = ee.Number(ee.Algorithms.If(baseline_std.eq(0), 1, baseline_std))
            
            hsi = lst_current.select('LST').subtract(baseline_mean).divide(baseline_std_safe).rename('heat_stress_index')
            return hsi
        except ee.EEException as e:
            logger.error(f"Failed to compute baseline HSI: {e}")
            raise

    def compute_lst_anomaly(self, lst_current: ee.Image, lst_baseline: ee.Image) -> ee.Image:
        """
        Compute LST anomaly (simple difference from baseline).

        Formula: anomaly = LST_current - LST_baseline

        Args:
            lst_current: Current period LST.
            lst_baseline: Baseline LST.

        Returns:
            ee.Image with 'lst_anomaly' band in degrees Celsius.
        """
        logger.info("Computing LST anomaly.")
        anomaly = lst_current.select('LST').subtract(lst_baseline.select('LST')).rename('lst_anomaly')
        return anomaly

    def classify(self, hsi_image: ee.Image) -> ee.Image:
        """
        Classify Heat Stress Index.

        Classes:
            0: Below average (HSI < 0)
            1: Moderate stress (0 <= HSI < 1)
            2: High stress (1 <= HSI < 2)
            3: Extreme stress (HSI >= 2)

        Args:
            hsi_image: ee.Image with heat_stress_index band.

        Returns:
            ee.Image with 'hsi_class' band.
        """
        logger.info("Classifying Heat Stress Index.")
        classified = ee.Image(0) \
            .where(hsi_image.lt(0), 0) \
            .where(hsi_image.gte(0).And(hsi_image.lt(1)), 1) \
            .where(hsi_image.gte(1).And(hsi_image.lt(2)), 2) \
            .where(hsi_image.gte(2), 3) \
            .rename('hsi_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for Heat Stress Index."""
        return {
            'min': -2,
            'max': 3,
            'palette': ['#313695', '#4575b4', '#fee090', '#f46d43', '#d73027', '#a50026']
        }

    def export_to_drive(self, hsi_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'heat_stress_export',
                        folder: str = 'heat_stress', scale: int = 30) -> ee.batch.Task:
        """Export HSI image to Google Drive."""
        logger.info(f"Exporting HSI to Drive: {description}")
        task = ee.batch.Export.image.toDrive(
            image=hsi_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        return task

    def export_to_asset(self, hsi_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'heat_stress_asset',
                        scale: int = 30) -> ee.batch.Task:
        """Export HSI image to GEE Asset."""
        logger.info(f"Exporting HSI to Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=hsi_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        return task
