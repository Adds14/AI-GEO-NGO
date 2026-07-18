"""
Rainfall Anomaly computation module.

Computes rainfall deviation from historical averages using CHIRPS data.

Rainfall Anomaly = (Rainfall_current - Rainfall_mean) / Rainfall_std

Interpretation:
  < -1.5: Severe drought
  -1.5 to -0.5: Moderate drought
  -0.5 to 0.5: Normal
  0.5 to 1.5: Above normal
  > 1.5: Extreme wet
"""
import ee
from loguru import logger
from typing import Dict, Optional


class RainfallAnomalyProcessor:
    """Compute rainfall anomalies from CHIRPS precipitation data."""

    CHIRPS_COLLECTION = 'UCSB-CHG/CHIRPS/DAILY'

    def __init__(self):
        """Initialize the Rainfall Anomaly processor."""
        logger.debug("Initialized RainfallAnomalyProcessor.")

    def compute_annual_total(self, aoi: ee.Geometry, year: int) -> ee.Image:
        """
        Compute total annual rainfall for a given year.

        Args:
            aoi: Area of interest.
            year: Year to compute.

        Returns:
            ee.Image with 'annual_rainfall' band in mm.
        """
        logger.info(f"Computing annual rainfall for {year}.")
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        annual = ee.ImageCollection(self.CHIRPS_COLLECTION) \
            .filterBounds(aoi) \
            .filterDate(start, end) \
            .sum() \
            .clip(aoi) \
            .rename('annual_rainfall')
        return annual

    def compute_baseline(self, aoi: ee.Geometry, start_year: int, end_year: int) -> dict:
        """
        Compute baseline rainfall statistics (mean and std) over multiple years.

        Args:
            aoi: Area of interest.
            start_year: Start year of baseline period.
            end_year: End year of baseline period.

        Returns:
            Dictionary with 'mean' and 'std' ee.Image objects.
        """
        logger.info(f"Computing rainfall baseline from {start_year} to {end_year}.")
        annual_images = []
        for year in range(start_year, end_year + 1):
            annual = self.compute_annual_total(aoi, year)
            annual_images.append(annual)

        collection = ee.ImageCollection(annual_images)
        baseline_mean = collection.mean().rename('rainfall_mean')
        baseline_std = collection.reduce(ee.Reducer.stdDev()).rename('rainfall_std')

        return {'mean': baseline_mean, 'std': baseline_std}

    def compute_anomaly(self, current_rainfall: ee.Image, baseline_mean: ee.Image,
                        baseline_std: ee.Image) -> ee.Image:
        """
        Compute rainfall anomaly (standardized z-score).

        Formula: anomaly = (rainfall - mean) / std

        Args:
            current_rainfall: Current period rainfall image.
            baseline_mean: Historical mean rainfall image.
            baseline_std: Historical std deviation image.

        Returns:
            ee.Image with 'rainfall_anomaly' band.
        """
        try:
            logger.info("Computing rainfall anomaly (z-score).")
            # Avoid division by zero
            safe_std = baseline_std.where(baseline_std.eq(0), 1)
            anomaly = current_rainfall.subtract(baseline_mean).divide(safe_std).rename('rainfall_anomaly')
            return anomaly
        except ee.EEException as e:
            logger.error(f"Failed to compute rainfall anomaly: {e}")
            raise

    def compute_percent_departure(self, current_rainfall: ee.Image,
                                   baseline_mean: ee.Image) -> ee.Image:
        """
        Compute rainfall percent departure from mean.

        Formula: departure = ((current - mean) / mean) * 100

        Args:
            current_rainfall: Current period rainfall.
            baseline_mean: Historical mean rainfall.

        Returns:
            ee.Image with 'rainfall_departure_pct' band.
        """
        logger.info("Computing rainfall percent departure.")
        safe_mean = baseline_mean.where(baseline_mean.eq(0), 1)
        departure = current_rainfall.subtract(baseline_mean).divide(safe_mean).multiply(100) \
            .rename('rainfall_departure_pct')
        return departure

    def classify(self, anomaly_image: ee.Image) -> ee.Image:
        """
        Classify rainfall anomaly into drought/wet categories.

        Classes:
            0: Severe drought (anomaly < -1.5)
            1: Moderate drought (-1.5 <= anomaly < -0.5)
            2: Normal (-0.5 <= anomaly < 0.5)
            3: Above normal (0.5 <= anomaly < 1.5)
            4: Extreme wet (anomaly >= 1.5)

        Args:
            anomaly_image: ee.Image with rainfall_anomaly band.

        Returns:
            ee.Image with 'rainfall_class' band.
        """
        logger.info("Classifying rainfall anomaly.")
        classified = ee.Image(2) \
            .where(anomaly_image.lt(-1.5), 0) \
            .where(anomaly_image.gte(-1.5).And(anomaly_image.lt(-0.5)), 1) \
            .where(anomaly_image.gte(-0.5).And(anomaly_image.lt(0.5)), 2) \
            .where(anomaly_image.gte(0.5).And(anomaly_image.lt(1.5)), 3) \
            .where(anomaly_image.gte(1.5), 4) \
            .rename('rainfall_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for rainfall anomaly."""
        return {
            'min': -2,
            'max': 2,
            'palette': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
        }

    def export_to_drive(self, anomaly_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'rainfall_anomaly_export',
                        folder: str = 'rainfall_anomaly', scale: int = 5000) -> ee.batch.Task:
        """Export rainfall anomaly to Google Drive."""
        logger.info(f"Exporting rainfall anomaly to Drive: {description}")
        task = ee.batch.Export.image.toDrive(
            image=anomaly_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        return task

    def export_to_asset(self, anomaly_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'rainfall_anomaly_asset',
                        scale: int = 5000) -> ee.batch.Task:
        """Export rainfall anomaly to GEE Asset."""
        logger.info(f"Exporting rainfall anomaly to Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=anomaly_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        return task
