"""
Vegetation Change Detection module.

Computes change in vegetation cover between two time periods
using NDVI differencing.

Vegetation Change = NDVI(t2) - NDVI(t1)
  Positive: Vegetation gain (greening)
  Negative: Vegetation loss (browning/deforestation)
"""
import ee
from loguru import logger
from typing import Dict, Optional, Tuple


class VegetationChangeProcessor:
    """Detect vegetation change between two time periods."""

    def __init__(self):
        """Initialize the Vegetation Change processor."""
        logger.debug("Initialized VegetationChangeProcessor.")

    def compute_change(self, ndvi_t1: ee.Image, ndvi_t2: ee.Image) -> ee.Image:
        """
        Compute vegetation change between two NDVI images.

        Formula: change = NDVI(t2) - NDVI(t1)

        Args:
            ndvi_t1: NDVI image for earlier time period.
            ndvi_t2: NDVI image for later time period.

        Returns:
            ee.Image with 'vegetation_change' band.
            Positive values indicate vegetation gain,
            negative values indicate vegetation loss.
        """
        try:
            logger.info("Computing vegetation change (NDVI difference).")
            change = ndvi_t2.subtract(ndvi_t1).rename('vegetation_change')
            return change
        except ee.EEException as e:
            logger.error(f"Failed to compute vegetation change: {e}")
            raise

    def compute_percent_change(self, ndvi_t1: ee.Image, ndvi_t2: ee.Image) -> ee.Image:
        """
        Compute percentage change in vegetation.

        Formula: ((NDVI_t2 - NDVI_t1) / |NDVI_t1|) * 100

        Args:
            ndvi_t1: NDVI image for earlier time period.
            ndvi_t2: NDVI image for later time period.

        Returns:
            ee.Image with 'vegetation_pct_change' band.
        """
        try:
            logger.info("Computing vegetation percentage change.")
            diff = ndvi_t2.subtract(ndvi_t1)
            pct_change = diff.divide(ndvi_t1.abs().add(0.001)).multiply(100).rename('vegetation_pct_change')
            return pct_change
        except ee.EEException as e:
            logger.error(f"Failed to compute vegetation percent change: {e}")
            raise

    def compute_from_composites(self, composite_t1: ee.Image, composite_t2: ee.Image,
                                 nir_band: str = 'B8', red_band: str = 'B4') -> ee.Image:
        """
        Compute vegetation change directly from two satellite composites.

        Args:
            composite_t1: Earlier satellite composite.
            composite_t2: Later satellite composite.
            nir_band: NIR band name.
            red_band: Red band name.

        Returns:
            ee.Image with 'vegetation_change' band.
        """
        logger.info("Computing vegetation change from composites.")
        ndvi_t1 = composite_t1.normalizedDifference([nir_band, red_band]).rename('NDVI')
        ndvi_t2 = composite_t2.normalizedDifference([nir_band, red_band]).rename('NDVI')
        return self.compute_change(ndvi_t1, ndvi_t2)

    def classify_change(self, change_image: ee.Image) -> ee.Image:
        """
        Classify vegetation change into categories.

        Classes:
            0: Significant loss (change < -0.2)
            1: Moderate loss (-0.2 <= change < -0.1)
            2: Stable (-0.1 <= change < 0.1)
            3: Moderate gain (0.1 <= change < 0.2)
            4: Significant gain (change >= 0.2)

        Args:
            change_image: ee.Image with vegetation_change band.

        Returns:
            ee.Image with 'veg_change_class' band.
        """
        logger.info("Classifying vegetation change.")
        classified = ee.Image(2) \
            .where(change_image.lt(-0.2), 0) \
            .where(change_image.gte(-0.2).And(change_image.lt(-0.1)), 1) \
            .where(change_image.gte(-0.1).And(change_image.lt(0.1)), 2) \
            .where(change_image.gte(0.1).And(change_image.lt(0.2)), 3) \
            .where(change_image.gte(0.2), 4) \
            .rename('veg_change_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for vegetation change."""
        return {
            'min': -0.3,
            'max': 0.3,
            'palette': ['#d73027', '#fc8d59', '#ffffbf', '#91cf60', '#1a9850']
        }

    def export_to_drive(self, change_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'vegetation_change_export',
                        folder: str = 'vegetation_change', scale: int = 10) -> ee.batch.Task:
        """Export vegetation change image to Google Drive."""
        logger.info(f"Exporting vegetation change to Drive: {description}")
        task = ee.batch.Export.image.toDrive(
            image=change_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        return task

    def export_to_asset(self, change_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'vegetation_change_asset',
                        scale: int = 10) -> ee.batch.Task:
        """Export vegetation change image to GEE Asset."""
        logger.info(f"Exporting vegetation change to Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=change_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        return task
