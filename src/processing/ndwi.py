"""
NDWI (Normalized Difference Water Index) computation module.

NDWI = (Green - NIR) / (Green + NIR)
Range: -1 to 1 (water bodies: > 0.3)
"""
import ee
from loguru import logger
from typing import Dict, Optional


class NDWIProcessor:
    """Compute NDWI from satellite imagery using Google Earth Engine."""

    def __init__(self):
        """Initialize the NDWI processor."""
        logger.debug("Initialized NDWIProcessor.")

    def compute_from_sentinel2(self, image: ee.Image) -> ee.Image:
        """
        Compute NDWI from a Sentinel-2 image.

        Formula: (B3 - B8) / (B3 + B8)

        Args:
            image: Sentinel-2 ee.Image with B3 (Green) and B8 (NIR) bands.

        Returns:
            ee.Image with a single 'NDWI' band.
        """
        try:
            logger.info("Computing NDWI from Sentinel-2 image.")
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            return ndwi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDWI from Sentinel-2: {e}")
            raise

    def compute_from_landsat(self, image: ee.Image) -> ee.Image:
        """
        Compute NDWI from a Landsat 8/9 image.

        Formula: (SR_B3 - SR_B5) / (SR_B3 + SR_B5)

        Args:
            image: Landsat ee.Image with SR_B3 (Green) and SR_B5 (NIR) bands.

        Returns:
            ee.Image with a single 'NDWI' band.
        """
        try:
            logger.info("Computing NDWI from Landsat image.")
            ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
            return ndwi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDWI from Landsat: {e}")
            raise

    def compute(self, image: ee.Image, green_band: str, nir_band: str) -> ee.Image:
        """
        Compute NDWI from any image with specified band names.

        Args:
            image: ee.Image containing the Green and NIR bands.
            green_band: Name of the Green band.
            nir_band: Name of the NIR band.

        Returns:
            ee.Image with a single 'NDWI' band.
        """
        try:
            logger.info(f"Computing NDWI using bands: Green={green_band}, NIR={nir_band}")
            ndwi = image.normalizedDifference([green_band, nir_band]).rename('NDWI')
            return ndwi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDWI: {e}")
            raise

    def extract_water_mask(self, ndwi_image: ee.Image, threshold: float = 0.3) -> ee.Image:
        """
        Extract binary water mask from NDWI image.

        Args:
            ndwi_image: ee.Image with NDWI band.
            threshold: NDWI threshold for water detection (default 0.3).

        Returns:
            ee.Image with binary 'water_mask' band (1=water, 0=non-water).
        """
        logger.info(f"Extracting water mask with threshold={threshold}")
        water_mask = ndwi_image.gt(threshold).rename('water_mask')
        return water_mask

    def classify(self, ndwi_image: ee.Image) -> ee.Image:
        """
        Classify NDWI into water availability categories.

        Classes:
            0: No water (NDWI < -0.3)
            1: Low moisture (-.3 <= NDWI < 0)
            2: Moderate moisture (0 <= NDWI < 0.3)
            3: Water body (NDWI >= 0.3)

        Args:
            ndwi_image: ee.Image with NDWI band.

        Returns:
            ee.Image with classified 'NDWI_class' band.
        """
        logger.info("Classifying NDWI into water availability categories.")
        classified = ee.Image(0) \
            .where(ndwi_image.lt(-0.3), 0) \
            .where(ndwi_image.gte(-0.3).And(ndwi_image.lt(0)), 1) \
            .where(ndwi_image.gte(0).And(ndwi_image.lt(0.3)), 2) \
            .where(ndwi_image.gte(0.3), 3) \
            .rename('NDWI_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for NDWI."""
        return {
            'min': -0.5,
            'max': 0.5,
            'palette': ['#8b4513', '#d2b48c', '#87ceeb', '#4169e1', '#00008b']
        }

    def export_to_drive(self, ndwi_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'NDWI_export',
                        folder: str = 'ndwi', scale: int = 10) -> ee.batch.Task:
        """Export NDWI image to Google Drive as GeoTIFF."""
        logger.info(f"Exporting NDWI to Drive: {description} at {scale}m.")
        task = ee.batch.Export.image.toDrive(
            image=ndwi_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        logger.info(f"Export task '{description}' started.")
        return task

    def export_to_asset(self, ndwi_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'NDWI_asset',
                        scale: int = 10) -> ee.batch.Task:
        """Export NDWI image to a GEE Asset."""
        logger.info(f"Exporting NDWI to GEE Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=ndwi_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        logger.info(f"Asset export task '{description}' started.")
        return task
