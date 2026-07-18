"""
NDBI (Normalized Difference Built-up Index) computation module.

NDBI = (SWIR - NIR) / (SWIR + NIR)
Range: -1 to 1 (built-up areas: > 0)
"""
import ee
from loguru import logger
from typing import Dict, Optional


class NDBIProcessor:
    """Compute NDBI from satellite imagery using Google Earth Engine."""

    def __init__(self):
        """Initialize the NDBI processor."""
        logger.debug("Initialized NDBIProcessor.")

    def compute_from_sentinel2(self, image: ee.Image) -> ee.Image:
        """
        Compute NDBI from a Sentinel-2 image.

        Formula: (B11 - B8) / (B11 + B8)

        Args:
            image: Sentinel-2 ee.Image with B11 (SWIR1) and B8 (NIR) bands.

        Returns:
            ee.Image with a single 'NDBI' band.
        """
        try:
            logger.info("Computing NDBI from Sentinel-2 image.")
            ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
            return ndbi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDBI from Sentinel-2: {e}")
            raise

    def compute_from_landsat(self, image: ee.Image) -> ee.Image:
        """
        Compute NDBI from a Landsat 8/9 image.

        Formula: (SR_B6 - SR_B5) / (SR_B6 + SR_B5)

        Args:
            image: Landsat ee.Image with SR_B6 (SWIR1) and SR_B5 (NIR) bands.

        Returns:
            ee.Image with a single 'NDBI' band.
        """
        try:
            logger.info("Computing NDBI from Landsat image.")
            ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
            return ndbi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDBI from Landsat: {e}")
            raise

    def compute(self, image: ee.Image, swir_band: str, nir_band: str) -> ee.Image:
        """
        Compute NDBI from any image with specified band names.

        Args:
            image: ee.Image containing the SWIR and NIR bands.
            swir_band: Name of the SWIR band.
            nir_band: Name of the NIR band.

        Returns:
            ee.Image with a single 'NDBI' band.
        """
        try:
            logger.info(f"Computing NDBI using bands: SWIR={swir_band}, NIR={nir_band}")
            ndbi = image.normalizedDifference([swir_band, nir_band]).rename('NDBI')
            return ndbi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDBI: {e}")
            raise

    def extract_builtup_mask(self, ndbi_image: ee.Image, threshold: float = 0.0) -> ee.Image:
        """
        Extract binary built-up area mask from NDBI.

        Args:
            ndbi_image: ee.Image with NDBI band.
            threshold: NDBI threshold for built-up detection (default 0.0).

        Returns:
            ee.Image with binary 'builtup_mask' band.
        """
        logger.info(f"Extracting built-up mask with threshold={threshold}")
        builtup_mask = ndbi_image.gt(threshold).rename('builtup_mask')
        return builtup_mask

    def classify(self, ndbi_image: ee.Image) -> ee.Image:
        """
        Classify NDBI into urban density categories.

        Classes:
            0: Non-urban (NDBI < -0.1)
            1: Low density (-.1 <= NDBI < 0.1)
            2: Medium density (0.1 <= NDBI < 0.3)
            3: High density (NDBI >= 0.3)

        Args:
            ndbi_image: ee.Image with NDBI band.

        Returns:
            ee.Image with classified 'NDBI_class' band.
        """
        logger.info("Classifying NDBI into urban density categories.")
        classified = ee.Image(0) \
            .where(ndbi_image.lt(-0.1), 0) \
            .where(ndbi_image.gte(-0.1).And(ndbi_image.lt(0.1)), 1) \
            .where(ndbi_image.gte(0.1).And(ndbi_image.lt(0.3)), 2) \
            .where(ndbi_image.gte(0.3), 3) \
            .rename('NDBI_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for NDBI."""
        return {
            'min': -0.3,
            'max': 0.3,
            'palette': ['#228b22', '#adff2f', '#ffd700', '#ff8c00', '#ff0000']
        }

    def export_to_drive(self, ndbi_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'NDBI_export',
                        folder: str = 'ndbi', scale: int = 10) -> ee.batch.Task:
        """Export NDBI image to Google Drive as GeoTIFF."""
        logger.info(f"Exporting NDBI to Drive: {description} at {scale}m.")
        task = ee.batch.Export.image.toDrive(
            image=ndbi_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        logger.info(f"Export task '{description}' started.")
        return task

    def export_to_asset(self, ndbi_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'NDBI_asset',
                        scale: int = 10) -> ee.batch.Task:
        """Export NDBI image to a GEE Asset."""
        logger.info(f"Exporting NDBI to GEE Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=ndbi_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        logger.info(f"Asset export task '{description}' started.")
        return task
