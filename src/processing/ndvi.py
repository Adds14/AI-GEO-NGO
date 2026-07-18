"""
NDVI (Normalized Difference Vegetation Index) computation module.

NDVI = (NIR - Red) / (NIR + Red)
Range: -1 to 1 (healthy vegetation: 0.3-0.8)
"""
import ee
from loguru import logger
from typing import Dict, Optional


class NDVIProcessor:
    """Compute NDVI from satellite imagery using Google Earth Engine."""

    def __init__(self):
        """Initialize the NDVI processor."""
        logger.debug("Initialized NDVIProcessor.")

    def compute_from_sentinel2(self, image: ee.Image) -> ee.Image:
        """
        Compute NDVI from a Sentinel-2 image.

        Formula: (B8 - B4) / (B8 + B4)

        Args:
            image: Sentinel-2 ee.Image with B8 (NIR) and B4 (Red) bands.

        Returns:
            ee.Image with a single 'NDVI' band.

        Raises:
            ee.EEException: If computation fails.
        """
        try:
            logger.info("Computing NDVI from Sentinel-2 image.")
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            return ndvi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDVI from Sentinel-2: {e}")
            raise

    def compute_from_landsat(self, image: ee.Image) -> ee.Image:
        """
        Compute NDVI from a Landsat 8/9 image.

        Formula: (SR_B5 - SR_B4) / (SR_B5 + SR_B4)

        Args:
            image: Landsat ee.Image with SR_B5 (NIR) and SR_B4 (Red) bands.

        Returns:
            ee.Image with a single 'NDVI' band.

        Raises:
            ee.EEException: If computation fails.
        """
        try:
            logger.info("Computing NDVI from Landsat image.")
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            return ndvi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDVI from Landsat: {e}")
            raise

    def compute(self, image: ee.Image, nir_band: str, red_band: str) -> ee.Image:
        """
        Compute NDVI from any image with specified band names.

        Args:
            image: ee.Image containing the NIR and Red bands.
            nir_band: Name of the NIR band.
            red_band: Name of the Red band.

        Returns:
            ee.Image with a single 'NDVI' band.
        """
        try:
            logger.info(f"Computing NDVI using bands: NIR={nir_band}, Red={red_band}")
            ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
            return ndvi
        except ee.EEException as e:
            logger.error(f"Failed to compute NDVI: {e}")
            raise

    def classify(self, ndvi_image: ee.Image) -> ee.Image:
        """
        Classify NDVI into vegetation categories.

        Classes:
            0: Water/No vegetation (NDVI < 0)
            1: Bare soil (0 <= NDVI < 0.2)
            2: Sparse vegetation (0.2 <= NDVI < 0.4)
            3: Moderate vegetation (0.4 <= NDVI < 0.6)
            4: Dense vegetation (NDVI >= 0.6)

        Args:
            ndvi_image: ee.Image with NDVI band.

        Returns:
            ee.Image with classified 'NDVI_class' band.
        """
        logger.info("Classifying NDVI into vegetation categories.")
        classified = ee.Image(0) \
            .where(ndvi_image.gte(-1).And(ndvi_image.lt(0)), 0) \
            .where(ndvi_image.gte(0).And(ndvi_image.lt(0.2)), 1) \
            .where(ndvi_image.gte(0.2).And(ndvi_image.lt(0.4)), 2) \
            .where(ndvi_image.gte(0.4).And(ndvi_image.lt(0.6)), 3) \
            .where(ndvi_image.gte(0.6), 4) \
            .rename('NDVI_class')
        return classified

    def get_vis_params(self) -> dict:
        """
        Get visualization parameters for NDVI.

        Returns:
            Dictionary of visualization parameters for GEE/Folium.
        """
        return {
            'min': -0.2,
            'max': 0.8,
            'palette': ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
        }

    def get_class_vis_params(self) -> dict:
        """
        Get visualization parameters for classified NDVI.

        Returns:
            Dictionary of visualization parameters.
        """
        return {
            'min': 0,
            'max': 4,
            'palette': ['#2166ac', '#d1a555', '#cddc39', '#66bb6a', '#1b5e20']
        }

    def export_to_drive(self, ndvi_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'NDVI_export',
                        folder: str = 'ndvi', scale: int = 10) -> ee.batch.Task:
        """
        Export NDVI image to Google Drive as GeoTIFF.

        Args:
            ndvi_image: NDVI ee.Image to export.
            aoi: Area of interest geometry.
            description: Export task description.
            folder: Google Drive folder name.
            scale: Export scale in meters.

        Returns:
            ee.batch.Task: The started export task.
        """
        logger.info(f"Exporting NDVI to Drive: {description} at {scale}m.")
        task = ee.batch.Export.image.toDrive(
            image=ndvi_image,
            description=description,
            folder=folder,
            region=aoi,
            scale=scale,
            maxPixels=1e13,
            fileFormat='GeoTIFF'
        )
        task.start()
        logger.info(f"Export task '{description}' started.")
        return task

    def export_to_asset(self, ndvi_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'NDVI_asset',
                        scale: int = 10) -> ee.batch.Task:
        """
        Export NDVI image to a GEE Asset.

        Args:
            ndvi_image: NDVI ee.Image to export.
            aoi: Area of interest geometry.
            asset_id: Full GEE asset path (e.g., 'projects/your-project/assets/ndvi_2023').
            description: Export task description.
            scale: Export scale in meters.

        Returns:
            ee.batch.Task: The started export task.
        """
        logger.info(f"Exporting NDVI to GEE Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=ndvi_image,
            description=description,
            assetId=asset_id,
            region=aoi,
            scale=scale,
            maxPixels=1e13
        )
        task.start()
        logger.info(f"Asset export task '{description}' started.")
        return task
