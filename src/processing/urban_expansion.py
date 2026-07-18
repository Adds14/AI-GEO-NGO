"""
Urban Expansion Detection module.

Detects urban growth between two time periods using NDBI differencing.

Urban Growth = NDBI(t2) - NDBI(t1)
  Positive: Urban expansion
  Negative: Urban decline / greening
"""
import ee
from loguru import logger
from typing import Dict, Optional


class UrbanExpansionProcessor:
    """Detect urban expansion between two time periods."""

    def __init__(self, builtup_threshold: float = 0.0):
        """
        Initialize the Urban Expansion processor.

        Args:
            builtup_threshold: NDBI threshold for built-up detection.
        """
        self.builtup_threshold = builtup_threshold
        logger.debug(f"Initialized UrbanExpansionProcessor with threshold={builtup_threshold}.")

    def compute_builtup_mask(self, ndbi_image: ee.Image) -> ee.Image:
        """
        Create a binary built-up mask from NDBI.

        Args:
            ndbi_image: ee.Image with NDBI band.

        Returns:
            ee.Image with binary 'builtup_mask' band.
        """
        return ndbi_image.gt(self.builtup_threshold).rename('builtup_mask')

    def compute_change(self, ndbi_t1: ee.Image, ndbi_t2: ee.Image) -> ee.Image:
        """
        Compute urban expansion from two NDBI images.

        Args:
            ndbi_t1: NDBI for earlier time period.
            ndbi_t2: NDBI for later time period.

        Returns:
            ee.Image with 'urban_growth' band.
        """
        try:
            logger.info("Computing urban expansion (NDBI difference).")
            change = ndbi_t2.subtract(ndbi_t1).rename('urban_growth')
            return change
        except ee.EEException as e:
            logger.error(f"Failed to compute urban expansion: {e}")
            raise

    def compute_new_builtup(self, ndbi_t1: ee.Image, ndbi_t2: ee.Image) -> ee.Image:
        """
        Identify newly built-up areas (were non-urban, now urban).

        Args:
            ndbi_t1: NDBI for earlier period.
            ndbi_t2: NDBI for later period.

        Returns:
            ee.Image with binary 'new_builtup' band.
        """
        logger.info("Identifying newly built-up areas.")
        mask_t1 = self.compute_builtup_mask(ndbi_t1)
        mask_t2 = self.compute_builtup_mask(ndbi_t2)
        new_builtup = mask_t2.subtract(mask_t1).gt(0).rename('new_builtup')
        return new_builtup

    def compute_urban_area_stats(self, ndbi_t1: ee.Image, ndbi_t2: ee.Image,
                                  aoi: ee.Geometry, scale: int = 10) -> dict:
        """
        Compute urban area change statistics.

        Args:
            ndbi_t1: NDBI for earlier period.
            ndbi_t2: NDBI for later period.
            aoi: Area of interest.
            scale: Scale in meters.

        Returns:
            Dictionary with urban area stats.
        """
        logger.info("Computing urban area change statistics.")
        mask_t1 = self.compute_builtup_mask(ndbi_t1)
        mask_t2 = self.compute_builtup_mask(ndbi_t2)
        pixel_area = ee.Image.pixelArea()

        area_t1 = mask_t1.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e13, bestEffort=True
        ).get('builtup_mask')

        area_t2 = mask_t2.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e13, bestEffort=True
        ).get('builtup_mask')

        area_t1_val = ee.Number(area_t1).divide(1e6)
        area_t2_val = ee.Number(area_t2).divide(1e6)
        change = area_t2_val.subtract(area_t1_val)
        growth_rate = change.divide(area_t1_val.add(0.0001)).multiply(100)

        result = ee.Dictionary({
            'urban_area_t1_km2': area_t1_val,
            'urban_area_t2_km2': area_t2_val,
            'change_km2': change,
            'growth_rate_pct': growth_rate
        })
        return result.getInfo()

    def classify_change(self, change_image: ee.Image) -> ee.Image:
        """
        Classify urban growth.

        Classes:
            0: Urban decline (change < -0.1)
            1: Stable (-0.1 <= change < 0.1)
            2: Moderate growth (0.1 <= change < 0.2)
            3: Rapid growth (change >= 0.2)

        Args:
            change_image: ee.Image with urban_growth band.

        Returns:
            ee.Image with 'urban_growth_class' band.
        """
        logger.info("Classifying urban growth.")
        classified = ee.Image(1) \
            .where(change_image.lt(-0.1), 0) \
            .where(change_image.gte(-0.1).And(change_image.lt(0.1)), 1) \
            .where(change_image.gte(0.1).And(change_image.lt(0.2)), 2) \
            .where(change_image.gte(0.2), 3) \
            .rename('urban_growth_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for urban expansion."""
        return {
            'min': -0.2,
            'max': 0.3,
            'palette': ['#2166ac', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']
        }

    def export_to_drive(self, change_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'urban_expansion_export',
                        folder: str = 'urban_expansion', scale: int = 10) -> ee.batch.Task:
        """Export urban expansion image to Google Drive."""
        logger.info(f"Exporting urban expansion to Drive: {description}")
        task = ee.batch.Export.image.toDrive(
            image=change_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        return task

    def export_to_asset(self, change_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'urban_expansion_asset',
                        scale: int = 10) -> ee.batch.Task:
        """Export urban expansion image to GEE Asset."""
        logger.info(f"Exporting urban expansion to Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=change_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        return task
