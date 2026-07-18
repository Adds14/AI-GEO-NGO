"""
Water Body Change Detection module.

Detects changes in surface water extent between two time periods
using NDWI differencing and thresholded water masks.

Water Change = WaterMask(t2) - WaterMask(t1)
  +1: Water gain (new water body)
  0:  No change
  -1: Water loss (dried up)
"""
import ee
from loguru import logger
from typing import Dict, Optional


class WaterChangeProcessor:
    """Detect surface water body changes between two time periods."""

    def __init__(self, water_threshold: float = 0.3):
        """
        Initialize the Water Change processor.

        Args:
            water_threshold: NDWI threshold for water detection (default 0.3).
        """
        self.water_threshold = water_threshold
        logger.debug(f"Initialized WaterChangeProcessor with threshold={water_threshold}.")

    def compute_water_mask(self, ndwi_image: ee.Image) -> ee.Image:
        """
        Create a binary water mask from NDWI.

        Args:
            ndwi_image: ee.Image with NDWI band.

        Returns:
            ee.Image with binary 'water_mask' band (1=water, 0=non-water).
        """
        return ndwi_image.gt(self.water_threshold).rename('water_mask')

    def compute_change(self, ndwi_t1: ee.Image, ndwi_t2: ee.Image) -> ee.Image:
        """
        Compute water body change between two NDWI images.

        Args:
            ndwi_t1: NDWI image for earlier time period.
            ndwi_t2: NDWI image for later time period.

        Returns:
            ee.Image with 'water_change' band.
            +1 = water gain, 0 = no change, -1 = water loss
        """
        try:
            logger.info("Computing water body change.")
            mask_t1 = self.compute_water_mask(ndwi_t1)
            mask_t2 = self.compute_water_mask(ndwi_t2)
            change = mask_t2.subtract(mask_t1).rename('water_change')
            return change
        except ee.EEException as e:
            logger.error(f"Failed to compute water change: {e}")
            raise

    def compute_ndwi_difference(self, ndwi_t1: ee.Image, ndwi_t2: ee.Image) -> ee.Image:
        """
        Compute continuous NDWI difference (not thresholded).

        Args:
            ndwi_t1: NDWI image for earlier period.
            ndwi_t2: NDWI image for later period.

        Returns:
            ee.Image with 'ndwi_difference' band.
        """
        logger.info("Computing NDWI difference.")
        diff = ndwi_t2.subtract(ndwi_t1).rename('ndwi_difference')
        return diff

    def compute_water_area_change(self, ndwi_t1: ee.Image, ndwi_t2: ee.Image,
                                   aoi: ee.Geometry, scale: int = 10) -> dict:
        """
        Compute water area change statistics.

        Args:
            ndwi_t1: NDWI for earlier period.
            ndwi_t2: NDWI for later period.
            aoi: Area of interest.
            scale: Scale in meters.

        Returns:
            Dictionary with water area stats:
            {'area_t1_km2', 'area_t2_km2', 'change_km2', 'change_pct'}
        """
        logger.info("Computing water area change statistics.")
        mask_t1 = self.compute_water_mask(ndwi_t1)
        mask_t2 = self.compute_water_mask(ndwi_t2)

        pixel_area = ee.Image.pixelArea()

        area_t1 = mask_t1.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e13, bestEffort=True
        ).get('water_mask')

        area_t2 = mask_t2.multiply(pixel_area).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, scale=scale,
            maxPixels=1e13, bestEffort=True
        ).get('water_mask')

        area_t1_val = ee.Number(area_t1).divide(1e6)  # Convert to km2
        area_t2_val = ee.Number(area_t2).divide(1e6)
        change = area_t2_val.subtract(area_t1_val)
        pct = change.divide(area_t1_val.add(0.0001)).multiply(100)

        result = ee.Dictionary({
            'area_t1_km2': area_t1_val,
            'area_t2_km2': area_t2_val,
            'change_km2': change,
            'change_pct': pct
        })
        return result.getInfo()

    def classify_change(self, change_image: ee.Image) -> ee.Image:
        """
        Classify water change into categories.

        Classes:
            0: Water loss (-1)
            1: No change (0)
            2: Water gain (+1)

        Args:
            change_image: ee.Image with water_change band.

        Returns:
            ee.Image with 'water_change_class' band.
        """
        logger.info("Classifying water body change.")
        classified = ee.Image(1) \
            .where(change_image.lt(0), 0) \
            .where(change_image.eq(0), 1) \
            .where(change_image.gt(0), 2) \
            .rename('water_change_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for water change."""
        return {
            'min': -1,
            'max': 1,
            'palette': ['#d73027', '#ffffbf', '#4575b4']
        }

    def export_to_drive(self, change_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'water_change_export',
                        folder: str = 'water_change', scale: int = 10) -> ee.batch.Task:
        """Export water change image to Google Drive."""
        logger.info(f"Exporting water change to Drive: {description}")
        task = ee.batch.Export.image.toDrive(
            image=change_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        return task

    def export_to_asset(self, change_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'water_change_asset',
                        scale: int = 10) -> ee.batch.Task:
        """Export water change image to GEE Asset."""
        logger.info(f"Exporting water change to Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=change_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        return task
