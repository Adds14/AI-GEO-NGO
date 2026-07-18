"""
LST (Land Surface Temperature) computation module.

Computes LST from Landsat 8/9 thermal bands.
Output in degrees Celsius.
"""
import ee
from loguru import logger
from typing import Dict, Optional


class LSTProcessor:
    """Compute Land Surface Temperature from Landsat thermal imagery."""

    # Emissivity correction constants
    EMISSIVITY_CONSTANTS = {
        'soil': 0.966,
        'vegetation': 0.973,
        'water': 0.991,
        'built_up': 0.962
    }

    def __init__(self):
        """Initialize the LST processor."""
        logger.debug("Initialized LSTProcessor.")

    def compute_from_landsat(self, image: ee.Image) -> ee.Image:
        """
        Compute LST from a Landsat 8/9 image.

        Uses ST_B10 (Surface Temperature band) and converts from Kelvin to Celsius.
        Assumes scale factors have already been applied.

        Args:
            image: Landsat ee.Image with scaled ST_B10 band.

        Returns:
            ee.Image with a single 'LST' band in degrees Celsius.
        """
        try:
            logger.info("Computing LST from Landsat thermal band.")
            lst = image.select('ST_B10').subtract(273.15).rename('LST')
            return lst
        except ee.EEException as e:
            logger.error(f"Failed to compute LST from Landsat: {e}")
            raise

    def compute_with_emissivity(self, image: ee.Image, ndvi_image: ee.Image) -> ee.Image:
        """
        Compute LST with NDVI-based emissivity correction.

        Steps:
        1. Compute Proportion of Vegetation (Pv) from NDVI
        2. Estimate emissivity from Pv
        3. Apply emissivity correction to brightness temperature

        Args:
            image: Landsat ee.Image with scaled ST_B10.
            ndvi_image: ee.Image with NDVI band.

        Returns:
            ee.Image with emissivity-corrected 'LST' band in Celsius.
        """
        try:
            logger.info("Computing LST with emissivity correction.")

            # Proportion of vegetation
            ndvi_min = ee.Number(0.2)
            ndvi_max = ee.Number(0.5)
            pv = ndvi_image.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(2).rename('Pv')
            pv = pv.where(ndvi_image.lt(ndvi_min), 0)
            pv = pv.where(ndvi_image.gt(ndvi_max), 1)

            # Emissivity
            emissivity = pv.multiply(0.004).add(0.986).rename('emissivity')

            # Brightness temperature (already in Kelvin from scaled ST_B10)
            bt = image.select('ST_B10')

            # LST using Planck's function simplification
            # LST = BT / (1 + (wavelength * BT / rho) * ln(emissivity))
            # wavelength for band 10 = 10.8 um, rho = h*c/sigma = 14380 um*K
            wavelength = ee.Number(10.8)
            rho = ee.Number(14380)

            lst = bt.divide(
                ee.Image(1).add(
                    bt.multiply(wavelength).divide(rho).multiply(emissivity.log())
                )
            ).subtract(273.15).rename('LST')

            return lst
        except ee.EEException as e:
            logger.error(f"Failed to compute LST with emissivity correction: {e}")
            raise

    def compute_from_composite(self, composite: ee.Image) -> ee.Image:
        """
        Compute LST from a pre-processed Landsat composite.
        Expects the composite to already have LST_Celsius band from LandsatLoader.

        Args:
            composite: Landsat composite with 'LST_Celsius' band.

        Returns:
            ee.Image with 'LST' band.
        """
        try:
            logger.info("Extracting LST from Landsat composite.")
            lst = composite.select('LST_Celsius').rename('LST')
            return lst
        except ee.EEException as e:
            logger.error(f"Failed to extract LST from composite: {e}")
            raise

    def classify(self, lst_image: ee.Image) -> ee.Image:
        """
        Classify LST into heat categories (Celsius).

        Classes:
            0: Cool (< 20°C)
            1: Moderate (20-30°C)
            2: Warm (30-35°C)
            3: Hot (35-40°C)
            4: Extreme heat (>= 40°C)

        Args:
            lst_image: ee.Image with LST band in Celsius.

        Returns:
            ee.Image with 'LST_class' band.
        """
        logger.info("Classifying LST into heat categories.")
        classified = ee.Image(0) \
            .where(lst_image.lt(20), 0) \
            .where(lst_image.gte(20).And(lst_image.lt(30)), 1) \
            .where(lst_image.gte(30).And(lst_image.lt(35)), 2) \
            .where(lst_image.gte(35).And(lst_image.lt(40)), 3) \
            .where(lst_image.gte(40), 4) \
            .rename('LST_class')
        return classified

    def get_vis_params(self) -> dict:
        """Get visualization parameters for LST."""
        return {
            'min': 15,
            'max': 45,
            'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090',
                        '#fdae61', '#f46d43', '#d73027', '#a50026']
        }

    def export_to_drive(self, lst_image: ee.Image, aoi: ee.Geometry,
                        description: str = 'LST_export',
                        folder: str = 'lst', scale: int = 30) -> ee.batch.Task:
        """Export LST image to Google Drive as GeoTIFF."""
        logger.info(f"Exporting LST to Drive: {description} at {scale}m.")
        task = ee.batch.Export.image.toDrive(
            image=lst_image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
        task.start()
        logger.info(f"Export task '{description}' started.")
        return task

    def export_to_asset(self, lst_image: ee.Image, aoi: ee.Geometry,
                        asset_id: str, description: str = 'LST_asset',
                        scale: int = 30) -> ee.batch.Task:
        """Export LST image to a GEE Asset."""
        logger.info(f"Exporting LST to GEE Asset: {asset_id}")
        task = ee.batch.Export.image.toAsset(
            image=lst_image, description=description, assetId=asset_id,
            region=aoi, scale=scale, maxPixels=1e13
        )
        task.start()
        logger.info(f"Asset export task '{description}' started.")
        return task
