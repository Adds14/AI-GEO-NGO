"""
SRTM Elevation data loading and processing module.
"""
import ee
from loguru import logger
from typing import Optional

class SRTMLoader:
    """Loader for SRTM 30m Digital Elevation Model."""

    def __init__(self, gee_client=None):
        """
        Initialize SRTMLoader.
        
        Args:
            gee_client (GEEClient, optional): GEE client instance.
        """
        self.gee_client = gee_client
        logger.debug("Initialized SRTMLoader.")

    def load_elevation(self, aoi: ee.Geometry) -> ee.Image:
        """
        Load SRTM 30m DEM for the given AOI.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            
        Returns:
            ee.Image: DEM image clipped to AOI.
        """
        logger.info("Loading SRTM DEM data.")
        try:
            dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)
            return dem
        except ee.EEException as e:
            logger.error(f"Error loading SRTM DEM: {e}")
            raise

    def compute_slope(self, aoi: ee.Geometry) -> ee.Image:
        """
        Compute slope from DEM.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            
        Returns:
            ee.Image: Slope image in degrees.
        """
        logger.info("Computing slope from DEM.")
        dem = self.load_elevation(aoi)
        slope = ee.Terrain.slope(dem)
        return slope

    def compute_aspect(self, aoi: ee.Geometry) -> ee.Image:
        """
        Compute aspect from DEM.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            
        Returns:
            ee.Image: Aspect image in degrees.
        """
        logger.info("Computing aspect from DEM.")
        dem = self.load_elevation(aoi)
        aspect = ee.Terrain.aspect(dem)
        return aspect

    def get_terrain_data(self, aoi: ee.Geometry) -> ee.Image:
        """
        Return image with elevation, slope, and aspect bands.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            
        Returns:
            ee.Image: Multi-band terrain image.
        """
        logger.info("Generating multi-band terrain data (Elevation, Slope, Aspect).")
        dem = self.load_elevation(aoi).rename('elevation')
        slope = ee.Terrain.slope(dem).rename('slope')
        aspect = ee.Terrain.aspect(dem).rename('aspect')
        
        terrain = dem.addBands([slope, aspect])
        return terrain

    def export_dem(self, dem: ee.Image, aoi: ee.Geometry, description: str, folder: str = 'srtm', scale: int = 30) -> ee.batch.Task:
        """
        Export DEM image to Google Drive.
        
        Args:
            dem (ee.Image): Image to export.
            aoi (ee.Geometry): Area of interest.
            description (str): Description/filename.
            folder (str): Drive folder.
            scale (int): Scale in meters.
            
        Returns:
            ee.batch.Task: Export task.
        """
        logger.info(f"Exporting DEM data: {description} at {scale}m scale.")
        task = ee.batch.Export.image.toDrive(
            image=dem,
            description=description,
            folder=folder,
            region=aoi,
            scale=scale,
            maxPixels=1e13
        )
        task.start()
        return task
