"""
Sentinel-2 data loading and processing module.
"""
import ee
from loguru import logger
from typing import Optional, Dict

class Sentinel2Loader:
    """Loader for Sentinel-2 surface reflectance imagery."""
    
    # Band mappings
    B2 = 'B2'   # Blue
    B3 = 'B3'   # Green
    B4 = 'B4'   # Red
    B8 = 'B8'   # NIR
    B11 = 'B11' # SWIR1
    B12 = 'B12' # SWIR2
    
    COLLECTION_ID = 'COPERNICUS/S2_SR_HARMONIZED'

    def __init__(self, gee_client=None):
        """
        Initialize the Sentinel2Loader.
        
        Args:
            gee_client (GEEClient, optional): GEE client instance.
        """
        self.gee_client = gee_client
        logger.debug("Initialized Sentinel2Loader.")

    def load_collection(self, aoi: ee.Geometry, start_date: str, end_date: str, cloud_cover_max: float = 20.0) -> ee.ImageCollection:
        """
        Load Sentinel-2 L2A collection filtered by bounds, date, and cloud cover.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            cloud_cover_max (float): Maximum cloud cover percentage.
            
        Returns:
            ee.ImageCollection: Filtered Sentinel-2 ImageCollection.
        """
        logger.info(f"Loading Sentinel-2 collection from {start_date} to {end_date} over AOI with max cloud cover {cloud_cover_max}%.")
        try:
            collection = ee.ImageCollection(self.COLLECTION_ID) \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
            return collection
        except ee.EEException as e:
            logger.error(f"Error loading Sentinel-2 collection: {e}")
            raise

    def mask_clouds(self, image: ee.Image) -> ee.Image:
        """
        Use SCL band to mask clouds, shadows, cirrus.
        
        Args:
            image (ee.Image): Sentinel-2 image.
            
        Returns:
            ee.Image: Cloud-masked image.
        """
        scl = image.select('SCL')
        # SCL class 3: cloud shadows, 8: cloud medium prob, 9: cloud high prob, 10: cirrus
        mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(mask)

    def add_indices(self, image: ee.Image) -> ee.Image:
        """
        Add NDVI, NDWI, NDBI bands to an image.
        
        Args:
            image (ee.Image): Sentinel-2 image.
            
        Returns:
            ee.Image: Image with added index bands.
        """
        ndvi = image.normalizedDifference([self.B8, self.B4]).rename('NDVI')
        ndwi = image.normalizedDifference([self.B3, self.B8]).rename('NDWI')
        ndbi = image.normalizedDifference([self.B11, self.B8]).rename('NDBI')
        return image.addBands([ndvi, ndwi, ndbi])

    def create_composite(self, aoi: ee.Geometry, start_date: str, end_date: str, cloud_cover_max: float = 20.0, method: str = 'median') -> ee.Image:
        """
        Apply cloud masking and create a median/mean composite.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_date (str): Start date.
            end_date (str): End date.
            cloud_cover_max (float): Maximum cloud cover.
            method (str): Compositing method ('median' or 'mean').
            
        Returns:
            ee.Image: Cloud-masked composite image.
        """
        logger.info(f"Creating Sentinel-2 {method} composite from {start_date} to {end_date}.")
        collection = self.load_collection(aoi, start_date, end_date, cloud_cover_max)
        masked_collection = collection.map(self.mask_clouds)
        
        if method.lower() == 'mean':
            composite = masked_collection.mean()
        else:
            composite = masked_collection.median()
            
        composite = self.add_indices(composite).clip(aoi)
        return composite

    def create_yearly_composites(self, aoi: ee.Geometry, start_year: int, end_year: int, cloud_cover_max: float = 20.0) -> Dict[int, ee.Image]:
        """
        Loop through years and return a dictionary of year->composite.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_year (int): Start year.
            end_year (int): End year.
            cloud_cover_max (float): Maximum cloud cover.
            
        Returns:
            Dict[int, ee.Image]: Dictionary mapping year to its composite image.
        """
        composites = {}
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing yearly composite for {year}.")
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            comp = self.create_composite(aoi, start_date, end_date, cloud_cover_max)
            composites[year] = comp
        return composites

    def export_composite(self, composite: ee.Image, aoi: ee.Geometry, description: str, folder: str = 'sentinel2', scale: int = 10) -> ee.batch.Task:
        """
        Export composite to Google Drive.
        
        Args:
            composite (ee.Image): Image to export.
            aoi (ee.Geometry): Area of interest.
            description (str): Task description/filename.
            folder (str): Google Drive folder.
            scale (int): Export scale in meters.
            
        Returns:
            ee.batch.Task: Earth Engine export task.
        """
        logger.info(f"Exporting Sentinel-2 composite: {description} at {scale}m scale.")
        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=description,
            folder=folder,
            region=aoi,
            scale=scale,
            maxPixels=1e13
        )
        task.start()
        return task

    def get_band_names(self) -> list:
        """
        Get the list of key Sentinel-2 band names.
        
        Returns:
            list: Band names.
        """
        return [self.B2, self.B3, self.B4, self.B8, self.B11, self.B12]
