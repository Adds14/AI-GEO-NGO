"""
Landsat 8/9 data loading and processing module.
"""
import ee
from loguru import logger
from typing import Optional, Dict

class LandsatLoader:
    """Loader for Landsat 8 and 9 Collection 2 L2 imagery."""

    def __init__(self, gee_client=None, satellite: str = 'both'):
        """
        Initialize the LandsatLoader.
        
        Args:
            gee_client (GEEClient, optional): GEE client instance.
            satellite (str): '8', '9', or 'both'.
        """
        self.gee_client = gee_client
        self.satellite = satellite.lower()
        logger.debug(f"Initialized LandsatLoader for satellite: {self.satellite}")

    def load_collection(self, aoi: ee.Geometry, start_date: str, end_date: str, cloud_cover_max: float = 20.0) -> ee.ImageCollection:
        """
        Load Landsat Collection 2 L2 data filtered by bounds, date, and cloud cover.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_date (str): Start date.
            end_date (str): End date.
            cloud_cover_max (float): Maximum cloud cover.
            
        Returns:
            ee.ImageCollection: Filtered Landsat ImageCollection.
        """
        logger.info(f"Loading Landsat collection from {start_date} to {end_date}.")
        col8, col9 = None, None
        
        try:
            if self.satellite in ['8', 'both']:
                col8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                    .filterBounds(aoi) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))
                    
            if self.satellite in ['9', 'both']:
                col9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') \
                    .filterBounds(aoi) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max))

            if self.satellite == 'both':
                return col8.merge(col9)
            elif self.satellite == '8':
                return col8
            elif self.satellite == '9':
                return col9
            else:
                raise ValueError("satellite must be '8', '9', or 'both'")
        except ee.EEException as e:
            logger.error(f"Error loading Landsat collection: {e}")
            raise

    def mask_clouds(self, image: ee.Image) -> ee.Image:
        """
        Use QA_PIXEL band for cloud masking with bit manipulation.
        
        Args:
            image (ee.Image): Landsat image.
            
        Returns:
            ee.Image: Cloud-masked image.
        """
        qa = image.select('QA_PIXEL')
        # Bits 1, 2, 3, 4 represent dilated cloud, cirrus, cloud, cloud shadow respectively
        cloud_shadow_bit_mask = (1 << 4)
        clouds_bit_mask = (1 << 3)
        cirrus_bit_mask = (1 << 2)
        dilated_cloud_bit_mask = (1 << 1)

        mask = qa.bitwiseAnd(cloud_shadow_bit_mask).eq(0) \
            .And(qa.bitwiseAnd(clouds_bit_mask).eq(0)) \
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)) \
            .And(qa.bitwiseAnd(dilated_cloud_bit_mask).eq(0))
            
        return image.updateMask(mask)

    def apply_scale_factors(self, image: ee.Image) -> ee.Image:
        """
        Apply Collection 2 scale factors to SR and ST bands.
        
        Args:
            image (ee.Image): Landsat image.
            
        Returns:
            ee.Image: Scaled image.
        """
        optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal_bands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

    def compute_lst(self, image: ee.Image) -> ee.Image:
        """
        Convert thermal band (ST_B10) to Land Surface Temperature in Celsius.
        
        Args:
            image (ee.Image): Landsat image (must have scaled ST_B10).
            
        Returns:
            ee.Image: Image with LST_Celsius band.
        """
        lst_celsius = image.select('ST_B10').subtract(273.15).rename('LST_Celsius')
        return image.addBands(lst_celsius)

    def create_composite(self, aoi: ee.Geometry, start_date: str, end_date: str, cloud_cover_max: float = 20.0, method: str = 'median') -> ee.Image:
        """
        Create cloud-free composite.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_date (str): Start date.
            end_date (str): End date.
            cloud_cover_max (float): Maximum cloud cover.
            method (str): Compositing method.
            
        Returns:
            ee.Image: Landsat composite.
        """
        logger.info(f"Creating Landsat {method} composite from {start_date} to {end_date}.")
        collection = self.load_collection(aoi, start_date, end_date, cloud_cover_max)
        
        processed_col = collection.map(self.mask_clouds).map(self.apply_scale_factors)
        
        if method.lower() == 'mean':
            composite = processed_col.mean()
        else:
            composite = processed_col.median()
            
        composite = self.compute_lst(composite).clip(aoi)
        return composite

    def create_yearly_composites(self, aoi: ee.Geometry, start_year: int, end_year: int, cloud_cover_max: float = 20.0) -> Dict[int, ee.Image]:
        """
        Create yearly composites.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_year (int): Start year.
            end_year (int): End year.
            cloud_cover_max (float): Maximum cloud cover.
            
        Returns:
            Dict[int, ee.Image]: Dictionary of yearly composites.
        """
        composites = {}
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing Landsat yearly composite for {year}.")
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            comp = self.create_composite(aoi, start_date, end_date, cloud_cover_max)
            composites[year] = comp
        return composites

    def export_composite(self, composite: ee.Image, aoi: ee.Geometry, description: str, folder: str = 'landsat', scale: int = 30) -> ee.batch.Task:
        """
        Export composite to Google Drive.
        
        Args:
            composite (ee.Image): Image to export.
            aoi (ee.Geometry): Area of interest.
            description (str): Description/filename.
            folder (str): Drive folder.
            scale (int): Scale in meters.
            
        Returns:
            ee.batch.Task: Export task.
        """
        logger.info(f"Exporting Landsat composite: {description} at {scale}m scale.")
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
