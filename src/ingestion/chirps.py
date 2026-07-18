"""
CHIRPS rainfall data loading and processing module.
"""
import ee
from loguru import logger
from typing import Dict, Optional

class CHIRPSLoader:
    """Loader for CHIRPS Daily Precipitation data."""

    def __init__(self, gee_client=None):
        """
        Initialize CHIRPSLoader.
        
        Args:
            gee_client (GEEClient, optional): GEE client instance.
        """
        self.gee_client = gee_client
        logger.debug("Initialized CHIRPSLoader.")

    def load_daily(self, aoi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Load CHIRPS daily precipitation data.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_date (str): Start date.
            end_date (str): End date.
            
        Returns:
            ee.ImageCollection: Filtered daily precipitation.
        """
        logger.info(f"Loading CHIRPS daily data from {start_date} to {end_date}.")
        try:
            collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date)
            return collection
        except ee.EEException as e:
            logger.error(f"Error loading CHIRPS daily collection: {e}")
            raise

    def load_monthly(self, aoi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
        """
        Aggregate CHIRPS daily data to monthly precipitation.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_date (str): Start date.
            end_date (str): End date.
            
        Returns:
            ee.ImageCollection: Monthly precipitation.
        """
        logger.info(f"Aggregating CHIRPS to monthly from {start_date} to {end_date}.")
        daily_col = self.load_daily(aoi, start_date, end_date)
        
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        start_month = int(start_date.split('-')[1])
        end_month = int(end_date.split('-')[1])

        months = ee.List.sequence(1, 12)
        years = ee.List.sequence(start_year, end_year)

        def by_year_month(y):
            def by_month(m):
                filtered = daily_col.filter(ee.Filter.calendarRange(y, y, 'year')) \
                                    .filter(ee.Filter.calendarRange(m, m, 'month'))
                return filtered.sum().set('year', y).set('month', m) \
                               .set('system:time_start', ee.Date.fromYMD(y, m, 1).millis())
            return months.map(by_month)

        monthly_list = years.map(by_year_month).flatten()
        monthly_col = ee.ImageCollection.fromImages(monthly_list) \
            .filterDate(start_date, end_date)
            
        return monthly_col

    def compute_annual_rainfall(self, aoi: ee.Geometry, year: int) -> ee.Image:
        """
        Compute total annual rainfall for a specific year.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            year (int): Year.
            
        Returns:
            ee.Image: Total annual precipitation image.
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        logger.info(f"Computing annual rainfall for {year}.")
        daily = self.load_daily(aoi, start_date, end_date)
        annual = daily.sum().clip(aoi).rename(f'precipitation_{year}')
        return annual

    def compute_monthly_average(self, aoi: ee.Geometry, start_year: int, end_year: int) -> ee.ImageCollection:
        """
        Compute mean monthly rainfall over a period.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_year (int): Start year.
            end_year (int): End year.
            
        Returns:
            ee.ImageCollection: Collection of 12 images, one for each month's average.
        """
        logger.info(f"Computing monthly average rainfall from {start_year} to {end_year}.")
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        daily = self.load_daily(aoi, start_date, end_date)
        
        months = ee.List.sequence(1, 12)
        
        def avg_month(m):
            return daily.filter(ee.Filter.calendarRange(m, m, 'month')) \
                        .sum().divide(end_year - start_year + 1) \
                        .set('month', m) \
                        .rename('avg_precipitation')
                        
        avg_monthly = ee.ImageCollection.fromImages(months.map(avg_month))
        return avg_monthly

    def create_yearly_rainfall(self, aoi: ee.Geometry, start_year: int, end_year: int) -> Dict[int, ee.Image]:
        """
        Create yearly rainfall maps.
        
        Args:
            aoi (ee.Geometry): Area of interest.
            start_year (int): Start year.
            end_year (int): End year.
            
        Returns:
            Dict[int, ee.Image]: Dictionary of yearly precipitation.
        """
        yearly_maps = {}
        for year in range(start_year, end_year + 1):
            yearly_maps[year] = self.compute_annual_rainfall(aoi, year)
        return yearly_maps

    def export_rainfall(self, rainfall: ee.Image, aoi: ee.Geometry, description: str, folder: str = 'chirps', scale: int = 5000) -> ee.batch.Task:
        """
        Export rainfall image to Google Drive.
        
        Args:
            rainfall (ee.Image): Image to export.
            aoi (ee.Geometry): Area of interest.
            description (str): Description/filename.
            folder (str): Drive folder.
            scale (int): Scale in meters.
            
        Returns:
            ee.batch.Task: Export task.
        """
        logger.info(f"Exporting CHIRPS rainfall data: {description} at {scale}m scale.")
        task = ee.batch.Export.image.toDrive(
            image=rainfall,
            description=description,
            folder=folder,
            region=aoi,
            scale=scale,
            maxPixels=1e13
        )
        task.start()
        return task
