"""
Administrative boundary loading and processing module.
"""
import ee
import geopandas as gpd
import json
from loguru import logger
from typing import Union, List, Any

class BoundaryLoader:
    """Loader and processor for administrative boundaries."""

    def __init__(self):
        """Initialize BoundaryLoader."""
        logger.debug("Initialized BoundaryLoader.")

    def load_from_shapefile(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """
        Load shapefile into a GeoDataFrame.
        
        Args:
            shapefile_path (str): Path to the shapefile.
            
        Returns:
            gpd.GeoDataFrame: Loaded GeoDataFrame.
        """
        logger.info(f"Loading shapefile from {shapefile_path}.")
        try:
            gdf = gpd.read_file(shapefile_path)
            return self.validate_boundaries(gdf)
        except Exception as e:
            logger.error(f"Failed to load shapefile: {e}")
            raise

    def load_from_geojson(self, geojson_path: str) -> gpd.GeoDataFrame:
        """
        Load GeoJSON into a GeoDataFrame.
        
        Args:
            geojson_path (str): Path to the GeoJSON file.
            
        Returns:
            gpd.GeoDataFrame: Loaded GeoDataFrame.
        """
        logger.info(f"Loading GeoJSON from {geojson_path}.")
        try:
            gdf = gpd.read_file(geojson_path)
            return self.validate_boundaries(gdf)
        except Exception as e:
            logger.error(f"Failed to load GeoJSON: {e}")
            raise

    def load_from_gee(self, asset_id: str, region_name: str = None) -> ee.FeatureCollection:
        """
        Load boundary from a GEE FeatureCollection.
        
        Args:
            asset_id (str): Earth Engine asset ID (e.g., 'FAO/GAUL/2015/level1').
            region_name (str, optional): Specific region name to filter by.
            
        Returns:
            ee.FeatureCollection: Earth Engine FeatureCollection.
        """
        logger.info(f"Loading GEE FeatureCollection: {asset_id}.")
        try:
            fc = ee.FeatureCollection(asset_id)
            if region_name:
                # Assuming standard 'ADM1_NAME' or similar column, filtering might need adjustment based on dataset
                fc = fc.filter(ee.Filter.eq('ADM1_NAME', region_name))
            return fc
        except ee.EEException as e:
            logger.error(f"Failed to load GEE Asset {asset_id}: {e}")
            raise

    def get_aoi(self, geometry_or_gdf: Union[gpd.GeoDataFrame, ee.FeatureCollection, Any]) -> ee.Geometry:
        """
        Convert input to ee.Geometry for GEE queries.
        
        Args:
            geometry_or_gdf: GeoDataFrame, dict (GeoJSON), or ee.FeatureCollection.
            
        Returns:
            ee.Geometry: Earth Engine Geometry.
        """
        logger.info("Converting input to ee.Geometry.")
        try:
            if isinstance(geometry_or_gdf, ee.FeatureCollection):
                return geometry_or_gdf.geometry()
            elif isinstance(geometry_or_gdf, gpd.GeoDataFrame):
                # Ensure CRS is EPSG:4326
                gdf = self.validate_boundaries(geometry_or_gdf)
                # Convert geometry to GeoJSON dict
                geojson_dict = json.loads(gdf.to_json())
                # Extract features
                features = []
                for feature in geojson_dict.get('features', []):
                    features.append(ee.Feature(feature['geometry']))
                return ee.FeatureCollection(features).geometry()
            else:
                # Assume it's a dict representing GeoJSON
                return ee.Geometry(geometry_or_gdf)
        except Exception as e:
            logger.error(f"Failed to convert to AOI: {e}")
            raise

    def validate_boundaries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Check CRS, fix invalid geometries, and ensure EPSG:4326.
        
        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            
        Returns:
            gpd.GeoDataFrame: Validated and reprojected GeoDataFrame.
        """
        logger.info("Validating boundaries (fixing geometries and CRS).")
        # Fix invalid geometries
        if not gdf.is_valid.all():
            logger.warning("Found invalid geometries. Attempting to fix with buffer(0).")
            gdf['geometry'] = gdf['geometry'].buffer(0)
            
        # Ensure CRS is WGS84 (EPSG:4326)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            logger.info("Reprojecting to EPSG:4326.")
            gdf = gdf.to_crs(epsg=4326)
            
        return gdf

    def create_grid(self, gdf: gpd.GeoDataFrame, cell_size_deg: float = 0.1) -> gpd.GeoDataFrame:
        """
        Create a regular grid over the AOI.
        
        Args:
            gdf (gpd.GeoDataFrame): Boundary GeoDataFrame.
            cell_size_deg (float): Cell size in degrees.
            
        Returns:
            gpd.GeoDataFrame: Grid polygons covering the boundary.
        """
        logger.info(f"Creating regular grid with cell size {cell_size_deg} degrees.")
        import numpy as np
        from shapely.geometry import box
        
        validated_gdf = self.validate_boundaries(gdf)
        minx, miny, maxx, maxy = validated_gdf.total_bounds
        
        # Create grid cells
        x_coords = np.arange(minx, maxx, cell_size_deg)
        y_coords = np.arange(miny, maxy, cell_size_deg)
        
        grid_cells = []
        for x in x_coords:
            for y in y_coords:
                grid_cells.append(box(x, y, x + cell_size_deg, y + cell_size_deg))
                
        grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:4326")
        
        # Intersect with original boundary
        intersected = gpd.overlay(grid_gdf, validated_gdf, how='intersection')
        return intersected

    def get_region_list(self, gdf: gpd.GeoDataFrame, name_column: str) -> List[str]:
        """
        Return a list of region names from a specific column.
        
        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame.
            name_column (str): Column containing region names.
            
        Returns:
            List[str]: List of names.
        """
        if name_column not in gdf.columns:
            logger.error(f"Column '{name_column}' not found in GeoDataFrame.")
            raise ValueError(f"Column '{name_column}' not found.")
        return gdf[name_column].unique().tolist()

    def to_ee_feature_collection(self, gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
        """
        Convert GeoDataFrame to ee.FeatureCollection.
        
        Args:
            gdf (gpd.GeoDataFrame): Input GeoDataFrame.
            
        Returns:
            ee.FeatureCollection: Earth Engine FeatureCollection.
        """
        logger.info("Converting GeoDataFrame to ee.FeatureCollection.")
        validated_gdf = self.validate_boundaries(gdf)
        geojson_dict = json.loads(validated_gdf.to_json())
        
        features = []
        for feature in geojson_dict.get('features', []):
            geom = feature['geometry']
            props = feature['properties']
            features.append(ee.Feature(geom, props))
            
        return ee.FeatureCollection(features)
