"""
Geospatial utility functions for bounding boxes, grids, and CRS.
"""
from typing import Tuple, List
import ee
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import transform
import pyproj
import geopandas as gpd

def reproject_geometry(geom, src_crs: str, dst_crs: str):
    """Reproject a shapely geometry from src_crs to dst_crs."""
    project = pyproj.Transformer.from_crs(
        pyproj.CRS(src_crs), 
        pyproj.CRS(dst_crs), 
        always_xy=True
    ).transform
    return transform(project, geom)

def get_bbox(geometry) -> Tuple[float, float, float, float]:
    """Get bounding box of a geometry."""
    return geometry.bounds

def geometry_to_ee_geometry(shapely_geom) -> ee.Geometry:
    """Convert shapely geometry to Earth Engine geometry."""
    geom_mapping = mapping(shapely_geom)
    return ee.Geometry(geom_mapping)

def ee_geometry_to_shapely(ee_geom):
    """Convert Earth Engine geometry to shapely geometry."""
    return shape(ee_geom.getInfo())

def create_grid(boundary: Polygon, cell_size: float) -> List[Polygon]:
    """
    Create a grid of polygons within the boundary.
    cell_size should be in the units of the boundary's CRS.
    """
    minx, miny, maxx, maxy = boundary.bounds
    grid_cells = []
    
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = Polygon([
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y + cell_size),
                (x, y + cell_size)
            ])
            if cell.intersects(boundary):
                grid_cells.append(cell.intersection(boundary))
            y += cell_size
        x += cell_size
        
    return grid_cells

def calculate_area_km2(geometry: Polygon) -> float:
    """
    Calculate area in square kilometers.
    Assumes geometry is in EPSG:4326 and reprojects to an equal-area projection temporarily.
    """
    gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geometry])
    # Reproject to an equal area projection (e.g. EPSG:6933)
    gdf_ea = gdf.to_crs('epsg:6933')
    return gdf_ea.geometry.area.iloc[0] / 1e6
