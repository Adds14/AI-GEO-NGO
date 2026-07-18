"""
Common raster and geospatial processing utilities.

Provides shared functions for clipping, reprojection, mosaicing,
statistics, and map URL generation for GEE images.
"""
import ee
from loguru import logger
from typing import Dict, List, Optional, Tuple


def clip_to_boundary(image: ee.Image, geometry: ee.Geometry) -> ee.Image:
    """
    Clip an image to a geometry boundary.

    Args:
        image: ee.Image to clip.
        geometry: ee.Geometry boundary.

    Returns:
        Clipped ee.Image.
    """
    logger.debug("Clipping image to boundary.")
    return image.clip(geometry)


def reproject_image(image: ee.Image, crs: str = 'EPSG:4326', scale: float = None) -> ee.Image:
    """
    Reproject an image to a target CRS.

    Args:
        image: ee.Image to reproject.
        crs: Target CRS string.
        scale: Target scale in meters (optional).

    Returns:
        Reprojected ee.Image.
    """
    logger.debug(f"Reprojecting image to {crs}")
    if scale:
        return image.reproject(crs=crs, scale=scale)
    return image.reproject(crs=crs)


def get_image_stats(image: ee.Image, geometry: ee.Geometry, scale: int = 30) -> dict:
    """
    Compute regional statistics (mean, min, max, std) for an image.

    Args:
        image: ee.Image to compute statistics for.
        geometry: ee.Geometry region.
        scale: Scale in meters for reduction.

    Returns:
        Dictionary with band statistics.
    """
    logger.info(f"Computing image statistics at {scale}m scale.")
    try:
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13,
            bestEffort=True
        )
        return stats.getInfo()
    except ee.EEException as e:
        logger.error(f"Failed to compute image statistics: {e}")
        raise


def compute_zonal_stats(image: ee.Image, zones: ee.FeatureCollection,
                        scale: int = 30, band_name: str = None) -> ee.FeatureCollection:
    """
    Compute zonal statistics for each feature in a FeatureCollection.

    Args:
        image: ee.Image to compute statistics for.
        zones: ee.FeatureCollection of zone polygons.
        scale: Scale in meters.
        band_name: Specific band to reduce (optional, uses first band if None).

    Returns:
        ee.FeatureCollection with statistics as properties.
    """
    logger.info("Computing zonal statistics.")
    if band_name:
        image = image.select(band_name)

    def reduce_region(feature):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=feature.geometry(),
            scale=scale,
            maxPixels=1e13,
            bestEffort=True
        )
        return feature.set(stats)

    return zones.map(reduce_region)


def get_map_url(image: ee.Image, vis_params: dict) -> str:
    """
    Generate a tile URL for displaying an image on a web map.

    Args:
        image: ee.Image to visualize.
        vis_params: Visualization parameters dict.

    Returns:
        Tile URL string for use with folium/leaflet.
    """
    try:
        map_id_dict = image.getMapId(vis_params)
        return map_id_dict['tile_fetcher'].url_format
    except ee.EEException as e:
        logger.error(f"Failed to generate map URL: {e}")
        raise


def mosaic_images(images: List[ee.Image]) -> ee.Image:
    """
    Create a mosaic from a list of ee.Images.

    Args:
        images: List of ee.Image objects.

    Returns:
        Mosaicked ee.Image.
    """
    logger.info(f"Creating mosaic from {len(images)} images.")
    collection = ee.ImageCollection(images)
    return collection.mosaic()


def difference_image(image_t2: ee.Image, image_t1: ee.Image, band_name: str, output_name: str = None) -> ee.Image:
    """
    Compute the difference between two images for change detection.

    Formula: difference = image_t2 - image_t1

    Args:
        image_t2: Later time period ee.Image.
        image_t1: Earlier time period ee.Image.
        band_name: Band to compute difference for.
        output_name: Name for the output band (default: '{band_name}_change').

    Returns:
        ee.Image with difference band.
    """
    output_name = output_name or f"{band_name}_change"
    logger.info(f"Computing difference for band '{band_name}' -> '{output_name}'")
    diff = image_t2.select(band_name).subtract(image_t1.select(band_name)).rename(output_name)
    return diff


def normalize_image(image: ee.Image, geometry: ee.Geometry, band_name: str, scale: int = 30) -> ee.Image:
    """
    Normalize an image band to 0-1 range using min-max scaling.

    Args:
        image: ee.Image to normalize.
        geometry: ee.Geometry for computing min/max.
        band_name: Band to normalize.
        scale: Scale for reduction.

    Returns:
        ee.Image with normalized band.
    """
    logger.info(f"Normalizing band '{band_name}' to 0-1 range.")
    stats = image.select(band_name).reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=geometry,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True
    )
    band_min = ee.Number(stats.get(f"{band_name}_min"))
    band_max = ee.Number(stats.get(f"{band_name}_max"))

    normalized = image.select(band_name).subtract(band_min).divide(band_max.subtract(band_min)).rename(band_name)
    return normalized


def export_image(image: ee.Image, aoi: ee.Geometry, description: str,
                 folder: str, scale: int, to: str = 'drive') -> ee.batch.Task:
    """
    Generic export function for any indicator image.

    Args:
        image: ee.Image to export.
        aoi: Area of interest.
        description: Task description.
        folder: Folder name (Drive) or asset path prefix.
        scale: Export scale in meters.
        to: Export destination ('drive' or 'asset').

    Returns:
        ee.batch.Task: Started export task.
    """
    logger.info(f"Exporting image '{description}' to {to} at {scale}m.")
    if to == 'drive':
        task = ee.batch.Export.image.toDrive(
            image=image, description=description, folder=folder,
            region=aoi, scale=scale, maxPixels=1e13, fileFormat='GeoTIFF'
        )
    elif to == 'asset':
        task = ee.batch.Export.image.toAsset(
            image=image, description=description, assetId=f"{folder}/{description}",
            region=aoi, scale=scale, maxPixels=1e13
        )
    else:
        raise ValueError(f"Unknown export destination: {to}. Use 'drive' or 'asset'.")

    task.start()
    logger.info(f"Export task '{description}' started.")
    return task
