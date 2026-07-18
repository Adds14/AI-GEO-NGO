"""
GIS Visualization Utilities for Streamlit Dashboard.

Provides reusable Folium map components for rendering environmental
indicators, climate vulnerability, and WASH priority scores.
"""
import folium
import geopandas as gpd
import pandas as pd
from folium import plugins
from folium.features import GeoJsonTooltip, GeoJsonPopup
import json


def create_base_map(center_lat: float = 0.0, center_lon: float = 0.0, zoom_start: int = 5) -> folium.Map:
    """
    Initialize a Folium map with multiple basemaps.
    """
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True)
    
    # Add alternative basemaps
    folium.TileLayer('cartodbpositron', name='Light Map').add_to(m)
    folium.TileLayer('cartodbdark_matter', name='Dark Map').add_to(m)
    
    # Esri Satellite fallback
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite Imagery',
        overlay=False,
        control=True
    ).add_to(m)
    
    return m


def add_choropleth(m: folium.Map, 
                   gdf: gpd.GeoDataFrame, 
                   value_col: str, 
                   name: str, 
                   cmap: str = 'YlOrRd', 
                   legend_name: str = '',
                   show: bool = True):
    """
    Add a choropleth layer to the map.
    
    Args:
        m: Folium map instance.
        gdf: GeoDataFrame containing geometries and data.
        value_col: Column name to use for coloring.
        name: Name of the layer for the LayerControl.
        cmap: Color palette.
        legend_name: Text for the legend.
        show: Whether the layer is visible by default.
    """
    if value_col not in gdf.columns:
        return m
        
    choropleth = folium.Choropleth(
        geo_data=gdf,
        name=name,
        data=gdf,
        columns=[gdf.index if 'geographic_id' not in gdf.columns else 'geographic_id', value_col],
        key_on=f"feature.properties.geographic_id",
        fill_color=cmap,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
        show=show,
        nan_fill_color='white',
        highlight=True
    ).add_to(m)
    
    # Add interactive tooltips/popups to the choropleth geojson layer
    tooltip_cols = ['geographic_id', value_col]
    aliases = ['Region ID:', f'{legend_name}:']
    
    # Add extra context columns if they exist
    extra_cols = ['priority_class', 'vulnerability_category', 'explanation']
    for col in extra_cols:
        if col in gdf.columns:
            tooltip_cols.append(col)
            aliases.append(f"{col.replace('_', ' ').title()}:")
            
    tooltip = GeoJsonTooltip(
        fields=tooltip_cols,
        aliases=aliases,
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
    
    popup = GeoJsonPopup(
        fields=tooltip_cols,
        aliases=aliases,
        localize=True,
        labels=True,
        style="background-color: white;",
    )

    choropleth.geojson.add_child(tooltip)
    choropleth.geojson.add_child(popup)
    
    return m


def add_gee_layer(m: folium.Map, gee_map_id: dict, name: str, show: bool = False):
    """
    Add a Google Earth Engine TileLayer to the map.
    
    Args:
        m: Folium map instance.
        gee_map_id: Dictionary returned by ee.Image.getMapId() containing 'tile_fetcher'.
        name: Layer name.
        show: Visibility toggle.
    """
    folium.TileLayer(
        tiles=gee_map_id['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True,
        show=show
    ).add_to(m)
    return m


def build_integrated_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    """
    Build the complete interactive map with all requested layers.
    
    Args:
        gdf: GeoDataFrame containing all metrics (NDVI, Priority, Vuln, etc.).
        
    Returns:
        folium.Map instance.
    """
    # Calculate center
    if not gdf.empty:
        bounds = gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    else:
        center_lat, center_lon = 0, 0
        
    m = create_base_map(center_lat, center_lon, zoom_start=6)
    
    # 1. WASH Priority (Primary Layer)
    if 'priority_score' in gdf.columns:
        add_choropleth(m, gdf, 'priority_score', 'WASH Priority Score', 'YlOrRd', 'WASH Priority (0-100)', show=True)
        
    # 2. Climate Vulnerability
    if 'vulnerability_score' in gdf.columns:
        add_choropleth(m, gdf, 'vulnerability_score', 'Climate Vulnerability', 'Reds', 'Vulnerability Score (0-1)', show=False)
        
    # 3. Environmental Indicators (if present)
    indicators = {
        'ndvi': ('NDVI (Vegetation)', 'YlGn'),
        'ndwi': ('NDWI (Water)', 'Blues'),
        'lst': ('Land Surface Temp', 'OrRd'),
        'vegetation_change': ('Vegetation Change', 'RdYlGn'),
        'urban_growth_rate': ('Urban Expansion', 'Purples')
    }
    
    for col, (name, cmap) in indicators.items():
        if col in gdf.columns:
            # For vegetation change, we might want a diverging colormap centered on 0
            add_choropleth(m, gdf, col, name, cmap, f'{name}', show=False)
            
    # Add fullscreen control
    plugins.Fullscreen(position='topright').add_to(m)
    
    # Add Layer Control
    folium.LayerControl(position='topright').add_to(m)
    
    return m
