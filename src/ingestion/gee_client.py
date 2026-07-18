"""
Google Earth Engine Client configuration and session management.
"""
import os
import ee
from loguru import logger

class GEEClient:
    """Client for managing Google Earth Engine authentication and initialization."""

    def __init__(self, project_id: str = None, service_account: str = None, key_file: str = None):
        """
        Initialize the GEEClient.

        Args:
            project_id (str, optional): GEE Project ID. Defaults to env var GEE_PROJECT_ID.
            service_account (str, optional): Service account email. Defaults to env var GEE_SERVICE_ACCOUNT_EMAIL.
            key_file (str, optional): Path to JSON key file. Defaults to env var GEE_KEY_FILE.
        """
        self.project_id = project_id or os.getenv('GEE_PROJECT_ID')
        self.service_account = service_account or os.getenv('GEE_SERVICE_ACCOUNT_EMAIL')
        self.key_file = key_file or os.getenv('GEE_KEY_FILE')
        self._is_authenticated = False

    def authenticate(self) -> None:
        """
        Authenticate with Google Earth Engine.
        Tries service account authentication first, then falls back to interactive authentication.
        
        Raises:
            ee.EEException: If authentication fails.
        """
        try:
            if self.service_account and self.key_file:
                logger.info(f"Authenticating with service account: {self.service_account}")
                ee.Initialize(ee.ServiceAccountCredentials(self.service_account, self.key_file), project=self.project_id)
                self._is_authenticated = True
                logger.success("Successfully authenticated with service account.")
            else:
                logger.info("Service account credentials not fully provided. Falling back to interactive authentication.")
                ee.Authenticate()
                self._is_authenticated = True
                logger.success("Successfully authenticated interactively.")
        except ee.EEException as e:
            logger.error(f"Earth Engine authentication failed: {e}")
            raise

    def initialize(self) -> None:
        """
        Initialize the Google Earth Engine session.
        
        Raises:
            ee.EEException: If initialization fails.
        """
        try:
            if self.project_id:
                logger.info(f"Initializing Earth Engine session for project: {self.project_id}")
                ee.Initialize(project=self.project_id)
            else:
                logger.info("Initializing Earth Engine session (no project specified).")
                ee.Initialize()
            self._is_authenticated = True
            logger.success("Earth Engine initialized successfully.")
        except ee.EEException as e:
            logger.error(f"Earth Engine initialization failed: {e}")
            raise

    def is_authenticated(self) -> bool:
        """
        Check if the Earth Engine session is active/authenticated.
        
        Returns:
            bool: True if authenticated, False otherwise.
        """
        return self._is_authenticated

    def get_info(self) -> dict:
        """
        Return Google Earth Engine session info.
        
        Returns:
            dict: Dictionary containing session information.
        """
        return {
            'is_authenticated': self.is_authenticated(),
            'project_id': self.project_id,
            'service_account': self.service_account
        }

_gee_client_instance = None

def get_gee_client() -> GEEClient:
    """
    Returns a singleton instance of the GEEClient.
    
    Returns:
        GEEClient: Singleton instance.
    """
    global _gee_client_instance
    if _gee_client_instance is None:
        _gee_client_instance = GEEClient()
    return _gee_client_instance
