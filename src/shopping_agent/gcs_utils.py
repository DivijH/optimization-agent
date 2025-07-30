import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from google.cloud import storage

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


class GCSManager:
    def __init__(self, agent: "EtsyShoppingAgent", project: Optional[str] = None):
        self.agent = agent
        self.project = project or getattr(agent, 'gcs_project', 'etsy-search-ml-dev')
        self._gcs_client: Optional[storage.Client] = None

    def get_gcs_client(self) -> Optional[storage.Client]:
        """Initialize and return GCS client if GCS saving is enabled."""
        if not self.agent.save_gcs:
            return None

        if self._gcs_client is None:
            try:
                self._gcs_client = storage.Client(project=self.project)
            except Exception as e:
                self.agent._log(f"Failed to initialize GCS client: {e}", level="error")
                return None

        return self._gcs_client

    def _normalize_gcs_path(self, path: str) -> str:
        """Convert absolute local path to relative GCS path."""
        if not path:
            return path
        
        # Convert to Path object for easier manipulation
        path_obj = Path(path)
        
        # If it's an absolute path, try to make it relative to project root
        if path_obj.is_absolute():
            # Find the project root (look for 'optimization-agent' in the path)
            path_parts = path_obj.parts
            try:
                # Find the index of 'optimization-agent' directory
                opt_agent_idx = path_parts.index('optimization-agent')
                # Take everything after 'optimization-agent' 
                relative_parts = path_parts[opt_agent_idx + 1:]
                
                # Remove 'src' prefix if it's the first component
                if relative_parts and relative_parts[0] == 'src':
                    relative_parts = relative_parts[1:]
                
                relative_path = str(Path(*relative_parts)) if relative_parts else ""
                return relative_path
            except ValueError:
                # If 'optimization-agent' not found, just use the filename
                self.agent._log(f"Warning: Could not find project root in path {path}, using filename only", level="warning")
                return path_obj.name
        
        # If already relative, return as-is
        return str(path_obj)

    async def upload_to_gcs(self, local_file_path: str, gcs_file_path: str):
        """Upload a file to Google Cloud Storage."""
        if not self.agent.save_gcs:
            return

        client = self.get_gcs_client()
        if not client:
            return

        # Normalize the GCS path to remove absolute paths
        normalized_gcs_path = self._normalize_gcs_path(gcs_file_path)

        try:
            bucket = client.bucket(self.agent.gcs_bucket_name)
            blob = bucket.blob(f"{self.agent.gcs_prefix}/{normalized_gcs_path}")

            # Run the blocking upload operation in a separate thread
            await asyncio.to_thread(blob.upload_from_filename, local_file_path)
            self.agent._log(
                f"   - Uploaded to GCS: gs://{self.agent.gcs_bucket_name}/{self.agent.gcs_prefix}/{normalized_gcs_path}"
            )
        except Exception as e:
            self.agent._log(f"   - Failed to upload to GCS: {e}", level="error")

    async def upload_string_to_gcs(
        self, content: str, gcs_file_path: str, content_type: str = "application/json"
    ):
        """Upload string content to Google Cloud Storage."""
        if not self.agent.save_gcs:
            return

        client = self.get_gcs_client()
        if not client:
            return

        # Normalize the GCS path to remove absolute paths
        normalized_gcs_path = self._normalize_gcs_path(gcs_file_path)

        try:
            bucket = client.bucket(self.agent.gcs_bucket_name)
            blob = bucket.blob(f"{self.agent.gcs_prefix}/{normalized_gcs_path}")

            # Run the blocking upload operation in a separate thread
            await asyncio.to_thread(
                blob.upload_from_string, content, content_type=content_type
            )
            self.agent._log(
                f"   - Uploaded to GCS: gs://{self.agent.gcs_bucket_name}/{self.agent.gcs_prefix}/{normalized_gcs_path}"
            )
        except Exception as e:
            self.agent._log(f"   - Failed to upload to GCS: {e}", level="error")


def normalize_gcs_path(path: str) -> str:
    """
    Standalone function to convert absolute local path to relative GCS path.
    Can be used by files that don't have access to GCSManager.
    """
    if not path:
        return path
    
    # Convert to Path object for easier manipulation
    path_obj = Path(path)
    
    # If it's an absolute path, try to make it relative to project root
    if path_obj.is_absolute():
        # Find the project root (look for 'optimization-agent' in the path)
        path_parts = path_obj.parts
        try:
            # Find the index of 'optimization-agent' directory
            opt_agent_idx = path_parts.index('optimization-agent')
            # Take everything after 'optimization-agent' 
            relative_parts = path_parts[opt_agent_idx + 1:]
            
            # Remove 'src' prefix if it's the first component
            if relative_parts and relative_parts[0] == 'src':
                relative_parts = relative_parts[1:]
            
            relative_path = str(Path(*relative_parts)) if relative_parts else ""
            return relative_path
        except ValueError:
            # If 'optimization-agent' not found, just use the filename
            return path_obj.name
    
    # If already relative, return as-is
    return str(path_obj)


async def upload_file_to_gcs(
    local_file_path: str, 
    gcs_file_path: str, 
    bucket_name: str, 
    gcs_prefix: str, 
    project: str = 'etsy-search-ml-dev'
):
    """
    Standalone function to upload a file to GCS.
    Can be used by files that don't have access to GCSManager.
    """
    try:
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        
        # Normalize the GCS path
        normalized_gcs_path = normalize_gcs_path(gcs_file_path)
        blob = bucket.blob(f"{gcs_prefix}/{normalized_gcs_path}")
        
        # Upload the file
        await asyncio.to_thread(blob.upload_from_filename, local_file_path)
        print(f"   - Uploaded to GCS: gs://{bucket_name}/{gcs_prefix}/{normalized_gcs_path}")
        
    except Exception as e:
        print(f"   - Failed to upload to GCS: {e}")


async def upload_string_to_gcs(
    content: str,
    gcs_file_path: str, 
    bucket_name: str, 
    gcs_prefix: str, 
    project: str = 'etsy-search-ml-dev',
    content_type: str = "application/json"
):
    """
    Standalone function to upload string content to GCS.
    Can be used by files that don't have access to GCSManager.
    """
    try:
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        
        # Normalize the GCS path
        normalized_gcs_path = normalize_gcs_path(gcs_file_path)
        blob = bucket.blob(f"{gcs_prefix}/{normalized_gcs_path}")
        
        # Upload the content
        await asyncio.to_thread(blob.upload_from_string, content, content_type=content_type)
        print(f"   - Uploaded to GCS: gs://{bucket_name}/{gcs_prefix}/{normalized_gcs_path}")
        
    except Exception as e:
        print(f"   - Failed to upload to GCS: {e}")