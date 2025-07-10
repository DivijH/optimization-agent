import asyncio
from typing import TYPE_CHECKING, Optional

from google.cloud import storage

if TYPE_CHECKING:
    from src.shopping_agent.agent import EtsyShoppingAgent


class GCSManager:
    def __init__(self, agent: "EtsyShoppingAgent"):
        self.agent = agent
        self._gcs_client: Optional[storage.Client] = None

    def get_gcs_client(self) -> Optional[storage.Client]:
        """Initialize and return GCS client if GCS saving is enabled."""
        if not self.agent.save_gcs:
            return None

        if self._gcs_client is None:
            try:
                self._gcs_client = storage.Client()
            except Exception as e:
                self.agent._log(f"Failed to initialize GCS client: {e}", level="error")
                return None

        return self._gcs_client

    async def upload_to_gcs(self, local_file_path: str, gcs_file_path: str):
        """Upload a file to Google Cloud Storage."""
        if not self.agent.save_gcs:
            return

        client = self.get_gcs_client()
        if not client:
            return

        try:
            bucket = client.bucket(self.agent.gcs_bucket_name)
            blob = bucket.blob(f"{self.agent.gcs_prefix}/{gcs_file_path}")

            # Run the blocking upload operation in a separate thread
            await asyncio.to_thread(blob.upload_from_filename, local_file_path)
            self.agent._log(
                f"   - Uploaded to GCS: gs://{self.agent.gcs_bucket_name}/{self.agent.gcs_prefix}/{gcs_file_path}"
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

        try:
            bucket = client.bucket(self.agent.gcs_bucket_name)
            blob = bucket.blob(f"{self.agent.gcs_prefix}/{gcs_file_path}")

            # Run the blocking upload operation in a separate thread
            await asyncio.to_thread(
                blob.upload_from_string, content, content_type=content_type
            )
            self.agent._log(
                f"   - Uploaded to GCS: gs://{self.agent.gcs_bucket_name}/{self.agent.gcs_prefix}/{gcs_file_path}"
            )
        except Exception as e:
            self.agent._log(f"   - Failed to upload to GCS: {e}", level="error") 