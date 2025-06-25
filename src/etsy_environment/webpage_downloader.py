import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
from urllib.parse import urljoin, urlparse
import re
from tqdm import tqdm
import time

class EtsyPageDownloader:
    def __init__(self, progress_position: int = 0, delay: float = 0.0):
        """Create a new *EtsyPageDownloader*.

        Parameters
        ----------
        progress_position:
            "Vertical" offset (starting at ``0``) used by *tqdm* when creating
            progress bars.  Passing a unique value for each concurrent worker
            ensures that their respective bars are rendered on separate lines
            and do not clobber one another.
        delay:
            An optional delay (in seconds) to wait after processing each
            search query.
        """

        self.progress_position = progress_position
        self.delay = delay

        self.session = requests.Session()
        # Use a realistic user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        # Set up base data directory
        self.base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'etsy_pages')
        os.makedirs(self.base_data_dir, exist_ok=True)
    
    def download_page(self, url, save_path):
        """Download a single Etsy page and save all assets"""
        try:
            # Convert relative save_path to absolute path under data directory
            full_save_path = os.path.join(self.base_data_dir, save_path)
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Create directories
            page_dir = os.path.join(full_save_path, 'assets')
            os.makedirs(page_dir, exist_ok=True)
            
            # Download and update CSS files
            self._download_css_files(soup, url, page_dir)
            
            # Download and update JavaScript files
            self._download_js_files(soup, url, page_dir)
            
            # Download and update images
            self._download_images(soup, url, page_dir)
            
            # Save the modified HTML
            html_file = os.path.join(full_save_path, 'index.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            return True
            
        except Exception as e:
            print(f"Error downloading page: {e}")
            return False
    
    def _download_css_files(self, soup, base_url, assets_dir):
        """Download CSS files and update links"""
        css_dir = os.path.join(assets_dir, 'css')
        os.makedirs(css_dir, exist_ok=True)
        
        for link in soup.find_all('link', rel='stylesheet'):
            href = link.get('href')
            if href:
                css_url = urljoin(base_url, href)
                filename = self._get_filename_from_url(css_url, '.css')
                local_path = os.path.join(css_dir, filename)
                
                if self._download_file(css_url, local_path):
                    link['href'] = f'assets/css/{filename}'
    
    def _download_js_files(self, soup, base_url, assets_dir):
        """Download JavaScript files and update links"""
        js_dir = os.path.join(assets_dir, 'js')
        os.makedirs(js_dir, exist_ok=True)
        
        for script in soup.find_all('script', src=True):
            src = script.get('src')
            if src:
                js_url = urljoin(base_url, src)
                filename = self._get_filename_from_url(js_url, '.js')
                local_path = os.path.join(js_dir, filename)
                
                if self._download_file(js_url, local_path):
                    script['src'] = f'assets/js/{filename}'
    
    def _download_images(self, soup, base_url, assets_dir):
        """Download images and update src attributes"""
        img_dir = os.path.join(assets_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        for img in soup.find_all('img', src=True):
            src = img.get('src')
            if src:
                img_url = urljoin(base_url, src)
                filename = self._get_filename_from_url(img_url, '.jpg')
                local_path = os.path.join(img_dir, filename)
                
                if self._download_file(img_url, local_path):
                    img['src'] = f'assets/images/{filename}'
    
    def _download_file(self, url, local_path):
        """Download a file from URL to local path"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False
    
    def _get_filename_from_url(self, url, default_ext):
        """Extract filename from URL or generate one"""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or '.' not in filename:
            filename = f"file_{hash(url) % 10000}{default_ext}"
        
        return filename
    
    def _extract_listing_id(self, url):
        """Extract the numeric listing ID from an Etsy listing URL"""
        match = re.search(r"/listing/(\d+)", url)
        return match.group(1) if match else None

    def _download_search_page_with_listings(self, search_url, save_dir, desc_prefix: str = ""):
        """Download a search results page, all its assets, and every linked listing.
        The links on the saved search page are rewritten so that clicking a listing
        opens the locally saved copy instead of navigating to etsy.com.
        """
        try:
            # Absolute directory where this search page will be stored
            full_save_dir = os.path.join(self.base_data_dir, save_dir)
            os.makedirs(full_save_dir, exist_ok=True)

            response = self.session.get(search_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Keep track so we do not download the same listing twice
            downloaded_ids = set()

            # Find all anchors that link to a listing page
            listing_anchors = [a for a in soup.find_all('a', href=True) if '/listing/' in a['href']]

            # Use a progress bar to show download progress for listings on this search
            # page.  ``position`` ensures that each worker's bar is locked to its
            # own terminal row.
            for anchor in tqdm(
                listing_anchors,
                desc=f"{desc_prefix}Processing...",
                unit="listing",
                position=self.progress_position,
                leave=False,
            ):
                href = anchor['href']

                full_listing_url = urljoin(search_url, href)
                listing_id = self._extract_listing_id(full_listing_url)
                if not listing_id:
                    # Skip anchors that don't contain a valid listing ID
                    continue

                listing_rel_dir = os.path.join(save_dir, f'listing_{listing_id}')

                if listing_id not in downloaded_ids:
                    # Download the listing page and its assets (only once per unique listing)
                    self.download_page(full_listing_url, listing_rel_dir)
                    downloaded_ids.add(listing_id)

                    if self.delay > 0:
                        time.sleep(self.delay)

                # Update the anchor so it points to the local copy
                anchor['href'] = f'listing_{listing_id}/index.html'

            # Download assets (CSS, JS, images) referenced in the search page itself
            assets_dir = os.path.join(full_save_dir, 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            self._download_css_files(soup, search_url, assets_dir)
            self._download_js_files(soup, search_url, assets_dir)
            self._download_images(soup, search_url, assets_dir)

            # Save the modified HTML
            html_file = os.path.join(full_save_dir, 'index.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(str(soup))

            return True

        except Exception as e:
            print(f"Error downloading search page and listings: {e}")
            return False

    def download_search_results(self, search_query, num_pages=1, desc_prefix: str = ""):
        """Download Etsy search results pages"""
        base_search_url = "https://www.etsy.com/search"
        
        for page in range(1, num_pages + 1):
            params = {
                'q': search_query,
                'page': page
            }
            
            search_url = f"{base_search_url}?{urllib.parse.urlencode(params)}"
            page_dir = f"search_{search_query.replace(' ', '_')}_page_{page}"
            
            self._download_search_page_with_listings(search_url, page_dir, desc_prefix=desc_prefix)

# Usage example
if __name__ == "__main__":
    downloader = EtsyPageDownloader()
    
    # Download search results for a specific query
    search_query = "inflatable halloween spider"
    downloader.download_search_results(search_query, num_pages=1)
    
    # Or download a specific product page
    # product_url = "https://www.etsy.com/listing/123456789/some-product"
    # downloader.download_page(product_url, "product_123456789")