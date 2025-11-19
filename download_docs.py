#!/usr/bin/env python3
"""
Download LangChain and LangGraph documentation files.

This script downloads:
- llms.txt (resource map)
- llms-full.txt (reference documentation)
- All markdown files referenced in llms.txt

Logs download results to data/download_log.txt for troubleshooting.
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
from tqdm.asyncio import tqdm


# URLs to download
LLMS_TXT_URL = "https://docs.langchain.com/llms.txt"
LLMS_FULL_TXT_URL = "https://docs.langchain.com/llms-full.txt"

# Output directories
DATA_DIR = Path(__file__).parent / "data"
DOCS_DIR = DATA_DIR / "docs"
LOG_FILE = DATA_DIR / "download_log.txt"

# Request settings
TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_CONCURRENT = 15  # Limit concurrent downloads


async def download_file_async(
    session: aiohttp.ClientSession,
    url: str,
    output_path: Path,
    semaphore: asyncio.Semaphore,
    progress_bar: tqdm = None,
    retries: int = MAX_RETRIES
) -> bool:
    """
    Download a file from URL to output_path asynchronously.

    Returns:
        True if successful, False otherwise
    """
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as response:
                    if response.status == 404:
                        if progress_bar:
                            progress_bar.write(f"  ✗ File not found (404): {url}")
                        return False

                    response.raise_for_status()
                    content = await response.read()

                    # Ensure parent directory exists
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write content
                    output_path.write_bytes(content)
                    if progress_bar:
                        progress_bar.update(1)
                    return True

            except asyncio.TimeoutError:
                if progress_bar:
                    progress_bar.write(f"  ✗ Timeout: {output_path.name} (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    await asyncio.sleep(RETRY_DELAY)

            except aiohttp.ClientError as e:
                if progress_bar:
                    progress_bar.write(f"  ✗ Error: {output_path.name} - {e} (attempt {attempt + 1}/{retries})")
                if attempt < retries - 1:
                    await asyncio.sleep(RETRY_DELAY)

        if progress_bar:
            progress_bar.write(f"  ✗ Failed after {retries} attempts: {output_path.name}")
        return False


def parse_llms_txt(content: str) -> list[str]:
    """
    Parse llms.txt content to extract markdown URLs.

    llms.txt format:
    - Lines starting with '# ' are section headers (ignore)
    - Lines with markdown links: '- [Title](URL)'
    - We extract the URL from the link
    """
    urls = []
    # Pattern to match markdown links: [text](url)
    pattern = r'\[.*?\]\((.*?)\)'

    for line in content.splitlines():
        line = line.strip()

        # Skip empty lines and headers (but not list items)
        if not line or line.startswith('# '):
            continue

        # Find all markdown links in the line
        matches = re.findall(pattern, line)
        for url in matches:
            # Only include markdown files
            if url.endswith('.md'):
                urls.append(url)

    return urls


async def main():
    """Download all documentation files."""
    print("=" * 60)
    print("Documentation Download")
    print("=" * 60)
    print()

    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    print(f"Data directory: {DATA_DIR}")
    print(f"Docs directory: {DOCS_DIR}")
    print(f"Log file: {LOG_FILE}")
    print()

    # Initialize log file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entries = [
        f"{'=' * 60}",
        f"Documentation Download Log",
        f"Timestamp: {timestamp}",
        f"{'=' * 60}",
        ""
    ]

    # Create aiohttp session for all downloads
    async with aiohttp.ClientSession() as session:
        # Download initial files sequentially (using semaphore=1)
        sequential_semaphore = asyncio.Semaphore(1)

        # Download llms.txt
        print("[1/3] Downloading llms.txt...")
        llms_txt_path = DATA_DIR / "llms.txt"
        if not await download_file_async(session, LLMS_TXT_URL, llms_txt_path, sequential_semaphore):
            print("\n✗ Failed to download llms.txt. Cannot continue.")
            raise SystemExit(1)
        print("  ✓ Saved: llms.txt")
        print()

        # Download llms-full.txt (reference, optional)
        print("[2/3] Downloading llms-full.txt (reference)...")
        llms_full_path = DATA_DIR / "llms-full.txt"
        await download_file_async(session, LLMS_FULL_TXT_URL, llms_full_path, sequential_semaphore)
        print("  ✓ Saved: llms-full.txt")
        print()

        # Parse llms.txt to get markdown URLs
        print("[3/3] Downloading markdown files from llms.txt...")
        llms_txt_content = llms_txt_path.read_text(encoding='utf-8')
        markdown_urls = parse_llms_txt(llms_txt_content)

        if not markdown_urls:
            print("  ✗ No markdown URLs found in llms.txt")
            raise SystemExit(1)

        print(f"  Found {len(markdown_urls)} markdown files to download")
        print(f"  Downloading with {MAX_CONCURRENT} concurrent connections...")
        print()

        # Prepare download tasks
        download_tasks = []
        failed_urls = []

        # Create semaphore to limit concurrent downloads
        parallel_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        for url in markdown_urls:
            # Convert relative URLs to absolute if needed
            if not url.startswith('http'):
                base_url = LLMS_TXT_URL.rsplit('/', 1)[0] + '/'
                url = urljoin(base_url, url)

            # Extract full path from URL to preserve directory structure
            parsed = urlparse(url)
            relative_path = parsed.path.lstrip('/')
            output_path = DOCS_DIR / relative_path

            download_tasks.append((url, output_path))

        # Execute all downloads in parallel with progress bar
        with tqdm(total=len(download_tasks), desc="Downloading", unit="file") as progress_bar:
            tasks = [
                download_file_async(session, url, output_path, parallel_semaphore, progress_bar)
                for url, output_path in download_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        success_count = sum(1 for result in results if result is True)
        for i, result in enumerate(results):
            if result is not True:
                failed_urls.append(download_tasks[i][0])

    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Total files: {len(markdown_urls)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_urls)}")

    # Add summary to log
    log_entries.extend([
        "",
        "Summary:",
        f"  Total files attempted: {len(markdown_urls)}",
        f"  Successful downloads: {success_count}",
        f"  Failed downloads: {len(failed_urls)}",
        ""
    ])

    if failed_urls:
        print("\nFailed URLs:")
        log_entries.append("Failed URLs:")
        for url in failed_urls:
            print(f"  - {url}")
            log_entries.append(f"  - {url}")
        log_entries.append("")

    # Write log file
    LOG_FILE.write_text('\n'.join(log_entries), encoding='utf-8')

    print()
    print("✓ Documentation download complete!")
    print(f"  Files saved to: {DOCS_DIR}")
    print(f"  Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
