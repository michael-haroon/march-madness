import asyncio
import os
import re
from playwright.async_api import async_playwright

# Your specific path
TARGET_DIR = "/Users/michaelharoon/Projects/tasty/march-madness/data"
os.makedirs(TARGET_DIR, exist_ok=True)

async def get_page_metadata(page):
    """Extracts the Stat Name, Season, and Selected Date from the UI."""
    try:
        # 1. Target the specific header inside the national_ranking_div
        # This bypasses the navigation tabs at the top
        stat_header_el = page.locator("#national_ranking_div .card-header").first
        
        header_text = await stat_header_el.inner_text() if await stat_header_el.count() > 0 else "Unknown_Stat"
        
        # Clean up: strip whitespace and handle potential newlines
        clean_header = header_text.strip().split('\n')[0]

        # 2. Get the Season from the page title (it's the most consistent place for it)
        page_title = await page.title()
        season_match = re.search(r'\d{4}-\d{2}', page_title)
        season = season_match.group(0) if season_match else ""

        # 3. Get the specific date selected in the dropdown
        date_text = await page.evaluate("() => document.querySelector('#rp option:checked')?.textContent")
        formatted_date = "No-Date"
        if date_text:
            date_match = re.search(r'(\d{2})/(\d{2})/(\d{4})', date_text)
            if date_match:
                formatted_date = f"{date_match.group(3)}-{date_match.group(1)}-{date_match.group(2)}"
            else:
                formatted_date = re.sub(r'\W+', '_', date_text.strip())

        # Combine: e.g., "Assist_Turnover_Ratio_2024-25_2025-03-29"
        final_name = f"{clean_header}_{season}_{formatted_date}"
        return final_name
        
    except Exception as e:
        print(f"Metadata Error: {e}")
        return "NCAA_Stat_Download"

async def automate_download(page):
    print("\n--- Watchdog Active ---")
    print(f"Target Directory: {TARGET_DIR}")
    print("1. Navigate to a stat page.")
    print("2. Select your desired date.")
    print("3. I will auto-download and name the file.")
    
    last_processed_key = ""

    while True:
        try:
            # We check for the Excel button
            excel_btn = page.locator("a.buttons-excel")
            
            if await excel_btn.count() > 0:
                # Create a unique key based on URL + current selected date index
                current_rp = await page.input_value("#rp") if await page.query_selector("#rp") else ""
                current_key = f"{page.url}_{current_rp}"

                if current_key != last_processed_key:
                    # Brief wait to ensure DataTables has updated the button's internal link
                    await asyncio.sleep(0.5)
                    
                    filename_base = await get_page_metadata(page)
                    # Sanitize for filesystem (replace / and spaces)
                    safe_filename = re.sub(r'[^\w\-_\. ]', '_', filename_base) + ".xls"
                    
                    print(f"Detected: {filename_base}. Downloading...")

                    async with page.expect_download() as download_info:
                        await excel_btn.first.click()
                    
                    download = await download_info.value
                    save_path = os.path.join(TARGET_DIR, safe_filename)
                    
                    await download.save_as(save_path)
                    print(f"✅ Saved: {safe_filename}")
                    
                    last_processed_key = current_key
            
            await asyncio.sleep(1.5)
            
        except Exception as e:
            await asyncio.sleep(1)

async def main():
    async with async_playwright() as p:
        # Launching with a specific user data dir can sometimes help stay logged in/bypass masks
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto("https://stats.ncaa.org/rankings/change_sport_year_div")
        await automate_download(page)

if __name__ == "__main__":
    asyncio.run(main())