import asyncio
from playwright.async_api import async_playwright
async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_viewport_size({'width': 1200, 'height': 800})
        await page.goto('http://localhost:42069')
        await asyncio.sleep(2)
        await page.screenshot(path='C:/Users/wjohn/.gemini/antigravity/brain/c1faebd6-db48-4c90-ae49-46f335680b22/dark_electric_purple_dashboard.png')
        
        # Test input clear
        await page.fill('#search-input', 'test data')
        await asyncio.sleep(1)
        await page.screenshot(path='C:/Users/wjohn/.gemini/antigravity/brain/c1faebd6-db48-4c90-ae49-46f335680b22/dark_electric_purple_search_filled.png')
        
        await browser.close()
asyncio.run(run())
