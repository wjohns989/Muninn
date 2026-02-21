import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_viewport_size({'width': 1200, 'height': 800})
        await page.goto('http://localhost:42069')
        await asyncio.sleep(2)  # Give time for the runic watermark and fonts to render
        await page.screenshot(path='C:/Users/user/.gemini/antigravity/brain/c1faebd6-db48-4c90-ae49-46f335680b22/dark_electric_purple_dashboard.png')

        # Test search clear by typing
        await page.fill('#search-input', 'test data')
        await asyncio.sleep(1)
        await page.screenshot(path='C:/Users/user/.gemini/antigravity/brain/c1faebd6-db48-4c90-ae49-46f335680b22/dark_electric_purple_search_filled.png')

        # Clear it
        await page.click('.search-clear-btn')
        await asyncio.sleep(0.5)

        # Test loading spinners and toasts
        await page.evaluate("document.getElementById('save-profile-btn').click()")
        await asyncio.sleep(0.5)
        await page.screenshot(path='C:/Users/user/.gemini/antigravity/brain/c1faebd6-db48-4c90-ae49-46f335680b22/dark_electric_purple_loading_toasts.png')

        await browser.close()

if __name__ == '__main__':
    asyncio.run(run())
