const { test, expect } = require('@playwright/test');
const path = require('path');

const guideUrl = `file://${path.resolve(
  __dirname, '../reference/deepep-v2-ascend-950-guide.html')}`;

test('desktop renders six explanatory diagrams without page overflow', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(guideUrl);

    const diagrams = page.locator('figure.explainer');
    await expect(diagrams).toHaveCount(6);
    await expect(diagrams.locator('figcaption')).toHaveCount(6);

    const metrics = await page.evaluate(() => ({
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: document.documentElement.clientWidth,
      emptyFigures: [...document.querySelectorAll('figure.explainer')]
        .filter((figure) => !figure.textContent.trim()).length,
    }));

    expect(metrics.documentWidth).toBeLessThanOrEqual(metrics.viewportWidth + 1);
    expect(metrics.emptyFigures).toBe(0);
});
