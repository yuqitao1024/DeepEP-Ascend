const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const guideUrl = `file://${path.resolve(
  __dirname, '../reference/deepep-v2-ascend-950-guide.html')}`;
const communicationGuidePath = path.resolve(
  __dirname, '../reference/ascend-hcomm-simt-communication-overlap.html');
const communicationGuideUrl = `file://${communicationGuidePath}`;
const h800LessonPath = path.resolve(
  __dirname, '../lessons/0002-run-h800-representative.html');
const h800LessonUrl = `file://${h800LessonPath}`;
const ascendLessonPath = path.resolve(
  __dirname, '../lessons/0003-run-ascend950-representative.html');
const ascendLessonUrl = `file://${ascendLessonPath}`;

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

test('communication guide renders four explanatory diagrams without page overflow', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(communicationGuideUrl);

    await expect(page.locator('h1')).toContainText('HCOMM');
    await expect(page.locator('#roadmap')).toBeVisible();

    const diagrams = page.locator('figure.explainer');
    await expect(diagrams).toHaveCount(4);
    await expect(diagrams.locator('figcaption')).toHaveCount(4);

    const metrics = await page.evaluate(() => ({
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: document.documentElement.clientWidth,
      emptyFigures: [...document.querySelectorAll('figure.explainer')]
        .filter((figure) => !figure.textContent.trim()).length,
    }));

    const localSources = await page.locator('a[href^="../"]').evaluateAll(
      (links) => links.map((link) => link.getAttribute('href')));

    expect(metrics.documentWidth).toBeLessThanOrEqual(metrics.viewportWidth + 1);
    expect(metrics.emptyFigures).toBe(0);
    for (const source of localSources) {
      expect(
        fs.existsSync(path.resolve(path.dirname(communicationGuidePath), source)),
        `missing local source: ${source}`,
      ).toBeTruthy();
    }
});

test('H800 lesson renders executable commands and diagrams without page overflow', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(h800LessonUrl);

    await expect(page.locator('h1')).toContainText('representative case');
    await expect(page.locator('#copy-run')).toContainText(
      'python3 tests/benchmark/run_ep.py');
    await expect(page.locator('#copy-run')).toContainText(
      '--profile representative');

    const diagrams = page.locator('figure.explainer');
    await expect(diagrams).toHaveCount(2);
    await expect(diagrams.locator('figcaption')).toHaveCount(2);

    const metrics = await page.evaluate(() => ({
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: document.documentElement.clientWidth,
      emptyFigures: [...document.querySelectorAll('figure.explainer')]
        .filter((figure) => !figure.textContent.trim()).length,
    }));
    const localSources = await page.locator('a[href^="../"]').evaluateAll(
      (links) => links.map((link) => link.getAttribute('href')));

    expect(metrics.documentWidth).toBeLessThanOrEqual(metrics.viewportWidth + 1);
    expect(metrics.emptyFigures).toBe(0);
    for (const source of localSources) {
      expect(
        fs.existsSync(path.resolve(path.dirname(h800LessonPath), source)),
        `missing local source: ${source}`,
      ).toBeTruthy();
    }
});

test('Ascend lesson renders direct eight-rank commands without page overflow', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto(ascendLessonUrl);

    await expect(page.locator('h1')).toContainText('representative case');
    await expect(page.locator('#copy-run-ascend')).toContainText(
      'python tests/benchmark/run_ep.py');
    await expect(page.locator('#copy-run-ascend')).toContainText(
      '--backend ascend');
    await expect(page.locator('#copy-run-ascend')).toContainText(
      '--profile representative');
    await expect(page.locator('#copy-run-ascend')).not.toContainText(
      'task-submit');

    const diagrams = page.locator('figure.explainer');
    await expect(diagrams).toHaveCount(2);
    await expect(diagrams.locator('figcaption')).toHaveCount(2);

    const metrics = await page.evaluate(() => ({
      documentWidth: document.documentElement.scrollWidth,
      viewportWidth: document.documentElement.clientWidth,
      emptyFigures: [...document.querySelectorAll('figure.explainer')]
        .filter((figure) => !figure.textContent.trim()).length,
    }));
    const localSources = await page.locator('a[href^="../"]').evaluateAll(
      (links) => links.map((link) => link.getAttribute('href')));

    expect(metrics.documentWidth).toBeLessThanOrEqual(metrics.viewportWidth + 1);
    expect(metrics.emptyFigures).toBe(0);
    for (const source of localSources) {
      expect(
        fs.existsSync(path.resolve(path.dirname(ascendLessonPath), source)),
        `missing local source: ${source}`,
      ).toBeTruthy();
    }
});
