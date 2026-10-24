// Snapshot Requstory's published map for each pilot company as JSON.
//
//   node integrations/requstory/fetch_maps.js integrations/requstory/pilot.yaml companies/_requstory
//
// The page is the comparison baseline, so it is captured as it stands, with
// the date it was read. Nothing here interprets it.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const [pilotPath, outDir] = process.argv.slice(2);
const slugs = [...fs.readFileSync(pilotPath, 'utf8').matchAll(/slug:\s*([a-z0-9-]+)/g)].map(m => m[1]);
fs.mkdirSync(outDir, { recursive: true });

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  for (const slug of slugs) {
    const url = `https://requstory.com/companies/${slug}/`;
    await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 });
    const map = await page.evaluate(() => {
      const text = el => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
      const main = document.querySelector('main') || document.body;
      const status = { 'st-ok': 'ok', 'st-missing': 'missing', 'st-unknown': 'unknown' };
      const checks = [...main.querySelectorAll('details.checks li')].map(li => {
        const mark = li.querySelector('span');
        const proof = li.querySelector('a');
        return {
          status: status[mark?.className] || 'unknown',
          text: text(li.querySelectorAll('span')[1]).replace(/\s*·\s*Proof$/, ''),
          proof: proof ? proof.href : null,
        };
      });
      const bands = [...main.querySelectorAll('.band')].map(band => ({
        band: text(band.querySelector('.label')),
        units: [...band.querySelectorAll('.unit')].map(u => ({
          name: text(u.querySelector('b')),
          found_on_website: u.classList.contains('w'),
          activities: [...u.querySelectorAll('li')].map(text),
        })),
      }));
      const journey = [...main.querySelectorAll('.journey li')].map(li => ({
        stage: text(li.querySelector('b')), quote: text(li.querySelector('small')) || null,
      }));
      const findings = [...main.querySelectorAll('.finding')].map(f => ({
        title: text(f.querySelector('b')).replace(/^Add /, ''),
        description: text(f.querySelector('span')),
        source: f.querySelector('a') ? { title: text(f.querySelector('a')), url: f.querySelector('a').href } : null,
      }));
      const line = re => (main.innerText.match(re) || [])[1] || null;
      return {
        title: text(main.querySelector('h1')),
        summary: text(main.querySelector('h1 + p')) || text(main.querySelector('p')),
        score: (main.innerText.match(/(\d+)\/100/) || [])[1] || null,
        researched: line(/Researched ([^\n]+)/),
        read: line(/\nRead ([^\n]+)/),
        checks, journey, bands, findings,
      };
    });
    map.slug = slug; map.url = url; map.captured_at = new Date().toISOString();
    fs.writeFileSync(path.join(outDir, `${slug}.json`), JSON.stringify(map, null, 2));
    console.log(`${slug}: ${map.checks.length} checks, ${map.bands.reduce((n, b) => n + b.units.length, 0)} units, ${map.findings.length} findings, score ${map.score}`);
  }
  await browser.close();
})();
