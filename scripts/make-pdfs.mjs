// Makes a PDF of every week page marked `ready: true`.
// Run with: npm run pdf   (builds the site first)
// Only some weeks:  npm run pdf -- week-2 week-3
// Output: public/pdfs/<week-id>.pdf. Commit these files; Vercel serves them.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { chromium } from 'playwright';

const root = process.cwd();
const weeksDir = path.join(root, 'src/content/weeks');
const outDir = path.join(root, 'public/pdfs');
const dist = path.join(root, 'dist');
fs.mkdirSync(outDir, { recursive: true });

const only = process.argv.slice(2);
const ids = fs.readdirSync(weeksDir)
  .filter((f) => f.endsWith('.md'))
  .map((f) => ({ id: f.replace(/\.md$/, ''), text: fs.readFileSync(path.join(weeksDir, f), 'utf8') }))
  .filter(({ text }) => /^ready:\s*true\s*$/m.test(text) && !/^page:\s*false\s*$/m.test(text))
  .map(({ id }) => id)
  .filter((id) => only.length === 0 || only.includes(id));

if (ids.length === 0) {
  console.log('No weeks have `ready: true`. Nothing to do.');
  process.exit(0);
}
if (!fs.existsSync(dist)) {
  console.error('No dist/ folder. Run `npm run build` first (or use `npm run pdf`).');
  process.exit(1);
}

// Serve the built site (dist/) with a small local server.
const types = {
  '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.pdf': 'application/pdf',
};
const server = http.createServer((req, res) => {
  let f = path.join(dist, decodeURIComponent(req.url.split('?')[0]));
  if (fs.existsSync(f) && fs.statSync(f).isDirectory()) f = path.join(f, 'index.html');
  if (!f.startsWith(dist) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'Content-Type': types[path.extname(f)] || 'application/octet-stream' });
  fs.createReadStream(f).pipe(res);
});
await new Promise((r) => server.listen(0, r));
const base = `http://localhost:${server.address().port}`;

// Optional: CHROME_PATH=/path/to/chrome npm run pdf
const browser = await chromium.launch(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {});
try {
  const page = await browser.newPage();
  for (const id of ids) {
    await page.goto(`${base}/weeks/${id}/`, { waitUntil: 'networkidle' });
    await page.pdf({
      path: path.join(outDir, `${id}.pdf`),
      format: 'Letter',
      margin: { top: '0.75in', bottom: '0.75in', left: '0.75in', right: '0.75in' },
      printBackground: true,
      displayHeaderFooter: true,
      headerTemplate: '<span></span>',
      footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#666"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    });
    console.log(`public/pdfs/${id}.pdf`);
  }
} finally {
  await browser.close();
  server.close();
}
console.log('Done. Run `npm run build` again (or push) so the pages show the PDF links.');
