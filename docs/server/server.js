import express from 'express';
import fetch from 'node-fetch';
import { mkdirSync, existsSync, createWriteStream, statSync } from 'fs';
import { basename, extname, join, resolve } from 'path';
import mime from 'mime-types';

const app = express();
const PORT = process.env.PORT || 3000;

// Where we store cached images (public so static middleware can serve it)
const PUBLIC_DIR = resolve('public');
const CACHE_DIR  = join(PUBLIC_DIR, 'logo-cache');

if (!existsSync(CACHE_DIR)) mkdirSync(CACHE_DIR, { recursive: true });

// 1) Serve the static UI (your SPA in /docs) and the cached logos
app.use('/logo-cache', express.static(CACHE_DIR, {
  maxAge: '90d',
  immutable: true,
}));
app.use(express.static('docs', { maxAge: '1h' })); // your site

// --- Utilities ----------------------------------------------------
function slugifyTeam(name='') {
  return String(name)
    .toLowerCase()
    .replace(/&/g,'and')
    .replace(/[\u2019'’]/g,'')
    .replace(/[^a-z0-9]+/g,'-')
    .replace(/^-+|-+$/g,'');
}
function isCommons(u){ return /commons\.wikimedia\.org/i.test(u); }
function isSpecialFilePath(u){ return /\/wiki\/Special:FilePath\//i.test(u); }
function isFileTitle(u){ return /\/wiki\/File:/i.test(u); }

// Most reliable: Commons thumb.php?f=<File_Name>&width=160 (raster PNG for SVGs)
function commonsThumbPhpUrl(inputUrl) {
  try {
    const url = new URL(inputUrl);
    let file = '';
    if (isSpecialFilePath(inputUrl)) {
      file = decodeURIComponent(url.pathname.split('/').pop() || '');
    } else if (isFileTitle(inputUrl)) {
      const last = decodeURIComponent(url.pathname.split('/').pop() || '');
      file = last.replace(/^File:/i,'');
    } else { return null; }
    file = file.replace(/\s+/g,'_'); // commons expects underscores
    const thumb = new URL('https://commons.wikimedia.org/w/thumb.php');
    thumb.searchParams.set('f', file);
    thumb.searchParams.set('width', '160');
    return thumb.toString();
  } catch { return null; }
}

async function fetchToFile(srcUrl, destPath) {
  const r = await fetch(srcUrl, {
    headers: { 'User-Agent': 'OddsGenius/1.0 (logo cache)' }
  });
  if (!r.ok || !r.body) throw new Error(`Fetch failed ${r.status}`);
  await new Promise((resolve, reject) => {
    const ws = createWriteStream(destPath);
    r.body.pipe(ws);
    r.body.on('error', reject);
    ws.on('finish', resolve);
  });
  return true;
}

// --- The /api/logo endpoint --------------------------------------
app.get('/api/logo', async (req, res) => {
  try {
    const team = String(req.query.team || '').trim();
    const hint = String(req.query.hint || '').trim();

    if (!team && !hint) return res.status(400).send('Missing team or hint');

    const slug = slugifyTeam(team || basename(hint));
    const destBase = join(CACHE_DIR, slug); // we’ll add extension after sniffing
    let chosenPath = null;

    // 1) If we already cached this slug (any extension), serve it
    const candidates = ['.png','.jpg','.jpeg','.webp','.svg'];
    for (const ext of candidates) {
      const p = destBase + ext;
      if (existsSync(p) && statSync(p).size > 0) {
        chosenPath = p; break;
      }
    }
    if (chosenPath) {
      return res.sendFile(chosenPath, { headers: { 'Cache-Control': 'public, max-age=7776000, immutable' } });
    }

    // 2) Decide source URL (prefer Commons thumb when hint is Commons)
    let source = hint;
    if (isCommons(hint)) {
      const thumb = commonsThumbPhpUrl(hint);
      if (thumb) source = thumb;
    }

    // 3) Otherwise, optional: try TheSportsDB if no hint
    if (!source && team) {
      const sdb = `https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=${encodeURIComponent(team)}`;
      const r = await fetch(sdb);
      if (r.ok) {
        const j = await r.json();
        const t = (j?.teams || [])[0];
        source = t?.strTeamBadge || t?.strTeamLogo || t?.strTeamFanart1 || null;
      }
    }

    if (!source) return res.status(404).send('No source for logo');

    // 4) Fetch and save with a content-type-based extension
    const head = await fetch(source, { method:'GET', headers:{ 'User-Agent':'OddsGenius/1.0' }});
    if (!head.ok || !head.body) throw new Error(`Fetch failed ${head.status}`);

    const ct = head.headers.get('content-type') || '';
    const ext = (mime.extension(ct) && ('.' + mime.extension(ct))) ||
                (source.match(/\.(png|jpg|jpeg|webp|svg)(\?|$)/i)?.[0].replace(/\?.*/,'') || '.png');

    const dest = destBase + ext;
    await new Promise((resolve, reject) => {
      const ws = createWriteStream(dest);
      head.body.pipe(ws);
      head.body.on('error', reject);
      ws.on('finish', resolve);
    });

    res.setHeader('Cache-Control', 'public, max-age=7776000, immutable');
    return res.sendFile(dest);
  } catch (e) {
    console.error('logo proxy error:', e);
    res.status(500).send('Logo proxy error');
  }
});

// Start
app.listen(PORT, () => {
  console.log(`OG server on :${PORT}\n- static: /docs\n- cache:  /logo-cache\n- api:    /api/logo?team=...&hint=...`);
});
