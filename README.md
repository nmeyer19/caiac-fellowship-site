# CAIAC Advanced Technical Fellowship — course site

Static site (Astro) for the Fall 2026 fellowship. Homepage shows the schedule. Each week has its own page with concept notes, slides, and a PDF.

## Setup

```bash
npm install
npx playwright install chromium   # once; needed only for PDFs
npm run dev                        # http://localhost:4321
```

## Where things are

| What | File |
|---|---|
| Site name, facilitator, meeting time/place, ARENA links | `src/site.config.ts` |
| Schedule rows and week pages | `src/content/weeks/*.md` (one file per row) |
| Chapter names and grouping | `src/lib/chapters.ts` |
| Slides | `public/slides/` |
| Generated PDFs | `public/pdfs/` |
| Styles and chapter colors | `src/styles/global.css` |

## Publish a week

1. Open the week file, for example `src/content/weeks/week-2.md`.
2. Write the material below the front matter (Markdown). Delete the template comment.
3. Optional: put slides in `public/slides/` and add `slides: /slides/week-2.pdf` to the front matter.
4. Set `ready: true`. This removes the "not posted yet" notice.
5. Run `npm run pdf`. This builds the site and writes `public/pdfs/week-2.pdf`.
   To make only some PDFs: `npm run pdf -- week-2`.
6. Commit and push. Vercel builds the site. The "Download PDF" button shows when the PDF file exists.

Re-run `npm run pdf` after you edit a week, so the PDF matches the page.

## Week file fields

```yaml
order: 2                     # position in the schedule
label: "Week 2"
title: "ARENA 1.1 (Transformers from Scratch)"
start: 2026-10-04
end: 2026-10-10
arena:                       # optional; adds the "ARENA exercises" button
  section: "1.1"
  url: https://learn.arena.education/...
ready: false                 # true when the material is posted
page: true                   # false = schedule row only (e.g. a break week)
slides: /slides/week-2.pdf   # optional
note: "Optional week"        # optional; small note under the topic in the schedule
chapter: interp              # optional; normally derived from the ARENA section
```

To add, move, or rename a week, edit or add a file here. The homepage updates itself.

The first paragraph of the file (above the template comment) is the one-line
description. It shows on the week page and behind "show descriptions" on the schedule.

## Chapter colors

Each row is colored by ARENA chapter, and the same color runs across the top of the
week page. The chapter comes from `arena.section` — `0.x` is Fundamentals, `1.x` Mech
Interp, `3.x` LLM Evals, `4.x` Alignment Science; a week with no ARENA section is a
Capstone, and `page: false` is a Break. Set `chapter:` in the front matter to override.
Names live in `src/lib/chapters.ts`, colors in the `--ch-*` variables in `global.css`.

## Meeting countdown

Set `meetingDay` in `src/site.config.ts` (`0` = Sunday … `6` = Saturday) and the
schedule shows how many days until the next session. Leave it `null` to hide it.

## Math and code

- Inline math: `$d_k$`
- Display math: `$$` on its own line above and below the formula.
- Code: fenced blocks with a language, for example ` ```python `.

## Deploy on Vercel

Import the repo in Vercel. It detects Astro. No settings needed (build: `astro build`, output: `dist`).
Then set `site` in `astro.config.mjs` to the final URL.
