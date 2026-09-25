import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// One Markdown file per row of the schedule.
// The homepage schedule and the week pages both come from these files.
const weeks = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/weeks' }),
  schema: z.object({
    order: z.number(),                 // sort position in the schedule
    label: z.string(),                 // "Week 1", "Skip 1", ...
    title: z.string(),                 // topic shown in the schedule
    start: z.coerce.date(),
    end: z.coerce.date(),
    arena: z
      .object({ section: z.string(), url: z.string().url() })
      .optional(),
    chapter: z                         // optional; otherwise derived from `arena.section`
      .enum(['fundamentals', 'interp', 'evals', 'alignment', 'capstone', 'break'])
      .optional(),
    page: z.boolean().default(true),   // false = schedule row only, no page
    ready: z.boolean().default(false), // false = page shows "materials coming soon"
    slides: z.string().optional(),     // link or /slides/file.pdf
    note: z.string().optional(),       // short note shown in the schedule
  }),
});

export const collections = { weeks };
