// The course runs through four ARENA chapters. Each gets a color, used on the
// schedule and on the week page, so the shape of the term is visible at a glance.
export type ChapterKey =
  | 'fundamentals'
  | 'interp'
  | 'evals'
  | 'alignment'
  | 'capstone'
  | 'break';

export const CHAPTERS: Record<ChapterKey, { name: string; legend: boolean }> = {
  fundamentals: { name: 'Fundamentals', legend: true },
  interp: { name: 'Mech Interp', legend: true },
  evals: { name: 'LLM Evals', legend: true },
  alignment: { name: 'Alignment Science', legend: true },
  capstone: { name: 'Capstone', legend: true },
  break: { name: 'Break', legend: false },
};

const BY_LEADING_DIGIT: Record<string, ChapterKey> = {
  '0': 'fundamentals',
  '1': 'interp',
  '3': 'evals',
  '4': 'alignment',
};

type WeekData = {
  chapter?: ChapterKey;
  page: boolean;
  arena?: { section: string };
};

// Set `chapter:` in a week's front matter to override this.
export function chapterOf(data: WeekData): ChapterKey {
  if (data.chapter) return data.chapter;
  if (!data.page) return 'break';
  const digit = data.arena?.section.match(/\d/)?.[0];
  return (digit && BY_LEADING_DIGIT[digit]) || 'capstone';
}
