const fmt = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
export const range = (a: Date, b: Date) => `${fmt.format(a)} – ${fmt.format(b)}`;
