// Rewrites "../../../public/images/x.png" (the path that works in the VS Code
// preview) to "/images/x.png" (the path that works on the built site).
// Astro serves the contents of public/ at the site root.
const PUBLIC_REL = /(?:\.\.\/)+public\//g;

function walk(node) {
  if (node.type === 'raw' && typeof node.value === 'string') {
    node.value = node.value.replace(PUBLIC_REL, '/');
  } else if (node.type === 'element' && node.properties) {
    for (const k of ['src', 'href']) {
      if (typeof node.properties[k] === 'string') {
        node.properties[k] = node.properties[k].replace(PUBLIC_REL, '/');
      }
    }
  }
  if (Array.isArray(node.children)) node.children.forEach(walk);
}

export default function rehypePublicPaths() {
  return (tree) => walk(tree);
}
