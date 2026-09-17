import DOMPurify from "dompurify";

export function appendText(output, text) {
  // Append nodes so subsequent text cannot erase an earlier rich result.
  output.append(document.createTextNode(text.replace(/\x1b\[[0-9;]*m/g, "")));
  output.scrollTop = output.scrollHeight;
}

export function appendHtml(output, html) {
  const fragment = DOMPurify.sanitize(html, {
    RETURN_DOM_FRAGMENT: true,
    ALLOWED_TAGS: ["div", "span", "p", "br", "hr", "pre", "code", "strong", "b", "em", "i", "u", "s", "small", "sub", "sup", "ul", "ol", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "a", "table", "caption", "thead", "tbody", "tfoot", "tr", "th", "td"],
    ALLOWED_ATTR: ["style", "href", "title", "colspan", "rowspan", "start"],
  });
  // Preserve Symbolica's colors and spacing without allowing page-wide CSS,
  // positioning or external URLs in styles supplied by an object's formatter.
  const safeStyles = new Set(["color", "background-color", "font-weight", "font-style", "text-decoration", "text-align", "vertical-align", "white-space"]);
  for (const element of fragment.querySelectorAll("[style]")) {
    const properties = [...element.style].map(name => [name, element.style.getPropertyValue(name)]);
    element.removeAttribute("style");
    for (const [name, value] of properties) {
      if (safeStyles.has(name) && !/url\s*\(|var\s*\(/i.test(value)) element.style.setProperty(name, value);
    }
  }
  for (const link of fragment.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  const result = document.createElement("div");
  result.className = "rich-result";
  result.append(fragment);
  output.append(result);
  output.scrollTop = output.scrollHeight;
}
