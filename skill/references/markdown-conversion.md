# Plain Markdown Conversion

## Goal

Convert a generic Markdown document into a polished post for this repository.
Do not preserve the raw source formatting mechanically when the repo already has a better rendering primitive.

Use recent posts as the style baseline first.

Prefer these over much older posts because the newer ones reflect the completed math/code/callout tooling.
For exact block syntax, bilingual declarations, and beautification primitives, also learn from:

- `_posts/2025-12-25-hello-world.md`
- `_posts/2025-12-25-hello-world-en.md`

## Conversion Workflow

1. Identify the article type: technical note, crypto writeup, tutorial, or image-heavy essay.
2. Create proper front matter and opening structure.
3. Normalize formulas to this repo's MathJax habits.
4. Move or reference images under a dedicated `assets/images/<post-subdir>/` folder when the post owns local images.
5. Replace plain Markdown callouts and weak HTML with the repo's custom blocks.
6. Upgrade code fences, collapsible sections, and theorem-like environments where the content benefits.
7. If the post is bilingual or will likely get a translation, preserve the structure needed for paired articles.
8. For Chinese technical notes, localize section titles and subsection names instead of leaving large English heading chunks from the source.

## Math Rules

This repository intentionally uses `$$...$$` for both inline and display math.

Rules to enforce:

- Inline formulas: use `$$x+y$$` inline with the sentence.
- Display formulas: also use `$$...$$`, but surround the formula block with at least one blank line before and after it.
- When a displayed derivation spans multiple lines, keep it in a standalone display block with enough surrounding whitespace for reliable parsing.
- If the source Markdown uses `$...$`, convert it to `$$...$$`.
- If a formula contains absolute-value or norm bars such as `$|x|$` or `$|E(\mathbb{F}_p)|$`, rewrite the bars with LaTeX commands such as `\vert x \vert` or `\vert E(\mathbb{F}_p) \vert` to avoid accidental Markdown table parsing.
- If the source uses LaTeX theorem environments that Jekyll will not understand directly, convert them to repo-supported blocks.

Example inline:

```markdown
The verifier checks whether $$e(g_1, g_2) = e(h_1, h_2)$$ holds.
```

Example display:

```markdown
We obtain

$$
\begin{aligned}
f(x) &= x^2 + 1 \\
g(x) &= x^2 - 1
\end{aligned}
$$

which completes the derivation.
```

## Image Rules

For a post with local images, use a dedicated subfolder under `assets/images/`.

Recommended layout:

```text
assets/images/<post-subdir>/
```

Choose `<post-subdir>` from stable post context, for example:

- a short date code
- a date-plus-slug
- another concise unique folder name already agreed by the user

Do not scatter a post's local images across unrelated folders if the post should own them.

### Preferred Rendering

For local blog images, default to:

```liquid
{% include figure.html src="/assets/images/<post-subdir>/image.png" alt="..." width="85%" caption="..." %}
```

This is the preferred pattern in recent image-heavy posts.

Use `<img ...>` only when one of these is true:

- the asset already lives outside `assets/images`, such as challenge files under `/assets/ctf-stuff/...`
- special raw HTML styling is needed, such as `referrerpolicy`, custom zoom, or centered rendering that `figure.html` does not cover cleanly

### Image Style Heuristics

- Use root-relative paths beginning with `/assets/...`.
- Supply meaningful `alt`.
- Add `caption` when the image conveys context worth naming.
- Use width controls commonly seen in recent posts such as `50%`, `65%`, `85%`, `95%`, or `100%`.
- For mathematical diagrams or graph sketches copied from notes, start from `95%` and only narrow it when the browser rendering feels too loose.
- For image galleries or extra photos, consider wrapping them in `{% plain fold ... %}` blocks.

## Upgrading Plain Markdown Structure

Do not leave plain Markdown in its weakest form when the content clearly maps to repo components.

### Weak source pattern -> preferred target

- Plain intro paragraph -> `{: .info}` summary block near the top.
- Long code dump -> titled or foldable enhanced code block.
- `<details>` without styling -> repo-styled `<details class="warning|info|success|exploit">` or `{% plain fold ... %}`.
- LaTeX theorem/proof environments -> `{% theorem %}`, `{% lemma %}`, `{% definition %}`, `{% proof %}`, `{% remark %}`.
- Generic admonitions -> `.info`, `.success`, `.warning`, `.error`, `<section class="...">`, or the corresponding Liquid block.
- Plain image markdown `![alt](...)` -> `{% include figure.html %}` for local blog images unless a simpler inline image is clearly better.
- Bare property bullet lists after a definition -> a titled `{% remark %}` or `{% plain info %}` block.
- Naive-vs-correct protocol sketches -> separate `{% plain info title="..." %}` and `{% plain error title="..." %}` blocks.
- A long correctness argument in the main flow -> move it into `{% proof fold title="..." %}` after a short intuition paragraph.

## Beautification Heuristics

When converting, actively improve presentation using the repo's current components:

- Wrap formal statements, definitions, and propositions with theorem-like blocks.
- Convert proof sketches and long derivations into `{% proof fold %}` or `<details class="info">` when that improves readability.
- Mark core pitfalls, attack ideas, and important reminders with `warning` or `error` style blocks.
- Mark challenge statements, small observations, and checkpoints with `success` or `info` blocks.
- Add `title="..."`, `type="..."`, and `fold="true"` to code fences when filenames or semantics are useful.
- Use `{% plain fold ... %}` for long side notes, expanded examples, or supplementary images.

The target is not "same Markdown but valid."
The target is "looks like it belongs in this blog."

## Recent Style Preferences

Recent technical posts prefer:

- dense but structured prose
- highlighted core ideas using custom blocks
- foldable proofs and code
- consistent math notation
- selective use of raw HTML blocks only when they materially improve layout
- section and subsection names that match the post language
- fewer isolated one-line callouts, and more titled blocks that package a complete local idea
- a short intuition bridge before heavy formulas or protocol mechanics
- grouped external reading links near the relevant warning/background block, not dumped only at the end

Recent image-heavy posts prefer:

- `{% include figure.html %}` for local images
- widths tuned per image instead of one global default
- foldable photo collections with `{% plain fold ... %}`

Recent manually refined crypto notes also prefer:

- replacing English headings like `Isogenies`, `Examples`, `Correctness` with `同源映射`, `同源示例`, `SIDH 的正确性`
- merging overly fragmented paragraphs so the exposition reads as a continuous note rather than a literal Markdown conversion
- keeping English terminology on first mention in parentheses instead of every sentence

When you need the exact syntax for a block or bilingual declaration, prefer copying the pattern from the `hello-world` pair instead of inventing new Markdown or HTML.

When recent posts and older posts disagree, follow the recent posts.

## Bilingual Learning Requirement

When converting a post, always check whether the user wants or may later want a bilingual pair.

If the post is bilingual:

- create two Markdown files
- share the same `key`
- set `lang` explicitly
- put `bilingual: true` on the primary visible version
- put `hidden: true` on the secondary translation

Even when only one language is requested right now, avoid choices that would make a future bilingual pairing awkward, such as inconsistent titles, unstable file naming, or missing `key` planning when a pair is clearly expected.
