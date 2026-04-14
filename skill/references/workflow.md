# IcingMoon Blog Workflow

## Prerequisites

- Ruby/Bundler environment capable of running Jekyll.
- Gems installed from `Gemfile` and the local gemspec.
- `nokogiri` available, because `_plugins/native_code_enhancer.rb` depends on it.

If dependencies are missing, start with:

```bash
bundle install
```

## Primary Local Commands

### Fast build check

Use this after content edits or before pushing:

```bash
bundle exec jekyll build
```

This validates Markdown, Liquid tags, plugins, and generated output without keeping a server running.

### Local preview

The repo's purpose-built preview script is:

```bash
./run-local-server.sh
```

What it does:

- runs `bundle exec jekyll build` first and exits on failure
- serves with `bundle exec jekyll serve --incremental --livereload --force_polling`
- removes `_site` on exit through a trap

Use this when layout, code block rendering, MathJax, or callout styling should be inspected in a browser.

### Alternative npm wrappers

`package.json` also exposes standard wrappers:

```bash
npm run build
npm run serve
```

Prefer `./run-local-server.sh` for this repo's day-to-day preview because it bakes in the current local flags.

## Validation Checklist

After a meaningful post edit, check:

1. `bundle exec jekyll build` succeeds.
2. The excerpt split occurs at `<!--more-->`.
3. Math renders and any top-of-post macro block still works.
4. Custom theorem/proof/callout blocks render without broken HTML.
5. Enhanced code blocks show the expected title, fold state, and style.
6. Root-relative asset links under `/assets/...` resolve.
7. Bilingual posts still have consistent `key`, `lang`, `bilingual`, and `hidden` values.

## Config Caveat

`_config.yml` is not hot-reloaded by `jekyll serve`.
Restart the preview server after config edits.

## Deployment And Publishing

This repo appears to publish from the Git repository directly:

- `_config.yml` sets `repository: IcingMoon/icingmoon.github.io`
- `update_blog.sh` stages all changes, commits, and pushes `origin master`

The helper script is:

```bash
./update_blog.sh "commit message"
```

Use it cautiously:

- it runs `git add .`
- it commits the entire worktree
- it pushes `master` immediately

Prefer checking `git status` first so unrelated changes are not accidentally published.

## Release Notes That Are Theme-Specific

`HOW_TO_RELEASE.md` is mainly for releasing the underlying theme package, not for normal blog posting.
Use it only when modifying the theme as a distributable package rather than publishing a new article.
