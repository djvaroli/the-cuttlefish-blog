# The Cuttlefish Blog

A personal blog built with Jekyll and hosted on GitHub Pages.

## Tech Stack
- **Jekyll** (static site generator)
- **GitHub Pages** (hosting, via GitHub Actions)
- **Minima** theme

## Local Development

```bash
bundle install          # Install dependencies (first time only)
bundle exec jekyll serve  # Start local server at http://localhost:4000
bundle exec jekyll build  # Build the site without serving
```

## Project Structure

```
_posts/          Blog posts (format: YYYY-MM-DD-title.md)
_drafts/         Unpublished draft posts
_pages/          Static pages (about, etc.)
_layouts/        Custom layout overrides
_includes/       Reusable HTML partials
assets/css/      Custom stylesheets
assets/images/   Images for posts and pages
_config.yml      Site configuration
```

## Writing Posts

Create a new file in `_posts/` with the naming convention `YYYY-MM-DD-title.md`.

Every post needs front matter at the top:

```yaml
---
layout: post
title: "Post Title"
date: YYYY-MM-DD
categories: [category1, category2]
---
```

Common categories: `technical`, `personal`, `general`

## Writing Drafts

Put draft posts in `_drafts/` (no date prefix needed in filename). Preview drafts locally with:

```bash
bundle exec jekyll serve --drafts
```

## Adding Pages

Create a new `.md` file in `_pages/` with front matter:

```yaml
---
layout: page
title: "Page Title"
permalink: /page-url/
---
```

## Deployment

Push to the `main` branch. GitHub Actions will build and deploy automatically.
