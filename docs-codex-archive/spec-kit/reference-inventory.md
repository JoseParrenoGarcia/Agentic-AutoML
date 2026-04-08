# Reference Inventory

This is the seed list of external inspiration captured from the rough notes. It is intentionally framed as a working inventory to mine, validate, and structure, rather than a fully curated bibliography.

## Purpose

- track which papers and repositories are influencing the design
- clarify what each source is useful for
- avoid shallow reference-dropping
- prepare a future `references/` directory with extracted notes

## Seed Sources From The Notes

### AI Analyst plugin

- Source: `https://github.com/ai-analyst-lab/ai-analyst-plugin/tree/master`
- Why it matters:
  - example of agentic analytics workflows
  - potential inspiration for analysis decomposition and tool boundaries
- What to extract later:
  - orchestration patterns
  - artifact design
  - agent role definitions

### Karpathy autoresearch

- Source: `https://github.com/karpathy/autoresearch/tree/master`
- Why it matters:
  - inspiration for iterative research loops
  - memory and long-running exploration patterns
- What to extract later:
  - iteration structure
  - history handling
  - research workflow design

### AutoKaggle

- Source: `https://github.com/ShaneZhong/autokaggle`
- Why it matters:
  - benchmark-oriented autonomous competition workflow inspiration
  - possible ideas for round-based experimentation
- What to extract later:
  - competition loop structure
  - evaluation strategy
  - run organization

### Google DS-style papers from the notes

- Sources listed in the rough notes:
  - `https://arxiv.org/pdf/2509.21825`
  - `https://arxiv.org/pdf/2410.02958`
- Why they matter:
  - planning, execution, review, and corrective loops
  - small-step iteration philosophy
  - agent decomposition for data science work
- What to extract later:
  - planner-reviewer-router patterns
  - prompt and artifact structures
  - control loops and recovery mechanisms

### Personal knowledge base idea

- Source: `https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f`
- Why it matters:
  - inspiration for a reusable wiki or knowledge-base layer
- What to extract later:
  - organizational principles
  - note templates
  - retrieval-friendly structure

### Newsletter Engine repo

- Source: `https://github.com/JoseParrenoGarcia/newsletter-engine`
- Why it matters:
  - existing reusable patterns, skills, and rules from prior agentic work
- What to extract later:
  - import and indexing skills
  - rule structure
  - workflow conventions that can transfer cleanly

### PDF import skill idea

- Source noted in rough notes:
  - `https://github.com/JoseParrenoGarcia/newsletter-engine/blob/main/.claude/skills/import-pdf/SKILL.md`
- Why it matters:
  - useful for ingesting papers without bloating context
- What to extract later:
  - document-conversion workflow
  - ingestion rules
  - reportable summaries

## Reference Ingestion Principles

1. Do not store references as a loose pile of links.
2. Every reference should have:
   - a short summary
   - relevance to this project
   - sections or files worth focusing on
   - reusable tactics extracted from it
3. Prefer text-extractable versions of sources when building local searchable notes.
4. Separate raw source captures from cleaned notes and reusable patterns.

## Recommended Next Step

Create a `references/index.md` file and begin adding one structured note per important paper or repo, starting with whichever source most directly informs the planner-reviewer-router loop.
