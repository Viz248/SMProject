# Project Abstract: Workflow Utility & Similarity Metrics for CI/CD Automation

**Course:** Software Metrics
**Target outcome:** Course project + patent submission

---

## Abstract

Continuous Integration and Continuous Deployment (CI/CD) automation, particularly
GitHub Actions, has become a standard part of modern software development. As projects
mature, teams accumulate dozens of automated workflows — but existing research and
tooling only evaluates whether these workflows are *correctly configured* (free of
security flaws, syntax issues, or outdated dependencies). No existing work measures
whether a workflow is still *delivering value* to the project, or whether multiple
workflows have become functionally redundant and could be consolidated.

This project proposes two complementary metrics — a **Workflow Utility Score (WUS)**
and a **Workflow Similarity Score (WSS)** — computed entirely from data available
through GitHub's public API (workflow definitions, run history, triggers, and job
structure). Together, these metrics feed a rule-based decision engine that recommends
whether a workflow should be **kept, merged with another, retired, or manually
reviewed** — turning passive metric reporting into an actionable, explainable
recommendation system.

---

## Problem Statement

Every GitHub Actions workflow a team writes keeps running on every relevant push or
pull request indefinitely, whether or not it still serves a purpose. Over a project's
lifetime this leads to two forms of waste:

1. **Dead automation** — workflows that no longer target existing code, whose outputs
   nobody consumes, or that always pass/fail trivially without providing real signal.
2. **Redundant automation** — multiple workflows independently performing
   near-identical jobs (e.g., duplicated test/lint/build steps introduced by different
   contributors over time), wasting CI time and creating maintenance overhead.

Both problems are acknowledged informally in practice (real teams manually consolidate
duplicate workflows, as seen in public GitHub pull requests), but no existing research
or tool automatically **measures** either problem or recommends an action.

---

## Why This Is Novel

We reviewed the closest existing research and tooling in this space:

- **GitHub Actions workflow "smell" detection** (e.g., Khatami et al., GASH) — detects
  security, performance, and configuration anti-patterns. Does not measure ongoing
  usefulness.
- **Workflow maintenance-burden studies** (e.g., Valenzuela-Toledo et al., "The Hidden
  Costs of Automation") — measures the human effort required to keep workflows working.
  Measures cost of upkeep, not value delivered.
- **Workflow outdatedness research** (e.g., Decan et al.) — flags workflows using stale
  action versions. A correctness/freshness check, not a usefulness check.
- **CI/CD industry metrics** (DORA metrics, "CI Vitals") — measure pipeline speed,
  cost, and reliability at an aggregate level, not the value of any individual workflow.
- **Scientific workflow similarity/clustering research** — exists, but for an entirely
  different domain (data-pipeline tools for scientific computing), not CI/CD automation.

None of the above asks the two questions this project asks directly: *"Is this workflow
still worth keeping?"* and *"Does this workflow already exist elsewhere in disguise?"*

---

## Proposed Approach

### Workflow Utility Score (WUS)
Measures whether a workflow still delivers value, using objective, API-derivable
signals such as:
- Outcome variability (does it ever meaningfully fail, or always trivially pass?)
- Whether other workflows or processes depend on its output
- Whether it targets code paths that still exist in the current codebase
- Recency and frequency of runs relative to project activity

### Workflow Similarity Score (WSS)
Measures whether two workflows are functionally redundant, using:
- Trigger similarity (matching `on:` events, branch/path filters)
- Action fingerprint overlap (Jaccard similarity over `uses:` actions, with common
  boilerplate actions like `checkout` weighted down and distinctive actions weighted up)
- Step-intent classification (grouping `run:` commands into categories like TEST,
  BUILD, DEPLOY without parsing shell code)
- Job structure similarity (number of jobs, dependency depth)

### Decision Engine
Combines WUS and WSS through explainable, rule-based thresholds (not a single opaque
similarity score) to recommend one of: **Keep, Merge, Retire, or Manual Review** — with
a plain-language justification for each recommendation (e.g., "92% action overlap, same
trigger, same target paths, no unique downstream consumers").

---

## Data & Feasibility

All data is pulled from GitHub's public REST/GraphQL API: workflow YAML definitions,
run history, timestamps, and outcomes. No proprietary access, shell-code parsing, or
abstract syntax tree analysis is required — this keeps the project buildable within a
single semester and fully reproducible by anyone reviewing the work.

---

## Why This Supports a Patent Submission

Patent offices generally do not grant patents on a bare metric or formula. What makes
this project defensible as a method/system claim is that it doesn't stop at scoring —
it pairs two novel measurements with an **automated decision process that acts on
them** (flagging or recommending merge/retirement of specific workflows). That
input → computation → triggered action structure is what elevates this from "a new
metric" to "a novel system," which is the framing patent reviewers look for.

---

## Status

- Idea validated against existing literature; closest related work identified above.
- Metric definitions and decision rules scoped.
- Implementation and validation (against real public repositories) in progress.
- Three faculty reviews are scheduled across the semester; details to follow.
