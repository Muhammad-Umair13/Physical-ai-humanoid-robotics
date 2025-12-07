<!--
Version change:  -> 1.0.0
List of modified principles:
  - PRINCIPLE_1_NAME -> Accuracy
  - PRINCIPLE_2_NAME -> Clarity
  - PRINCIPLE_3_NAME -> Reproducibility
  - PRINCIPLE_4_NAME -> Rigor
Added sections:
  - Key Standards
  - Constraints
Removed sections:
  - PRINCIPLE_5, PRINCIPLE_6 (Removed placeholders due to user-defined sections)
Templates requiring updates:
  - .specify/templates/plan-template.md: ⚠ pending
  - .specify/templates/spec-template.md: ⚠ pending
  - .specify/templates/tasks-template.md: ⚠ pending
  - .specify/templates/commands/*.md: ⚠ pending
  - CLAUDE.md: ✅ updated (for GUIDANCE_FILE reference)
  - README.md: ⚠ pending
  - docs/quickstart.md: ⚠ pending
Follow-up TODOs: None
-->
# Textbook on Physical AI & Humanoid Robotics with Integrated RAG Chatbot Constitution

## Core Principles

### Accuracy
All technical explanations, code snippets, and robotics simulations must be verified against authoritative sources (ROS 2 docs, Gazebo/Unity manuals, NVIDIA Isaac SDK guides, peer-reviewed robotics literature).

### Clarity
Content should be understandable by learners with a background in AI, robotics, or computer science; technical terms explained when first introduced.

### Reproducibility
All exercises, labs, and capstone projects must be reproducible in simulation environments; all instructions fully traceable.

### Rigor
Preference for peer-reviewed sources, official documentation, and real-world robotics standards.

## Key Standards

Every factual claim, algorithm, or methodology must reference a source (official documentation, academic paper, or tutorial).
Citation format: APA style for all sources.
Source types: minimum 50% peer-reviewed or official robotics platform documentation.
Plagiarism: 0% tolerance; all code snippets and text must be original or properly attributed.
Writing clarity: target Flesch-Kincaid grade 10-12 for readability.
RAG chatbot: responses must be accurate, only use selected chapter text if provided, and cite chapter/subsection when relevant.

## Constraints

Word count per module: 2,500–3,500 words; total book: 10,000–12,000 words.
Minimum 20 sources across the book (journals, official docs, tutorials).
Format: Markdown for Docusaurus, deployable on GitHub Pages; include interactive code blocks and images.
Labs and exercises must include step-by-step instructions, expected results, and sample code.

## Governance

Constitution supersedes all other practices; Amendments require documentation, approval, migration plan; All PRs/reviews must verify compliance; Complexity must be justified; Use CLAUDE.md for runtime development guidance.

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06
