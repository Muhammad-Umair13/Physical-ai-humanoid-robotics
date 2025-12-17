---
id: 1
title: Initialize Project Constitution
stage: constitution
date: 2025-12-06
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution Project: Textbook on Physical AI & Humanoid Robotics with Integrated RAG Chatbot

Core principles:

Accuracy: All technical explanations, code snippets, and robotics simulations must be verified against authoritative sources (ROS 2 docs, Gazebo/Unity manuals, NVIDIA Isaac SDK guides, peer-reviewed robotics literature).

Clarity: Content should be understandable by learners with a background in AI, robotics, or computer science; technical terms explained when first introduced.

Reproducibility: All exercises, labs, and capstone projects must be reproducible in simulation environments; all instructions fully traceable.

Rigor: Preference for peer-reviewed sources, official documentation, and real-world robotics standards.

Key standards:

Every factual claim, algorithm, or methodology must reference a source (official documentation, academic paper, or tutorial).

Citation format: APA style for all sources.

Source types: minimum 50% peer-reviewed or official robotics platform documentation.

Plagiarism: 0% tolerance; all code snippets and text must be original or properly attributed.

Writing clarity: target Flesch-Kincaid grade 10-12 for readability.

RAG chatbot: responses must be accurate, only use selected chapter text if provided, and cite chapter/subsection when relevant.

Constraints:

Word count per module: 2,500–3,500 words; total book: 10,000–12,000 words.

Minimum 20 sources across the book (journals, official docs, tutorials).

Format: Markdown for Docusaurus, deployable on GitHub Pages; include interactive code blocks and images.

Labs and exercises must include step-by-step instructions, expected results, and sample code.

C
labels: ["constitution", "initialization"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - N/A
---

## Prompt

Project: Textbook on Physical AI & Humanoid Robotics with Integrated RAG Chatbot

Core principles:

Accuracy: All technical explanations, code snippets, and robotics simulations must be verified against authoritative sources (ROS 2 docs, Gazebo/Unity manuals, NVIDIA Isaac SDK guides, peer-reviewed robotics literature).

Clarity: Content should be understandable by learners with a background in AI, robotics, or computer science; technical terms explained when first introduced.

Reproducibility: All exercises, labs, and capstone projects must be reproducible in simulation environments; all instructions fully traceable.

Rigor: Preference for peer-reviewed sources, official documentation, and real-world robotics standards.

Key standards:

Every factual claim, algorithm, or methodology must reference a source (official documentation, academic paper, or tutorial).

Citation format: APA style for all sources.

Source types: minimum 50% peer-reviewed or official robotics platform documentation.

Plagiarism: 0% tolerance; all code snippets and text must be original or properly attributed.

Writing clarity: target Flesch-Kincaid grade 10-12 for readability.

RAG chatbot: responses must be accurate, only use selected chapter text if provided, and cite chapter/subsection when relevant.

Constraints:

Word count per module: 2,500–3,500 words; total book: 10,000–12,000 words.

Minimum 20 sources across the book (journals, official docs, tutorials).

Format: Markdown for Docusaurus, deployable on GitHub Pages; include interactive code blocks and images.

Labs and exercises must include step-by-step instructions, expected results, and sample code.

C

## Response snapshot

Updated the project constitution at `.specify/memory/constitution.md` with the provided principles, standards, and constraints. The constitution is now version 1.0.0.

## Outcome

- ✅ Impact: Project constitution initialized and updated.
- 🧪 Tests: N/A
- 📁 Files: One file updated: .specify/memory/constitution.md
- 🔁 Next prompts: User will need to approve the changes and potentially update dependent templates.
- 🧠 Reflection: Successfully parsed user input and updated the constitution file. Fallback to agent-native PHR creation was successful after script not found.

## Evaluation notes (flywheel)

- Failure modes observed: `create-phr.sh` script not found, requiring agent-native fallback.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Investigate the `create-phr.sh` script path or ensure it's available.
