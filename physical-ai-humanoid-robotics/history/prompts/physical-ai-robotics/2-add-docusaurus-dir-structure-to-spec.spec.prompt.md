---
id: 2
title: Add Docusaurus Directory Structure to Spec
stage: spec
date: 2025-12-06
surface: agent
model: claude-sonnet-4-5-20250929
feature: physical-ai-robotics
branch: 1-physical-ai-robotics
user: User
command: /sp.clarify
labels: ["specification", "update"]
links:
  spec: specs/1-physical-ai-robotics/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/1-physical-ai-robotics/spec.md
tests:

---

## Prompt

Task: Update the feature specification for the "Physical AI & Humanoid Robotics Textbook with Integrated RAG Chatbot" project.

Objective: Include the full Docusaurus directory structure in the specification document so it clearly represents where all modules, chapters, labs, media, and components reside.

Requirements:

1. Add a new section in spec.md titled "Project Directory Structure".
2. Include the following hierarchy exactly as provided:

   physical-ai-humanoid-robotics/
   ├── docs/
   │   ├── module1-ros2/
   │   │   ├── 01-introduction-to-ros2.md
   │   │   ├── 02-ros2-architecture.md
   │   │   ├── 03-nodes-topics-services-actions.md
   │   │   ├── 04-launch-files-parameters.md
   │   │   ├── 05-python-agents.md
   │   │   ├── 06-bridging-python-rclpy.md
   │   │   ├── 07-writing-testing-packages.md
   │   │   ├── 08-practical-examples.md
   │   │   ├── 09-humanoid-urdf.md
   │   │   ├── 10-sdf-robot-format.md
   │   │   ├── 11-simple-humanoid-model.md
   │   │   └── labs/
   │   │       ├── lab1-ros2-node-sensor.ety-checks.md
│       ├── 08-capstone-autonomous.md
│       └── labs/
│           ├── lab1-voice-action-interface.md
│           ├── lab2-llm-command-planner.md
│           └── lab3-mini-humanoid-demo.md
├── static/
│   ├── images/
│   ├── diagrams/
│   └── assets/
├── src/
│   └── components/
│       └── rag-chatbot.jsx
├── sidebars.js
├── docusaurus.config.js
├── package.json
└── README.md

3. Ensure this section is added in Markdown format with proper code block formatting (triple backticks) so it renders correctly.

4. Clarify that this directory structure corresponds to the **organization of all modules, chapters, labs, media, and the RAG chatbot** in the Docusaurus-based textbook.

Constraints:

- Keep all existing specifications intact.
- The directory section should be added as a new subsection under "Requirements" or "Project Overview".
- Markdown formatting must be valid for Docusaurus.

Deliverable: Updated `spec.md` with the directory structure clearly included and formatted.

## Response snapshot

Updated `specs/1-physical-ai-robotics/spec.md` by adding the "Project Directory Structure" subsection under "Requirements" with the provided Docusaurus hierarchy.

## Outcome

- ✅ Impact: The feature specification now includes the detailed Docusaurus directory structure, clarifying content organization.
- 🧪 Tests: N/A
- 📁 Files: `specs/1-physical-ai-robotics/spec.md` was modified.
- 🔁 Next prompts: The specification is updated. The next logical step is to run `/sp.plan` to create the implementation plan.
- 🧠 Reflection: Successfully updated the spec file as requested, handling the `create-phr.sh` script failure by manually creating the PHR.

## Evaluation notes (flywheel)

- Failure modes observed: `create-phr.sh` script failed due to `pwsh` not being found.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Investigate environment setup for PowerShell or provide alternative Bash scripts.
