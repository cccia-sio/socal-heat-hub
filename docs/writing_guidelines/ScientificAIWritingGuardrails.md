# Avoiding AI Writing Artifacts in Scientific Work

This document defines **undesirable writing patterns associated with AI-generated scientific text** and provides enforceable guidelines to eliminate them.

These rules apply to:
- Papers and theses
- Methods sections
- Figure captions
- Code documentation
- Research notes and memos

The goal is to ensure writing reflects **real scientific reasoning, not synthetic fluency**.

---

# Core Principle

> Scientific writing should reflect **mechanism, uncertainty, and specificity** — not polish, balance, or rhetorical completeness.

---

# 1. No Inflated Scientific Significance

AI often exaggerates importance without evidence.

### Avoid:
- “This study provides critical insights into…”
- “This work significantly advances…”
- “These findings are crucial for understanding…”
- “This represents a major step forward…”

### Required:
- State **what was done and what was observed**
- Let importance emerge from:
  - Magnitude of effect
  - Comparison to prior work
  - Novelty of method or dataset

---

# 2. No Generic Framing of Research Questions

AI defaults to vague, universal framing.

### Avoid:
- “Understanding X is important for…”
- “X plays a critical role in…”
- “There is growing interest in…”

### Required:
- Anchor motivation in:
  - A specific gap
  - A known limitation
  - A measurable discrepancy

**Example:**
- ❌ “Sediment transport plays a crucial role in coastal systems”
- ✅ “Existing energetics-based transport models underpredict onshore flux under low-energy conditions observed at Torrey Pines”

---

# 3. Eliminate “Smooth Narrative” Bias

AI tends to fabricate a clean, linear story.

### Avoid:
- Implied inevitability
- Overly clean causality
- Perfect alignment between hypothesis and results

### Required:
- Preserve:
  - Ambiguity
  - Failed expectations
  - Nonlinear or conflicting signals

---

# 4. No Overgeneralization Beyond Data

AI frequently extends conclusions too far.

### Avoid:
- “These results suggest a general mechanism…”
- “This applies broadly to coastal systems…”

### Required:
- Explicitly constrain:
  - Spatial domain
  - Temporal scale
  - Forcing regime

---

# 5. Avoid Empty Technical Language

AI uses jargon without adding meaning.

### Avoid:
- “complex dynamics”
- “intricate interactions”
- “robust framework”
- “comprehensive analysis”

### Required:
- Replace with:
  - Equations
  - Variables
  - Measurable quantities

---

# 6. Methods Must Be Mechanistic, Not Descriptive

AI methods sections often describe actions without physical meaning.

### Avoid:
- “Data were analyzed using standard techniques”
- “We applied a model to evaluate…”

### Required:
- Specify:
  - Governing equations
  - Assumptions
  - Parameter definitions
  - Coordinate systems
  - Units

---

# 7. No “Black Box” Modeling Language

AI hides model structure behind vague phrasing.

### Avoid:
- “machine learning was used to predict…”
- “a model was trained to…”

### Required:
- Explicitly state:
  - Inputs
  - Outputs
  - Loss function (if applicable)
  - Training/validation split
  - Physical interpretation (if any)

---

# 8. Avoid Formulaic Paragraph Structure

AI tends to follow rigid templates:
1. Topic sentence  
2. General statement  
3. Example  
4. Summary  

### Required:
- Allow structure to follow:
  - Logical derivation
  - Data constraints
  - Argument development

---

# 9. Eliminate Redundant Restatement

AI repeats ideas with slight variation.

### Avoid:
- Rephrasing the same claim multiple times
- “This suggests… This indicates… This highlights…”

### Required:
- Each sentence must:
  - Add new information
  - Refine a claim
  - Introduce evidence

---

# 10. No Artificial Balance

AI forces symmetry where none exists.

### Avoid:
- “While X…, Y…” when imbalance is real
- Equal treatment of unequal effects

### Required:
- Reflect true asymmetry in:
  - Data
  - Mechanisms
  - Uncertainty

---

# 11. Avoid Overuse of Transitional Phrases

### Avoid:
- “Furthermore”
- “Moreover”
- “In addition”
- “However” (overused)

### Required:
- Use transitions only when logically necessary

---

# 12. Avoid “Conclusion-Sounding” Sentences Mid-Section

AI inserts mini-conclusions throughout.

### Avoid:
- “Overall, this demonstrates…”
- “Taken together…”

### Required:
- Reserve synthesis for:
  - End of sections
  - Discussion

---

# 13. Preserve Quantitative Anchoring

AI drifts toward qualitative summaries.

### Avoid:
- “large increase”
- “significant change”
- “strong relationship”

### Required:
- Include:
  - Magnitudes
  - Units
  - Scales
  - Uncertainty where possible

---

# 14. No Hallucinated or Generic Citations

### Avoid:
- Placeholder citations
- Misattributed findings

### Required:
- Cite only:
  - Known, verified sources
  - Directly relevant work

---

# 15. Figure Captions: No Narrative, Only Description

AI captions often read like summaries.

### Avoid:
- Interpretation
- Claims about significance

### Required:
- Include:
  - What is plotted
  - Variables and units
  - Time/space domain
  - Key processing steps (if needed)

---

# 16. Code and Analysis Writing Rules (MATLAB / Scientific Computing)

### Avoid:
- “This code calculates…”
- “We simply…”

### Required:
- Document:
  - Inputs and outputs
  - Assumptions
  - Edge cases
  - Units and coordinate conventions

---

# 17. Avoid Overly Polished or “Perfect” Tone

AI writing often:
- Feels frictionless
- Lacks intellectual struggle

### Required:
- Allow:
  - Partial understanding
  - Open questions
  - Competing interpretations

---

# 18. Domain-Specific Check (Coastal / Oceanographic Work)

Ensure writing includes:
- Explicit coordinate conventions (e.g., cross-shore positive offshore)
- Datum references (e.g., NAVD88)
- Forcing regimes (wave height, period, flux)
- Depth dependence of processes
- Mechanistic links (e.g., radiation stress gradients → bed shear stress)

---

# 19. Final Self-Check

Before accepting generated text, verify:

- Does every paragraph contain **at least one concrete quantity, mechanism, or constraint**?
- Are any sentences **purely rhetorical or decorative**?
- Could any sentence be removed without losing meaning?
- Does the text reflect **real uncertainty and asymmetry**?

---

# Enforcement Prompt (Use in LLM Workflows)

> “Write in a scientific style that avoids all AI-associated artifacts listed in the ‘Avoiding AI Writing Artifacts in Scientific Work’ guide. Prioritize mechanism, quantitative specificity, and domain realism over fluency or completeness. Do not use generic framing, inflated significance, or formulaic structure.”