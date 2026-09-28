# Automated Anomaly Explanation with Counterfactuals
## Frontend Design Board — Swiss Machine

**Work Mode:** Use this document as the single visual source of truth while implementing the Streamlit frontend. Preserve the design system across every page and component. Do not introduce new visual styles, random colors, alternate fonts, excessive rounded cards, gradients, glassmorphism, or decorative UI that conflicts with this specification.

---

# 1. Design Direction

## Style
**Swiss Machine**

### Core visual personality
- Precise
- Technical
- Calm
- Minimal
- Research-oriented
- Industrial
- Intelligent
- Highly structured

### Design principle

> **The interface should feel like a precision instrument for understanding machine intelligence.**

The design should communicate complexity through **grid, typography, spacing, labels, numbers, and hierarchy**, rather than through decoration.

### Visual references
Swiss / International Typographic design language:
- strong grid
- asymmetrical but intentional layouts
- generous whitespace
- large typography
- clear alignment
- thin rules
- restrained color
- functional visual elements

Avoid:
- gradients
- glassmorphism
- excessive rounded corners
- neon cyberpunk styling
- excessive shadows
- decorative blobs
- rainbow UI
- card-heavy dashboards
- generic SaaS visual language

---

# 2. Color Style

## Primary Color Style
**Swiss Machine**

The interface is predominantly neutral. Blue acts as the primary **signal / action color**, rather than being used as a constant decorative background.

### Base palette

| Role | Color | Hex |
|---|---|---|
| Primary Background | Warm Off-White | `#F7F7F3` |
| Primary Ink | Near Black | `#111111` |
| Brand / Action Blue | Cobalt Blue | `#174BFF` |
| Soft Blue | Pale Blue | `#DDE5FF` |
| Success | Green | `#18A765` |
| Warning / Error | Red-Orange | `#E94B35` |
| White Surface | White | `#FFFFFF` |
| Secondary Text | Neutral Gray | `#595959` |
| Divider | Light Gray | `#D9D9D4` |

## Color rules

### 70 / 20 / 10 principle
Approximate visual distribution:

- **70%** warm off-white / white
- **20%** black / gray typography and rules
- **10%** cobalt blue + semantic states

### Blue usage
Use `#174BFF` for:
- primary CTA
- active navigation
- selected state
- key links
- important system signal
- loading animation accent
- important data highlight

Do NOT use blue as the entire page background.

### Semantic colors
Use semantic colors only when meaning requires them:

**Green**
- VALID
- FEASIBLE
- NORMAL
- successful completion

**Red-orange**
- ANOMALY
- invalid candidate
- error
- warning

**Blue**
- active / processing / selected
- neutral system state
- primary action

---

# 3. Fonts

## Heading Font
### Helvetica Neue

Use Helvetica Neue as the primary visual identity for large headings and section titles.

Fallback:

```css
font-family:
  "Helvetica Neue",
  Helvetica,
  Arial,
  sans-serif;
```

### Recommended weights
- 700 — Bold
- 600 — Medium / Semibold where available
- 800 / 900 — only for major hero titles if the available font environment supports it

Helvetica Neue should provide the clean Swiss visual character.

---

## Body Font
### Inter

Use Inter for:
- paragraphs
- descriptions
- helper text
- navigation labels
- form labels
- UI controls

Fallback:

```css
font-family:
  Inter,
  Arial,
  sans-serif;
```

---

## Technical Font
### IBM Plex Mono

Use IBM Plex Mono for:
- sensor values
- RPM
- torque
- temperatures
- anomaly scores
- timestamps
- IDs
- candidate counts
- metric values
- technical metadata

Example:

```text
RPM       2254
TORQUE    43.2 Nm
SCORE     -0.0782
```

This creates a strong distinction between **human explanation** and **machine telemetry**.

---

# 4. Typography Hierarchy

The hierarchy must be obvious even if the user looks at the page for only two seconds.

## Level 1 — Hero / Page Title
**Helvetica Neue Bold**

Target:
- 44–56 px desktop
- line height ~0.95–1.05
- tight letter spacing
- left aligned

Example:

```text
UNDERSTAND
THE MACHINE STATE
```

---

## Level 2 — Section Heading
**Helvetica Neue Bold**

Target:
- 24–32 px
- line height ~1.05–1.15

Example:

```text
MODEL COUNTERFACTUAL
```

---

## Level 3 — Component Heading
**Helvetica Neue Medium/Bold**

Target:
- 16–20 px
- line height ~1.2

Example:

```text
CURRENT STATE
```

---

## Level 4 — Body
**Inter Regular**

Target:
- 14–17 px
- line height ~1.45–1.6

Use for:
- explanations
- descriptions
- helper text

---

## Level 5 — Metadata / Labels
**IBM Plex Mono**

Target:
- 10–12 px
- uppercase where appropriate
- increased letter spacing

Examples:

```text
ANALYSIS / 001
STATUS
MODEL SCORE
CANDIDATES
```

---

## Level 6 — Metrics
**IBM Plex Mono Medium/Bold**

Target:
- 22–36 px depending on importance

Example:

```text
2254
```

---

# 5. Typography Rules

1. Prefer **left alignment**.
2. Avoid center-aligning large blocks of content.
3. Use oversized typography only for primary messages.
4. Use uppercase mostly for short technical labels, not paragraphs.
5. Avoid more than three typography families/roles.
6. Never use decorative display fonts.
7. Do not use bold everywhere.
8. Let whitespace create hierarchy.

---

# 6. Layout System

## Grid

Use a consistent desktop grid.

Recommended conceptual structure:

```text
12-column grid
│
├── main content
├── secondary information
└── action / status area
```

### Core alignment
All major sections should share a common left edge.

### Spacing rhythm
Use consistent spacing based on a simple scale:

```text
4
8
12
16
24
32
48
64
80
```

Avoid random spacing values.

---

# 7. Borders and Rules

Swiss Machine uses **rules instead of decorative containers**.

Preferred:
- 1 px black rules
- 1 px gray dividers
- occasional 2 px black outline for important interactive elements

Avoid:
- thick card outlines everywhere
- heavy drop shadows
- decorative borders
- excessive rounded corners

### Border radius

Default:
```text
0–4 px
```

Small radius only when needed for usability.

Primary design language should feel **geometric and flat**.

---

# 8. Buttons

Buttons must look functional and editorial.

## Primary Button

Example:

```text
ANALYZE →
```

Style:
- black or cobalt-blue background
- white text
- minimal radius
- strong typography
- no gradient

## Secondary Button

Example:

```text
DOWNLOAD REPORT
```

Style:
- transparent / white
- black border
- black text

## Button behavior

Hover:
- subtle color shift
- slight movement
- no exaggerated animation

Pressed:
- small physical movement / reduction
- immediate visual feedback

---

# 9. Inputs

Do not make inputs look like generic rounded SaaS forms.

Use:

```text
ROTATIONAL SPEED
┌─────────────────────────────┐
│ 2254                      RPM│
└─────────────────────────────┘
```

Characteristics:
- white background
- thin black border
- minimal radius
- clear label above
- monospace value where appropriate
- units separated visually

Focus state:
- cobalt blue border
- subtle blue focus indication

---

# 10. Status Language

Use the following states consistently.

## Normal

```text
NORMAL
```

Color:
**Green**

## Anomaly

```text
ANOMALY DETECTED
```

Color:
**Red-orange**

## Processing

```text
ANALYZING
```

Color:
**Cobalt blue**

## Counterfactual Found

```text
COUNTERFACTUAL FOUND
```

Color:
**Cobalt blue / green depending on context**

## No Counterfactual

```text
NO VALID COUNTERFACTUAL FOUND
```

Supporting explanation:

> No valid counterfactual was found within the configured search space and budget. This does not imply that no real-world resolution exists.

---

# 11. Page Structure

## Page 01 — Welcome

Purpose:
Introduce the product.

Layout:

```text
AUTOMATED ANOMALY
EXPLANATION
WITH COUNTERFACTUALS

Explainable AI for machine states

                         ENTER →
```

### Animation
Logo / product mark should appear with a short controlled zoom or scale transition.

Do not make the intro cinematic.

Recommended duration:
**~0.8–1.2 sec**

Then transition directly into the main welcome interface.

---

# 12. Main Dashboard

Primary question:

> **HOW DO YOU WANT TO ANALYZE?**

Two primary choices:

```text
┌────────────────────┐    ┌────────────────────┐
│                    │    │                    │
│  SINGLE INPUT      │    │  BATCH INPUT       │
│                    │    │                    │
│  Analyze one       │    │  Upload a dataset  │
│  machine state     │    │                    │
│                    │    │                    │
└────────────────────┘    └────────────────────┘
```

These are not decorative cards.

They are **workflow choices**, so keep them simple and strong.

---

# 13. Single Input Page

Title:

```text
ANALYZE / SINGLE
```

Subtitle:

```text
ENTER MACHINE STATE
```

Inputs:

```text
AIR TEMPERATURE
PROCESS TEMPERATURE
ROTATIONAL SPEED
TORQUE
TOOL WEAR
```

Primary action:

```text
GENERATE →
```

Before generation, validate:
- required values
- numeric values
- configured bounds
- correct feature order

---

# 14. Batch Input Page

Title:

```text
ANALYZE / BATCH
```

Main action:

```text
UPLOAD DATASET
```

Then:

```text
CSV FILE
✓ READY

GENERATE →
```

Do not overcrowd this screen.

---

# 15. Loading / Processing Animation

## Core requirement

The loading animation must **NOT use cards**.

It should be:

- background-free
- vector-based
- SVG / Lottie compatible
- centered
- lightweight
- responsive
- simple
- technical
- visually aligned with the Swiss Machine system

### Three stages

```text
DETECT
ANALYZE
EXPLAIN
```

Each appears/flashes **one at a time**.

### Timing

Per stage:

**0.4 sec**

Total three-stage loop:

**1.2 sec**

Then repeat.

Conceptual loop:

```text
0.0s ─ 0.4s    DETECT
0.4s ─ 0.8s    ANALYZE
0.8s ─ 1.2s    EXPLAIN
1.2s ─ 1.6s    DETECT
...
```

### Animation behavior

Only one stage is visually active at a time.

Example:

```text
        ⊕
      DETECT
```

then:

```text
        ◔
     ANALYZE
```

then:

```text
        ?
      EXPLAIN
```

### No cards
Do not place each state inside a box.

Do not use bordered tiles.

Do not use a panel around the animation.

The animation should float directly on the page background.

### Motion language

**Detect**
- crosshair
- target
- scanning point

**Analyze**
- segmented circle
- rotating arc
- data sweep

**Explain**
- question mark
- branching lines
- simple explanatory glyph

### Visual treatment

Primary linework:
`#111111`

Accent:
`#174BFF`

Optional semantic accent:
`#18A765`

Background:
transparent

### Stroke style

- clean vector strokes
- consistent stroke width
- no 3D effects
- no glow
- no gradients

### Frame rate

Target:
**60 FPS**

Fallback acceptable:
**30 FPS**

The animation should remain smooth on normal laptop hardware.

### File-size goal

Prefer:
**under ~100–150 KB per animation asset**

If the complete animation can be represented as one optimized Lottie JSON, prefer one asset instead of three separate large files.

### Streamlit implementation intent

Preferred architecture:

```text
Streamlit
   ↓
Lottie / SVG animation
   ↓
centered transparent animation
   ↓
0.4s DETECT
0.4s ANALYZE
0.4s EXPLAIN
   ↓
loop
```

Do not create a custom React animation layer.

---

# 16. Explanation Page

This is the most important screen.

Title:

```text
EXPLANATION / 001
```

Status:

```text
ANOMALY DETECTED
```

Then show:

## Current State

```text
RPM       2254
TORQUE    43.2 Nm
```

## Model Counterfactual

```text
RPM       2254 → 1914
CHANGE              −340
```

## Validation

```text
VALIDITY        ✓
FEASIBILITY     ✓
PLAUSIBILITY    ?
```

## Explanation

Use a short human-readable explanation.

Example:

> A lower rotational speed produced a nearby machine state that the anomaly detector classified as normal.

Primary action:

```text
DOWNLOAD REPORT
```

Important disclaimer:

> Model-valid counterfactual ≠ causal root cause or safety recommendation.

---

# 17. Analytics Page — Optional

Keep analytics secondary.

Purpose:
Show system/model behavior.

Examples:

```text
DETECTOR

PRECISION       58.21%
RECALL          11.50%
F1              19.21%
```

Counterfactual:

```text
FOUND           36 / 39
FEASIBLE        36 / 36
PLAUSIBLE        9 / 36
```

Charts should be:
- restrained
- grid-aligned
- low decoration
- readable
- annotated clearly

---

# 18. Motion System

All motion should be purposeful.

## Welcome
Short scale/zoom

## Navigation
Very small fade / slide

## Button
Immediate micro-interaction

## Page state
Subtle transition

## Loading
Main visual motion:

**DETECT → ANALYZE → EXPLAIN**

Do not animate every component.

---

# 19. Swiss Machine Design Rules — LOCKED

### Always
- Use Helvetica Neue for headings.
- Use Inter for body text.
- Use IBM Plex Mono for machine values and technical labels.
- Use warm off-white as the main background.
- Use black as the primary visual anchor.
- Use cobalt blue as the main interaction/signal color.
- Use strong grid alignment.
- Use whitespace intentionally.
- Use thin rules and restrained components.
- Keep UI mostly square / low-radius.
- Keep Lottie background-free.

### Never
- No rainbow color palette.
- No gradients.
- No glassmorphism.
- No oversized rounded cards.
- No excessive shadows.
- No generic dashboard templates.
- No decorative animation everywhere.
- No card-based loading animation.
- No random font changes.
- No blue background on every component.
- No misleading safety/causal language.

---

# 20. Frontend Mental Model

The user journey should feel like one investigation:

```text
MACHINE STATE
      ↓
DETECT
      ↓
ANOMALY?
      ↓
SEARCH
      ↓
VALIDATE
      ↓
EXPLAIN
      ↓
UNDERSTAND
```

The UI should therefore move from:

**input → signal → investigation → explanation**

rather than feeling like separate unrelated pages.

---

# 21. Work Mode Instructions

When implementing the Streamlit frontend:

1. Read the project's source-of-truth documents before modifying or designing anything:
   - `docs/PRD.md`
   - `docs/SRS.md`
   - `docs/SYSTEM_ARCHITECTURE.md`
   - `docs/ML_PIPELINE.md`
   - `docs/DATA_DICTIONARY.md`
   - `docs/COUNTERFACTUAL_SPEC.md`
   - `docs/EVALUATION.md`
   - `Automated_Anomaly_Explanation_Project_Handoff.md` when available.

2. Treat existing Python code, configuration, and the above documents as the technical source of truth.

3. Treat this document as the **visual source of truth**.

4. Preserve the existing frozen project structure.

5. Do not create a React frontend.

6. Use Streamlit for the frontend.

7. Do not create extra frontend directories unless explicitly authorized.

8. Do not change ML logic merely to support the UI.

9. The frontend must consume the existing pipeline APIs instead of duplicating ML logic.

10. Do not directly edit project files through an assistant tool. Provide implementation instructions/code for manual application through the user's chosen coding workflow.

11. Keep the visual language consistent across every screen.

12. Review git diff/status after implementation.

---

# 22. Final Design Identity

## PRODUCT FEEL

**Swiss Machine**

### Three words

> **PRECISE. INTELLIGENT. CLEAR.**

### Visual formula

```text
Helvetica Neue
        +
Inter
        +
IBM Plex Mono
        +
Warm Off-White
        +
Black
        +
Cobalt Blue
        +
Strict Grid
        +
Whitespace
        +
Technical Motion
```

### Core loading identity

```text
DETECT
   ↓
ANALYZE
   ↓
EXPLAIN
```

**0.4 sec each • background-free • vector/Lottie • smooth 60 FPS target • continuous loop**

---

## Final Design Principle

> **Make the interface feel less like a dashboard and more like a precision instrument for understanding an intelligent machine.**
