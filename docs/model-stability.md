```mermaid
flowchart LR
  style LOOP fill:#111111,stroke:#FFD700,stroke-width:2px,color:#ffffff

  A[🔴 High Error<br/>l_i] --> B[🟡 High φ<br/>increase φ_i]
  B --> C[🟢 Amplified Learning Signal<br/>Δθ ∝ φ]
  C --> D[🔵 Runaway Risk<br/>(overfit / instability)]
  D -->|feedback| A

  classDef danger fill:#8b0000,stroke:#ffb3b3,color:#fff;
  classDef amplify fill:#ffda79,stroke:#8b6b00,color:#000;
  classDef stable fill:#a8ffb0,stroke:#2d7a00,color:#000;
  classDef risk fill:#7fb3ff,stroke:#274e7a,color:#000;

  class A danger;
  class B amplify;
  class C stable;
  class D risk;

---

### 2) Stabilization shields (Mermaid radial-ish layout)
This shows the four stabilization mechanisms as shields surrounding the loop:

```markdown
```mermaid
flowchart TB
  subgraph Stabilizers [🛡️ Stabilization Mechanisms]
    direction LR
    S1["🛡️ Regularization<br/>(clip, L1/L2)"]
    S2["🔁 Experience Replay<br/>(buffer + mixing)"]
    S3["⏱️ Persistence Filtering<br/>(E[l_i] moving avg)"]
    S4["⚖️ Normalization<br/>(layer-wise φ budget)"]
  end

  S1 -.-> A
  S2 -.-> A
  S3 -.-> A
  S4 -.-> A

  classDef shield fill:#0f766e,stroke:#14b8a6,color:#fff;
  class S1,S2,S3,S4 shield;

---

### 3) Combined layout (loop + shields + key equations)
If you prefer a single combined block (loop + equations below), use this:

```markdown
```mermaid
flowchart LR
  A[🔴 High Error<br/>l_i] --> B[🟡 Increase φ<br/>Δφ_i ∝ E[l_i]]
  B --> C[🟢 Amplified Learning Signal<br/>Δθ ∝ φ]
  C --> D[🔵 Runaway Risk]
  D -->|feedback| A

  subgraph Shields [🛡️ Stabilizers]
    direction LR
    R1["Regularization (clip, L1/L2)"]
    R2["Experience Replay (buffer)"]
    R3["Persistence Filtering (E[l_i])"]
    R4["Normalization (layer budget)"]
  end

  R1 -.-> B
  R2 -.-> A
  R3 -.-> B
  R4 -.-> B

  classDef eq fill:#f3f4f6,stroke:#9ca3af,color:#111,stroke-width:1px;
  subgraph Equations [Key Equations]
    direction TB
    E1["Gilder: Δφ_i = β * (E[l_i] - λ * φ_i)"]
    E2["Sculptor: Δθ_i = -α * ∇_{θ_i} (L_transformed + Ω(θ))"]
  end
  class E1,E2 eq;

---

### 4) How to add to your Markdown file
Open `docs/STABILITY_FRAMEWORK.md` (or your README section) and paste either of the code blocks above into the file. Example placement:

```markdown
## Core instability

(Short intro text...)

<!-- insert diagram -->
```mermaid
... (paste one of the diagram blocks here) ...

> **Note:** keep the triple backticks exactly as shown. The first line must be ```mermaid.

---

### 5) Previewing Mermaid
- **On GitHub**: GitHub supports Mermaid in `.md` files; the diagram will render in PRs and the repo UI.
- **Locally**:
  - Use **VS Code** + *Markdown Preview Enhanced* or *Mermaid Preview* extension.
  - Or use `npx @mermaid-js/mermaid-cli` to render to PNG/SVG:
    ```bash
    npx @mermaid-js/mermaid-cli -i docs/STABILITY_FRAMEWORK.md -o docs/diagrams/kintsugi_loop.svg
    ```

---

### 6) Commit steps (quick)
From your repo root:

```bash
git add docs/STABILITY_FRAMEWORK.md
git commit -m "Add Mermaid diagrams for stability framework"
git push origin <your-branch>
