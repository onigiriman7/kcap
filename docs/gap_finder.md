# Gap finder: detailed design and flow

The gap finder reads a document corpus and proposes **gap cards**. A gap card is a hypothesis about a decision or practice that experts follow but no document explains. Each card cites the documents that show the behaviour and carries an opening question to ask the expert. Gap cards are the agenda for interviewer strategies C and D.

It adapts the **idea-generation loop of [Sakana AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2)** (`perform_ideation_temp_free.py`): propose an idea, check novelty with a search tool, reflect, finalise. Here, "novel" means *not already written down in the corpus*.

| File | Role |
|---|---|
| `kcap/components/gap_miner.py` | Orchestrator (`mine`) |
| `kcap/components/gap_common.py` | Clustering (`cluster_corpus`), briefing (`workshop_file`), ranking and de-duplication (`finalize`) |
| `kcap/sakana_adapters/ideation_adapter.py` | Adapted AI Scientist loop (`generate_temp_free_gap_cards`, `CorpusSearch`, `FinalizeGapCard`, prompts, parser) |
| `kcap/components/gap_miner_simple.py` | Single-prompt baseline (`mine_simple`) for comparison |
| `kcap/components/corpus.py`, `kcap/retrieval/index.py` | Corpus access, hybrid BM25 search, similarity for de-duplication |

---

## 1. End-to-end overview

```mermaid
flowchart TB
  IN[("public/corpus.jsonl<br/>Document chunks")] --> PC["PublicCorpus.load<br/>strip ground truth · path guard · build HybridIndex"]
  PC --> S1

  subgraph S1["Stage 1 · Cluster (gap_common.cluster_corpus)"]
    direction TB
    C1["Batch 30 docs"] --> C2["LLM: assign each doc to a MAIN entity<br/>reuse known entity names"] --> C3["Validate IDs · grow known-entity list"]
    C3 -->|more batches| C1
    C3 --> C4["Group docs by entity<br/>size &lt; min_cluster_size → misc"]
  end

  S1 --> LOOP{{"For each cluster"}}

  subgraph S2["Stage 2 · Briefing (gap_common.workshop_file)"]
    W1["Cluster name + corpus description<br/>+ up to 25 docs (500 chars)<br/>+ SOP docs (200 chars)<br/>+ task statement"]
  end

  subgraph S3["Stage 3 · Ideation (ideation_adapter)"]
    I1["generations_per_cluster hypotheses<br/>each: propose → search → reflect → finalise"]
  end

  LOOP --> S2 --> S3 --> RAW[("raw cards + cluster tag")]
  RAW -->|next cluster| LOOP

  subgraph S4["Stage 4 · Rank &amp; de-duplicate (gap_common.finalize)"]
    F1["Build GapCard objects<br/>priority = importance × undocumentedness"] --> F2["Sort by priority ↓"] --> F3["Greedy de-dup<br/>similar hypothesis OR identical evidence"] --> F4["Keep top_k · rename G01, G02, …"]
  end

  RAW --> S4 --> OUT[("Gap cards")]
  S3 -. log .-> LOG[("gap_miner/ideation_log.json")]
  S2 -. save .-> WSF[("gap_miner/workshops/NN.md")]
  S1 -. save .-> CLF[("gap_miner/clusters.json")]
```

---

## 2. Stage 1: clustering

**Why:** an undocumented practice shows up as a *pattern across related documents*, for example several specs that all make the same unexplained choice. Clustering puts related documents in front of the LLM together.

```mermaid
flowchart TB
  A["docs = corpus.docs<br/>known = []"] --> B["Take next batch of 30"]
  B --> C["Prompt (role: interviewer, task: gap_miner.cluster)<br/>'Assign each record to its MAIN entity …<br/>Reuse these names: known[:60]'<br/>each doc shown as [id] first 250 chars"]
  C --> D["LLM JSON: {assignments: {doc_id: entity}}"]
  D --> E{"doc_id exists<br/>and entity is a string?"}
  E -- no --> X["ignore (hallucinated / malformed)"]
  E -- yes --> F["assign[doc_id] = entity<br/>append to known if new"]
  F --> G{"more batches?"}
  X --> G
  G -- yes --> B
  G -- no --> H["group docs by entity<br/>(unassigned → 'misc')"]
  H --> I{"group size ≥ min_cluster_size<br/>and not 'misc'?"}
  I -- yes --> J["keep as cluster"]
  I -- no --> K["merge into 'misc'"]
  J --> L[("clusters: {entity: [doc_ids]}")]
  K --> L
```

| Detail | Value |
|---|---|
| Batch size | 30 documents per call |
| Text shown per document | first 250 characters |
| Known-name hint | up to 60 previous entity names, so synonyms merge |
| Minimum cluster size | `gap_miner.min_cluster_size` (default 3) |
| Calls | ⌈N / 30⌉ |
| Seed | batch offset (deterministic, cached) |

---

## 3. Stage 2: the briefing ("workshop file")

AI Scientist conditions idea generation on a *workshop call for papers*. kcap replaces that with a per-cluster briefing:

```text
# Cluster: <entity>

## Organization
<corpus description supplied at ingest>

## Records about this entity (<total> total, showing ≤25)
[<id> | <doc_type> | <date>]
<first 500 characters>
…

## Official procedures that may apply (SOPs)
- [<id>] <first 200 characters>     (documents whose doc_type == "sop"; "(none)" otherwise)

## Task
Find decisions, priorities or judgments in these documents that deviate from procedures or from what a
newcomer would do, and whose reasons no document explains. Cite document IDs.
```

Every briefing is saved to `gap_miner/workshops/NN.md`, so you can audit exactly what the LLM saw.

---

## 4. Stage 3: the ideation loop (AI Scientist, adapted)

### 4.1 Protocol

The LLM runs as an agent in one continuous conversation per hypothesis. Each reply must follow this format:

```text
ACTION:
CorpusSearch | FinalizeGapCard

ARGUMENTS:
{"query": "..."}                                  # for CorpusSearch
{"gap_card": {"hypothesis": "...",                # for FinalizeGapCard
              "evidence_doc_ids": ["..."],
              "importance": 0.0-1.0,
              "undocumentedness": 0.0-1.0,
              "opening_question": "..."}}
```

| Tool | Replaces (AI Scientist) | What it does |
|---|---|---|
| `CorpusSearch` | `SearchSemanticScholar` | Hybrid search over the local corpus (top 6, 600 chars each). Used both to find supporting evidence **and** to check whether the practice is already written down. No network |
| `FinalizeGapCard` | `FinalizeIdea` | Emits the gap card fields |

**System prompt (summary).** You are an analyst looking for decisions no document explains. Propose hypotheses about undocumented expert knowledge: repeated choices, priorities, exclusions or judgments that deviate from procedures or newcomer expectations. Ground each in specific documents, prefer recurring patterns a successor would get wrong, and don't propose things already stated. Search at least once before finalising.

### 4.2 Detailed flow for one cluster

```mermaid
flowchart TB
  START(["cluster briefing ws<br/>archive = []"]) --> G0{"gen_idx &lt; generations_per_cluster?"}
  G0 -- no --> RET(["return archive as cards"])
  G0 -- yes --> INIT["msg_history = []<br/>searched = False<br/>last_tool_results = ''"]
  INIT --> R0["round 0: GENERATION_PROMPT<br/>= briefing + 'hypotheses already generated: archive'<br/>+ 'generate a NEW hypothesis that differs'"]
  R0 --> FINALCHK

  subgraph ROUND["One round (repeated up to num_reflections + 1 times)"]
    direction TB
    FINALCHK{"last round?"} -- yes --> FORCE["append: 'final round: you MUST FinalizeGapCard now'"]
    FINALCHK -- no --> CALL
    FORCE --> CALL["LLM call (role: interviewer, task: gap_miner.ideation)<br/>seed = seed·1000 + gen_idx·10 + round"]
    CALL --> PARSE{"parse ACTION / ARGUMENTS<br/>(regex + fenced JSON)"}
    PARSE -- fail --> PERR["last_tool_results =<br/>'Could not parse … follow the format'"]
    PARSE -- ok --> ACT{"ACTION?"}
    ACT -- CorpusSearch --> SRCH["CorpusSearch(query)<br/>top 6 docs × 600 chars"] --> SET["last_tool_results = hits<br/>searched = True"]
    ACT -- FinalizeGapCard --> NS{"searched?<br/>or last round?"}
    NS -- no --> REFUSE["last_tool_results =<br/>'You must run at least one CorpusSearch before finalizing'"]
    NS -- yes --> VAL["drop evidence IDs not in corpus"]
    VAL --> OKC{"hypothesis present<br/>and ≥1 real evidence ID?"}
    OKC -- yes --> ARCH["archive.append(card)<br/>log {gen, rounds, card}"] --> DONE(["end this hypothesis"])
    OKC -- no --> DROP(["discard; end this hypothesis"])
    ACT -- other --> INV["last_tool_results = 'Invalid action'"]
  end

  PERR --> NEXT
  SET --> NEXT
  REFUSE --> NEXT
  INV --> NEXT
  NEXT{"rounds left?"} -- yes --> RN["REFLECTION_PROMPT (round r+1)<br/>'consider: does evidence support it? already documented?<br/>is the opening question concrete and cites evidence?<br/>Results from your last action: last_tool_results'"] --> FINALCHK
  NEXT -- no --> DONE2(["no card for this hypothesis"])

  DONE --> INC["gen_idx += 1"] --> G0
  DROP --> INC
  DONE2 --> INC
```

**Robustness**
- Any exception inside one hypothesis is caught and logged, and the loop continues with the next.
- Malformed replies cost one round but never crash the run.
- Every call is cached by seed, so re-running is free and deterministic.

### 4.3 What each mechanism contributes

| Mechanism | Effect on card quality |
|---|---|
| Archive of earlier cards in the prompt | Diversity: each new hypothesis must differ from earlier ones in the cluster |
| Forced search before finalising | The LLM must look for the rule in the documents instead of guessing that it is undocumented |
| Reflection with search results | Self-falsification: drop hypotheses the evidence does not support or that are already written |
| Evidence-ID validation | Removes hallucinated citations; cards with no real evidence are discarded |
| Final-round force | Bounds cost per hypothesis to `num_reflections + 1` calls |
| `opening_question` field | Turns the hypothesis into a correction question for the expert ("In H012 and H045 you did X when Y, unlike the procedure. Why?") |

### 4.4 Example trace (illustrative)

```text
user  (round 0)  # Cluster: Rooftop units … Hypotheses already generated: (none) … generate a new hypothesis
asst             ACTION: CorpusSearch  ARGUMENTS: {"query": "capacitor replaced on every rooftop unit visit"}
tool             [H0107] … swapped capacitor although it tested fine …  [H0115] … replaced capacitor preventively …
user  (round 1)  Round 2/4. Consider whether the evidence supports it, whether it is documented …  Results: …
asst             ACTION: FinalizeGapCard  ARGUMENTS: {"gap_card": {"hypothesis": "When servicing rooftop units older than
                 ~8 years, technicians replace the run capacitor even if it tests fine (likely to avoid summer call-backs)",
                 "evidence_doc_ids": ["H0107","H0115","H9999"], "importance": 0.8, "undocumentedness": 0.7,
                 "opening_question": "In H0107 and H0115 the capacitor was replaced although it tested fine. When do you
                 do that, and why?"}}
code             searched = True → allowed;  H9999 not in corpus → removed;  2 real IDs → archived (priority 0.56)
```

---

## 5. Stage 4: ranking and de-duplication

```mermaid
flowchart TB
  A[("raw cards from all clusters")] --> B["Build GapCard<br/>defaults: importance 0.5, undocumentedness 0.5<br/>opening_question ← hypothesis + ' Is that right?' if missing<br/>skip malformed"]
  B --> C["priority = importance × undocumentedness"]
  C --> D["sort by priority, descending"]
  D --> E["kept = []"]
  E --> F{"next card c"}
  F --> G{"similarity(c, k) ≥ threshold<br/>for any kept k?"}
  G -- yes --> SKIP["skip (lower-priority duplicate)"]
  G -- no --> H{"same evidence set<br/>as any kept k?"}
  H -- yes --> SKIP
  H -- no --> KEEP["kept.append(c)"]
  SKIP --> F
  KEEP --> F
  F -- done --> I["take kept[:top_k]"] --> J["rename G01 … Gnn"] --> K[("final gap cards")]
```

| Setting | Default | Note |
|---|---|---|
| `top_k` | 40 | Cards kept |
| `dedup_threshold` | 0.6 | Cosine similarity, used when `retrieval.embeddings: true` |
| `dedup_threshold_lexical` | 0.4 | Token overlap, used otherwise |

Because the list is sorted first, a duplicate pair always keeps the higher-priority card.

---

## 6. After mining: experiment use (Experiment 3)

```mermaid
flowchart LR
  GC[("gap cards")] --> L["LLM: neutral 3-6 word topic label per card<br/>(names the area, NOT the hypothesis)"]
  L --> SP["random shuffle (seeded)<br/>assign cards round-robin to arms B / C / D"]
  SP --> SJ[("split.json<br/>cards · labels · arm per card · blinded session order")]
  SJ --> B["B: only labels as focus areas"]
  SJ --> C["C: cards' opening questions + fixed follow-ups"]
  SJ --> D["D: cards as 'widen' actions in AB-MCTS"]
```

Neutral labels let a human expert, or a labels-only interviewer, work on the same topics without seeing the AI's hypotheses.

---

## 7. Baseline: `gap_miner_simple`

```mermaid
flowchart LR
  CL[("same clusters")] --> P["one prompt per cluster:<br/>briefing + 'propose up to N hypotheses' → JSON list"]
  P --> V["validate evidence IDs"] --> F["same finalize()"] --> O[("baseline cards")]
```

There are no tools, no reflection and no archive. Comparing the two (Experiment 2: precision and recall against verified rules) measures what the AI Scientist loop adds.

---

## 8. Parameters, cost and outputs

| Parameter (`config/default.yaml → gap_miner`) | Default | Effect |
|---|---|---|
| `min_cluster_size` | 3 | Smaller clusters are merged into `misc` |
| `generations_per_cluster` | 4 | Hypotheses attempted per cluster |
| `num_reflections` | 3 | Extra rounds per hypothesis (total rounds = 4) |
| `top_k` | 40 | Final cards |
| `dedup_threshold` / `dedup_threshold_lexical` | 0.6 / 0.4 | De-duplication strictness |

**LLM calls (upper bound):**

$$
\lceil N_{docs}/30 \rceil \;+\; N_{clusters} \times \text{generations\_per\_cluster} \times (\text{num\_reflections}+1)
$$

For example, 200 chunks in 10 clusters gives 7 + 10 × 4 × 4 = **167** calls at most. In practice there are fewer, because many hypotheses finalise in round 2.

| Output (under the experiment folder) | Contents |
|---|---|
| `gap_miner/clusters.json` | entity → document IDs |
| `gap_miner/workshops/NN.md` | Exact briefing per cluster |
| `gap_miner/ideation_log.json` | Every accepted card with its generation index and number of rounds used |
| `split.json` (Exp 3) or `results/gap_cards*.json` (Exp 2) | Final cards |

---

## 9. Known limitations and directions

- **Flat, bottom-up view.** Clusters are groups of documents, not a model of *principles → policies → decisions → details*. Cards are therefore isolated "undocumented practice" hypotheses, and interviews tend to collect isolated facts rather than reusable principles.
- **Self-assessed scores.** `importance` and `undocumentedness` are the LLM's own estimates. They are only checked later by the expert or by ground truth.
- **Procedure-centric task wording.** The briefing contrasts behaviour with procedures (`sop` documents). Corpora without procedures fall back to "what a newcomer would do".
- **Direction under consideration:** first build a knowledge tree from the corpus (goal → principles → policies → decisions → evidence), then type gaps by their position in the tree (missing principle, missing reason, missing boundary, orphan principle, contradiction). That makes each question explicitly target a principle, a reason, a boundary or an example.
