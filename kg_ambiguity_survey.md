# Graph-Grounded Ambiguity & Clarification Benchmark: Survey, Gap Analysis, Dataset Design

Status date: 2026-10-07. Starting point: KG-CoQA (HF `vikash0132/KG-CoQA`) and Abg-CoQA.

## 0. Provenance and limits (read first)

- **Tools available:** web search only. Direct fetches of arXiv, ACL Anthology, Hugging Face and alphaxiv were blocked by the network proxy. **I did not read any paper's full text.** Every fact about a paper comes from search-engine summaries of its abstract or metadata.
- **Tags:** `[S]` = metadata or abstract-level fact seen in search results this session. `[M]` = from memory, not re-verified. `[D]` = my own measurement on the KG-CoQA release (script: `analysis/kgcoqa_artifact_audit.py`).
- **Not done:** no Semantic Scholar or OpenReview citation tracing, no reading of reviews, no systematic coverage claim. About 60 queries run. Forward/backward citation tracing was limited to what surfaced in results.
- **KG-CoQA's own paper:** I could not find a paper, card or repo describing it. The HF page was blocked and searches returned nothing. Everything I say about KG-CoQA comes from the 1,000-row release. The pipeline description is yours.
- **Author lists** for a few well-known papers (ASQA, QuAC, QReCC, CSQA, ConvQuestions) are from memory; titles, venues and IDs were seen in results.
- **Ratings and probabilities** (novelty, P(incremental), P(strong paper)) are my judgement, not data.
- **Before writing Related Work:** read the full text of the six papers in §8.2, and run Semantic Scholar "cited-by" on Abg-CoQA, AmbigQA, DeepAmbigQA and MARCH.

---

## 1. Executive summary

1. **KG-CoQA as released cannot measure ambiguity. [D]**
   - Of the 500 ambiguous questions, all 500 contain a personal pronoun. Only 18/500 non-ambiguous ones do. A pronoun rule gets about 98% (a broader pronoun regex: 96.9%).
   - A question-only TF-IDF logistic regression, with 5-fold CV grouped by story, gets **97.6% accuracy / 97.6 macro-F1**. It never sees the passage or the KG.
   - Ambiguous questions average 4.9 words, non-ambiguous 9.75.
   - The label is a deterministic function of the triple count (`triples ≥ 2` gives 99.4%).
   - The release has no answers, no history and no interpretations.
   - Any "ambiguity detection" number on this release is a pronoun-detection number.
2. **Premise correction.** The release has **363 distinct stories** (1,000 questions, 500/500, every story contributes both classes). It does not have ~1,000 distinct stories.
3. **"KG controls validity, LLM controls surface form" is established, not novel.** Prior art: KQA Pro, KGConv, DeepAmbigQAGen, SQUAB, CLARITY, AMBROSIA, Ambig-DS. A reviewer will call "KG-CoQA but bigger/more natural" incremental.
4. **The field is crowded (2024–26).** CLAMBER, CondAmbigQA, DeepAmbigQA, MARCH, ClarifyMT-Bench, RegretBench, DiscoBench, AbstentionBench, CLARITY and others. A defensible paper needs a precise scientific question, not a pipeline.
5. **Recommended direction (§11):** a counterfactual, factorial, graph-grounded benchmark. It crosses surface form × evidence multiplicity × history × answer-divergence, with typed edits whose effect on the interpretation set is known by construction.
   - Natural CoQA turns are mined as seeds, so ambiguity is not only synthetic.
   - It tests whether models track the interpretation structure of the evidence or react to surface cues.
   - Honest novelty rating: Moderate-to-High, not Very High.
   - Closest threats: DeepAmbigQA/MARCH, the AmbigQA/CAmbigNQ/Abg-CoQA lineage, and CLARITY/AMBROSIA/SQUAB/AmbiCoref.
6. **The KG-CoQA audit is itself a publishable section.** Showing that a graph-generated ambiguity benchmark is solved by a pronoun rule motivates the factorial design.
7. **The weakest link is KG extraction recall.** A missed candidate turns a truly ambiguous item into a false "unambiguous". It has to be measured against exhaustive human enumeration (§16), not assumed.

---

## 2. Literature landscape

### 2.1 Ambiguous / underspecified QA (A)

- **Natural, human-annotated lineage.**
  - AmbigQA/AmbigNQ: 14,042 NQ-open questions, over half ambiguous, one QA pair + disambiguated rewrite per interpretation, F1_ans / EDIT-F1 metrics. Min et al., EMNLP 2020 [S] (aclanthology 2020.emnlp-main.466; arXiv 2004.10645).
   - ASQA: long-form answers resolving AmbigQA ambiguity. Stelmakh et al., EMNLP 2022 [S] (2022.emnlp-main.566).
   - **Abg-CoQA**: 8,615 questions / 994 ambiguous (≈11.5%) over 3,968 CoQA passages. MTurk workers flag ambiguity and write a clarification question plus all possible replies. Four types: coreference, time-dependency, answer-type, event. Guo, Zhang, Reddy, Alikhani, AKBC 2021 [S] (doi 10.24432/C5F30Z).
   - CAmbigNQ: 5,653 ambiguous questions, clarification questions drafted by InstructGPT then manually revised. Lee et al., Findings of EMNLP 2023 [S] (2023.findings-emnlp.772). Note: **not** Guo et al.
- **Entity-name ambiguity.**
  - AmbER: Wikidata/Wikipedia name-collision retrieval sets. Chen et al., ACL 2021 [S] (2021.acl-long.345).
  - AmbigDocs: Wikipedia disambiguation pages, questions with an ambiguous name. Lee, Ye, Choi, COLM 2024 [S] (arXiv 2404.12447).
  - RAMDocs: AmbigDocs plus misinformation plus noise. Preprint [S] (arXiv 2504.13079).
- **Linguistic ambiguity.**
  - AmbiEnt: 1,645 linguist-annotated NLI examples. GPT-4 disambiguations judged correct 32% vs 90% for gold. Liu et al., EMNLP 2023 [S] (2023.emnlp-main.51).
  - **AmbiCoref: minimal pairs of ambiguous vs unambiguous coreference.** Humans are less sure on ambiguous items; most coreference models barely differ. Yuan, Malaviya, Yatskar, Findings of EACL 2023 [S] (2023.findings-eacl.75). The closest prior art for the minimal-pair idea.
  - Correct-Detect: LLMs cannot do coreference disambiguation and ambiguity detection at the same time. Preprint [S] (arXiv 2509.14456).
- **Taxonomy-driven benchmarks.**
  - CLAMBER: ≈12K examples, taxonomy of 3 dimensions / 8 categories, five qualified experts validated. Zhang et al., ACL 2024 [S] (2024.acl-long.578; arXiv 2405.12063).
  - CondAmbigQA: 2,000 queries; "conditions" that resolve ambiguity found by retrieval-based annotation. EMNLP 2025 [S] (arXiv 2502.01523).
- **Structured-source controlled generation of ambiguity (outside QA-over-text).**
  - AMBROSIA: ambiguous questions to SQL with scope/attachment/vagueness, databases generated from scratch so ambiguity persists given the schema. Saparina & Lapata, NeurIPS 2024 D&B [S] (arXiv 2406.19073).
  - SQUAB: automatic generator of ambiguous/unanswerable semantic-parsing tests, SQL + LLM. EMNLP 2025 [S] (2025.emnlp-main.906).
  - CLARITY: constraint-driven SQL to ambiguous queries with single- and multi-turn continuations. ACL 2026 Industry [S] (arXiv 2604.22313).
  - Ambig-DS: pairs each fully-specified task with a controlled-edit ambiguous variant, human + LLM verified. Preprint, May 2026 [S] (arXiv 2605.09698).

### 2.2 Conversational QA (B)

- **CoQA:** 127k questions / 8k conversations / 7 domains. Notes coreference and pragmatic reasoning as challenging phenomena. Reddy, Chen, Manning, TACL 2019 [S] (Q19-1016). **Ambiguity is not annotated;** it occurs naturally and is not labelled.
- **QuAC:** 14K dialogs / 100K questions, student-teacher over hidden Wikipedia text. Questions often open-ended, unanswerable, or meaningful only in dialogue context. EMNLP 2018 [S] (D18-1241). No ambiguity labels.
- **QReCC:** 14K conversations / 81K QA pairs with human rewrites; CANARD rewrites QuAC questions (coreference, ellipsis). Anantha et al., NAACL 2021 [S] (arXiv 2010.04898). Rewriting is *resolution*, not *detection of irresolvability*.
- **TopiOCQA:** 3,920 conversations with topic switches [S] (arXiv 2110.00768).
- **Conversational KGQA:** CSQA (≈200K dialogs, AAAI 2018, arXiv 1801.10314); ConvQuestions/CONVEX (11,200 conversations over Wikidata, CIKM 2019, arXiv 1910.03262) [S]. CONVEX infers missing/ambiguous pieces by graph exploration but does not label ambiguity.
- **KGConv:** 71K conversations, 603,905 questions grounded in Wikidata facts, ≈12 question variants per fact (templates + human + rules + rewriting model). LREC-COLING 2024 [S] (arXiv 2308.15298). No ambiguity focus.
- **Survey:** "Disambiguation in Conversational QA in the Era of LLMs and Agents" (Tanjim et al., EMNLP 2025 main) [S] (arXiv 2505.12543; 2025.emnlp-main.482). Read it fully; it is the most direct competitor for any framing/taxonomy claim.

### 2.3 KGQA and graph-derived ambiguity (C)

- **CLAQUA:** ≈40K KB-based clarification examples (identify need, generate question, answer after feedback). Xu et al., EMNLP 2019 [S] (D19-1172). Oldest KB-grounded precursor.
- **PReFNet:** ambiguous KG queries via pseudo-relevance feedback. Liu et al., WWW 2023 [S].
- **CLEAR-KGQA:** Bayesian quantification of entity + intent ambiguity driving an AskForClarification tool, on WebQSP/CWQ. Wen et al., preprint [S] (arXiv 2504.09665).
- **AmbiGraph-Eval:** 560 ambiguous queries over graph databases (Cypher). Taxonomy: attribute / relationship / attribute-relationship ambiguity × same-entity / cross-entity. Preprint [S] (arXiv 2508.09631).
- **DeepAmbigQA:** 3,600 multi-hop questions, half with explicit name ambiguity, built by DeepAmbigQAGen from text corpora plus linked KGs. GPT-5 exact-match 0.13 on ambiguous vs 0.21 on non-ambiguous. Preprint [S] (arXiv 2511.01323).
- **MARCH:** 2,209 multi-hop ambiguous questions; multi-LLM verification + human validation; ambiguity can occur at any hop. Park et al., ACL 2026 Findings [S] (arXiv 2509.22750; 2026.findings-acl.1352).
- **Finding on the core question:** I did **not** find a paper that generates *conversational, passage-grounded* ambiguity from graph topology with interpretation sets, history conditioning and counterfactual edits. DeepAmbigQA/MARCH do graph/multi-hop ambiguity without conversational history and without clarification annotations as far as the abstracts say. This is "not found in this search", not "does not exist".

### 2.4 Clarification and interactive QA (D)

- **Policy learning:**
  - Zhang & Choi, "Clarify When Necessary" (arXiv 2311.09469; Findings NAACL 2025) [S].
  - Zhang, Knox, Choi, future-turn simulation for preference labels (arXiv 2410.13788; ICLR 2025) [S].
  - Chen et al., ACT (arXiv 2406.00222; ICLR 2025) [S].
  - APA: align LLMs with their own perceived ambiguity. Kim et al., EMNLP 2024 [S] (arXiv 2404.11972).
  - Saparina & Lapata, "Reasoning about Intent for Ambiguous Requests": RL to emit multiple interpretation–answer pairs; evaluated on conversational QA and semantic parsing (the Abg-CoQA search surfaced it, so likely Abg-CoQA, unverified). Preprint [S] (arXiv 2511.10453).
  - CLAIM: uncertainty-driven clarification with synthetic data. Preprint, Aug 2026 [S] (arXiv 2608.11631).
- **Information-theoretic clarification:** expected-information-gain question selection in interactive text-to-SQL [S] (arXiv 2507.06467); CLARINET for retrieval [S] (arXiv 2405.15784).
- **Multi-turn clarification benchmarks:**
  - ClarifyMT-Bench: 6,120 dialogues, five-dimensional taxonomy, six user personas, hybrid LLM–human; finds consistent **under-clarification** [S] (arXiv 2512.21120).
  - RegretBench: clarification as sequential policy with a regret objective [S] (arXiv 2607.21143).
  - DiscoBench: 211 samples / 463 ambiguity instances, 11 domains, user simulator; asking beats repeated searching [S] (arXiv 2606.27669).
- **IR clarification datasets:** Qulac, ClariQ (≈18K single-turn) [S] (arXiv 2009.11352).
- **Abstention:** AbstentionBench, 20 datasets / 35k+ queries; reasoning fine-tuning degrades abstention by 24% on average. Kirichenko et al., NeurIPS 2025 D&B [S] (arXiv 2506.09038).
- **Shortcoming:** simulated-user reliability is a known weak point (reported Sim2Real gaps in 2026 work; find and cite before relying on this). Our design avoids LLM simulators for the core clarification metric (§19).

### 2.5 LLM behaviour under ambiguity (E)

- Selective answering: sample-repetition confidence is the most reliable calibrator on ambiguous questions. Cole et al., EMNLP 2023 [S] (arXiv 2305.14613).
- Semantic entropy: Kuhn, Gal, Farquhar, ICLR 2023 [S] (arXiv 2302.09664); Farquhar et al., Nature 2024 [S].
- Aleatoric/epistemic decomposition via input clarification ensembling. Hou et al., ICML 2024 [S] (arXiv 2311.08718).
- "The Illusion of Certainty": UQ methods degrade to near-random under ambiguity; MAQA*/AmbigQA* with ground-truth answer distributions. Preprint [S] (arXiv 2511.04418).
- "From Answers to Interpretations": ambiguity-induced aleatoric uncertainty is better estimated in the *interpretation space* than the response space; reported accepted to Findings of EMNLP 2026 [S, from a secondary page; verify] (arXiv 2609.04543). **This paper wants gold interpretation sets, which is what we would supply.**
- Overconfidence/under-clarification: CLAMBER (CoT/few-shot give only marginal gains, overconfidence), ClarifyMT-Bench (premature answering), Correct-Detect, AbstentionBench [S].

---

## 3. Chronological development

| Period | Development |
|---|---|
| 2018–19 | CoQA, QuAC, CSQA: conversational context modelled, ambiguity not labelled. CLAQUA (2019): first KB-grounded clarification dataset. ConvQuestions. |
| 2019–21 | IR clarification (Qulac, ClariQ). **AmbigQA (2020)** makes interpretation sets + rewrites the unit of annotation. **Abg-CoQA (2021)** brings it to conversational QA with crowd-written clarifications. AmbER (entity collisions). |
| 2022–23 | ASQA. LLM era begins: AmbiEnt, AmbiCoref (minimal pairs), CAmbigNQ, selective answering (Cole), semantic entropy (Kuhn), PReFNet. |
| 2024 | Taxonomy-driven benchmarks (CLAMBER); structured-source generation (AMBROSIA); entity-collision with LLMs (AmbigDocs); alignment (APA); uncertainty decomposition (Hou); KGConv. |
| 2025 | Multi-answer/conditions (CondAmbigQA); generated ambiguity at scale (SQUAB, DeepAmbigQA); abstention (AbstentionBench); graph-query ambiguity (AmbiGraph-Eval, CLEAR-KGQA); multi-turn (ClarifyMT-Bench); survey (EMNLP). |
| 2026 | Multi-hop ambiguity (MARCH); constraint-driven ambiguity with multi-turn continuations (CLARITY); agentic clarification (DiscoBench, Ambig-DS); clarification as policy (RegretBench); interpretation-space UQ. |

Pattern: natural-ambiguity annotation → LLM overconfidence benchmarks → structured-source generation → multi-turn/agentic clarification policy and uncertainty decomposition. The unfilled slot is **controlled, structure-labelled ambiguity in conversation, with ground truth that supports causal tests**.

---

## 4. Dataset comparison table

Columns: Nat = natural vs Syn = synthetic. "?" = could not verify.

| Dataset | Year · Venue | Source · Size | Turns | Ambiguity types | Nat/Syn | KG? | Clarif.? | Multi-interp.? | Human? | LLM-gen? | Main limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CoQA | 2019 TACL | 8k convs, 127k Q | multi | none labelled (coref noted) | Nat | no | no | no | yes | no | ambiguity unlabelled |
| QuAC | 2018 EMNLP | 14K dialogs | multi | none labelled | Nat | no | no | no | yes | no | same |
| QReCC / CANARD | 2021 NAACL | 14K convs / 81K QA | multi | via rewrites | Nat | no | no | no | yes | no | resolves, doesn't flag irresolvable |
| AmbigQA | 2020 EMNLP | NQ-open, 14,042 | single | event/entity refs etc. | Nat | no | no (rewrites) | **yes** | yes | no | open-domain Wikipedia, single turn, pre-LLM |
| ASQA | 2022 EMNLP | AmbigQA-derived | single | same | Nat | no | no | yes (long-form) | yes | no | long-form evaluation |
| CLAQUA | 2019 EMNLP | ≈40K | single | entity | mixed ? | **KB** | yes | ? | ? | no | KB-only, entity-centric, pre-LLM |
| **Abg-CoQA** | 2021 AKBC | 8,615 Q / 994 amb. | CoQA conv | coref, time, answer-type, event | Nat | no | **yes** | replies only | yes (MTurk) | no | ~11.5% ambiguous, 994 positives, no formal interpretation set/graph, pre-LLM |
| CAmbigNQ | 2023 Findings EMNLP | 5,653 amb. | single | AmbigNQ | Nat | no | yes | yes | yes (revised) | yes (InstructGPT) | single-turn, NQ |
| CLAMBER | 2024 ACL | ≈12K | single | 3 dims / 8 cats | mixed | no | yes | no | 5 experts | yes | taxonomy coarse; no graph evidence |
| AmbiEnt | 2023 EMNLP | 1,645 | sentence pairs | lexical/syntactic/etc. | Nat/ling. | no | no | yes (disambiguations) | linguists | no | NLI, small |
| AmbiCoref | 2023 Findings EACL | minimal pairs | sentence | coref | Syn | no | no | no | yes | no | sentence-level, not QA |
| AmbigDocs | 2024 COLM | Wikipedia disamb. pages | single | entity names | Syn | partial | no | yes | ? | yes | multi-doc, no conversation |
| RAMDocs | 2025 preprint | AmbigDocs ext. | single | + misinformation/noise | Syn | no | no | yes | ? | ? | conflict ≠ question ambiguity |
| CondAmbigQA | 2025 EMNLP | 2,000 queries | single | conditions | Nat/retrieval | no | no | yes | yes | ? | single-turn |
| **DeepAmbigQA** | 2025 preprint | 3,600 | single | name ambiguity + multi-hop | Syn | **yes** | no | yes | ? | yes | no conversation/clarification (per abstract) |
| **MARCH** | 2026 ACL Findings | 2,209 | single | multi-hop layered | Syn | ? | agentic | yes | yes | yes | single-turn |
| AmbiGraph-Eval | 2025 preprint | 560 | single | attr/rel × same/cross-entity | Syn | graph DB | no | yes | ? | ? | query-gen task, small |
| AMBROSIA | 2024 NeurIPS D&B | text-to-SQL | single | scope/attach./vagueness | Syn | DB | no | **yes** | yes | partial | SQL modality |
| SQUAB | 2025 EMNLP | generator | single | ambiguous + unanswerable | Syn | DB | no | yes | no | yes | SQL modality |
| CLARITY | 2026 ACL Industry | NL2SQL | single + multi | multi-faceted | Syn | DB | yes (continuations) | yes | ? | yes | SQL modality |
| ClarifyMT-Bench | 2025 preprint | 6,120 dialogues | **multi** | 5 dims | Syn | no | yes | ? | hybrid | yes | no graph evidence |
| RegretBench | 2026 preprint | open-domain QA + recs | multi | hidden intent | Syn | no | yes | ? | ? | ? | policy-eval focus |
| DiscoBench | 2026 preprint | 211 / 463 inst. | multi-step | 4 types | mixed | no | yes | ? | yes | ? | small |
| Ambig-DS | 2026 preprint | 112 tasks | agent | framing (target/objective) | Syn edits | no | 1 question | no | yes | yes | data-science tasks |
| AbstentionBench | 2025 NeurIPS D&B | 20 datasets, 35k+ | single | underspec. etc. | agg. | no | no (abstain) | no | – | – | aggregate, abstention only |
| MAQA*/AmbigQA* | 2025 preprint | – | single | multi-answer | – | no | no | answer distributions | – | – | UQ-oriented |
| KGConv | 2024 LREC-COLING | 71K convs | multi | none | Syn | **Wikidata** | no | no | partial | rewriting model | no ambiguity |
| KQA Pro | 2022 ACL | ≈120K | single | none | templates+crowd | **KB** | no | no | crowd paraphrase | no | no ambiguity (but a template→paraphrase precedent) |
| **KG-CoQA** (release) | ? | 1,000 Q / 363 stories | single | multi-parent only | Syn | **yes** | no | no | ? | ? | pronoun artifact, no answers/history/interpretations |

---

## 5. Ambiguity taxonomy

Ratings are my judgement. D = deterministic from a KG; L = LLM needed (at least for surface form); H = needs human judgement.

| # | Category | Definition | Example | Existing coverage | In CoQA naturally? | KG-gen? | Annot. difficulty | Usefulness |
|---|---|---|---|---|---|---|---|---|
| 1 | Entity (name collision) | mention matches ≥2 entities | "When did Sam join?" (two Sams) | AmbER, AmbigDocs, DeepAmbigQA, CLAQUA | yes (names) | D | low | high |
| 2 | Referential / coreference | pronoun/description has ≥2 admissible antecedents | "What does he do?" | **Abg-CoQA (majority)**, AmbiCoref, Correct-Detect, KG-CoQA | **yes, abundant** | D candidates, H salience | moderate (salience is graded) | high |
| 3 | Relation | predicate underdetermines the edge | "Who is she connected to?" | AmbiGraph-Eval | rare | D given overlapping schema, L phrasing | moderate | medium |
| 4 | Multi-parent KG | several subjects share relation/object | KG-CoQA's only type | KG-CoQA | n/a | D | low | **not a linguistic type** (see below) |
| 5 | Multi-hop / path | ambiguity at hop *k* propagates | "Who does the report's author report to?" | MARCH, DeepAmbigQA | rare | D | moderate | high |
| 6 | Temporal | time frame underspecified | "What was everyone doing?" | Abg-CoQA | yes (narratives) | D if times extracted (hard) | moderate | medium |
| 7 | Spatial | location frame underspecified | "near the station" | weak | occasional | partial | moderate | low |
| 8 | Numerical | which count/unit/scope | "How many attended?" (registered vs day 1 vs day 2) | weak | yes | partial | moderate | medium |
| 9 | Event | multiple similar events | "When did it open?" (opening vs reopening) | Abg-CoQA, AmbigQA | yes | needs event extraction (noisy) | moderate | medium |
| 10 | Scope | quantifier/negation scope | "Did every student read a book?" | AMBROSIA, AmbiEnt | rare | no | high (L, H) | low for passage QA |
| 11 | Lexical | polysemy | "bank" | AmbiEnt, CLAMBER | occasional | no (sense inventory) | moderate | low here |
| 12 | Syntactic | attachment | "saw the man with the telescope" | AMBROSIA, AmbiEnt | rare | no | high | low here |
| 13 | Pragmatic | implicature/vagueness | "Is it big?" | CLAMBER, AMBROSIA (vagueness) | occasional | no | high | low |
| 14 | Conversational-context | meaning depends on history | "And his brother?" | ClarifyMT-Bench | **yes** | D if dialogue built from graph | moderate | **high** |
| 15 | Missing argument / ellipsis | omitted slot | "And the last one?" | CANARD/QReCC (resolve), CLAMBER ? | **yes** | D | moderate | high |
| 16 | Multiple valid answers | answer-level multiplicity | "Who are the Beatles?" | ASQA, MAQA*, AmbigQA | yes | D | low | separate axis, not an interpretation property |
| 17 | Answer granularity | hierarchy of correct specificity | "Where was he born?" (suburb/city/state) | Abg-CoQA (answer type) | **yes** | **D** (located-in chains) | low | medium (usually harmless) |
| 18 | World-knowledge | needs outside knowledge | "which Jordan?" | AmbER, AmbigDocs | – | – | moderate | out of scope for passage QA |
| 19 | Contradictory evidence | sources disagree | two opening times | RAMDocs | – | D by injecting edges | low | separate axis (evidence quality) |
| 20 | Cross-turn | introduced/propagated over turns | topic shift then "it" | ClarifyMT-Bench, RegretBench, DiscoBench | yes | D if built from walks | high | high, **weakly explored** over passage/graph |

### Revised taxonomy: three orthogonal axes, not 20 flat classes

1. **Source of underspecification** (what slot is open): referent, relation, path-hop, time, event, quantity/scope, granularity.
2. **Resolution condition** (what resolves it):
   - R0: nothing needed; the question is fully specified.
   - R1: type/selectional constraints and passage content resolve it.
   - R2: dialogue history resolves it.
   - R3: only the user can resolve it.
3. **Consequence:** decision-relevant (interpretations give different answers) vs **harmless** (answer-invariant, or answers nested by granularity).

Moves: #4 (graph pattern, not linguistic type), #16 (answer property), #19 (evidence quality) are dropped as ambiguity *types*; #10–13 are out of scope for passage+KG and listed as an extension; #14 and #20 become the *resolution* axis, not types.

---

## 6. Limitations of Abg-CoQA

Stated from abstract-level facts [S]; verify against the full text.

- **Small positive set:** 994 ambiguous (≈11.5%). Too small to train or stratify by type; type counts are tiny.
- **Pre-LLM.** Best baselines F1 23.6 on ambiguity detection, 40.1 on QA after clarification. Not informative about modern LLM behaviour.
- **Crowd judgement of ambiguity.** MTurk workers decide ambiguity. Human label variation is probably high but unmodelled (Pavlick & Kwiatkowski, TACL 2019 [S via summary]; Plank, EMNLP 2022 [S]).
- **No formal interpretation set, no graph, no evidence spans.** Replies exist, but not a structured set with supporting evidence per interpretation.
- **No counterfactuals.** Cannot test whether models react to ambiguity or to surface cues.
- **No "harmless ambiguity" class** (to my knowledge from the abstract): over-clarification cannot be measured.
- **Source dependence:** CoQA stories are likely in LLM pretraining data; contamination unmeasured.

---

## 7. Limitations of the current KG-CoQA dataset

Measured [D] unless marked inferred.

| # | Issue | Evidence |
|---|---|---|
| 1 | **Label ≡ surface form.** Ambiguous questions: 500/500 contain a personal pronoun. Non-ambiguous: 18/500. | pronoun rule ≈98%; question-only LR 97.6% acc / 97.6 macro-F1 (char n-gram: 97.3), 5-fold, grouped by story |
| 2 | **Label ≡ triple count.** Ambiguous rows have ≥2 triples; 495/500 non-ambiguous have exactly 1. | `triples ≥ 2` rule: 99.4% |
| 3 | **Length shortcut.** 4.9 vs 9.75 words. | `len ≤ 6` rule: 92.3% |
| 4 | **Every ambiguous item has one shared relation and distinct subjects and objects** (499/500). | Answer-invariant multi-parent cases never occur, so there are no "harmless ambiguity" negatives. |
| 5 | **No hard negatives:** no pronoun + unique antecedent, and no full-name + collision. | the 2×2 (surface × multiplicity) has only two diagonal cells populated |
| 6 | **Low lexical diversity.** Only 333 distinct questions among 500 ambiguous; "Where is it located?" ×32; top starters what/who/where/when. | |
| 7 | **No answers, history, interpretations, clarifications, evidence spans** in the release. Cannot be used for QA, enumeration or clarification. | schema: question_number, coqa_story, question, context, based_on_pattern, is_ambiguous |
| 8 | **"context" is a verbalised KG, not the story.** Ambiguous items have a longer context (17 vs 9 words). | the model can solve from KG text alone |
| 9 | **Relation vocabulary fragmentation:** 670 distinct relation strings in 1,000 rows (`works for` / `works at`; `spouse of` / `married to`; `has profession` / `has occupation`). Relation overlap is not normalised. | |
| 10 | **Story diversity:** 363 distinct stories; 266 stories contribute exactly 2 questions. | |
| 11 | **Single-turn:** "conversational" is nominal; no history, no discourse salience. | |
| 12 | KG extraction accuracy/recall, human verification, LLM-generation provenance: **unknown to me** (no pipeline description available). | |

**Why multi-parent topology ≠ linguistic ambiguity.** A single-turn "Where is it located?" is "ambiguous" under 4 candidates but would be resolved by humans through salience when one candidate was just discussed. Psycholinguistic work (generalised by AmbiCoref) shows ambiguity depends on verb/argument structure, not candidate count alone. Topology gives candidate *enumeration*; it does not give *probability* or *resolvability*.

---

## 8. Research gaps

### 8.1 Gap classification

| Potential gap | Status in literature | Class |
|---|---|---|
| Ambiguity from simple multi-parent topology | DeepAmbigQA, AmbiGraph-Eval, CLAQUA, MARCH | Partially explored (not a contribution) |
| Template-like synthetic questions | KQA Pro, KGConv (paraphrase pipelines); DeepAmbigQAGen (natural verbalisation) | Partially explored |
| Surface-cue confound of the ambiguity label | measured here for KG-CoQA; no systematic cross-benchmark audit found | **Weakly explored, strong opportunity** (as an audit + design principle) |
| Lack of ambiguity severity / interpretation probability | MAQA* distributions; Hou et al.; interpretation-space UQ | Weakly explored |
| No taxonomy | CLAMBER, Abg-CoQA (4 types), ClarifyMT (5 dims), AmbiGraph-Eval | Solved generally; **not** graph-grounded with a resolvability axis |
| Multi-hop ambiguity | MARCH, DeepAmbigQA | Partially explored |
| Conversational ambiguity propagation | ClarifyMT-Bench, RegretBench, DiscoBench | Partially explored, but **not passage/graph-grounded** |
| Clarification questions | CAmbigNQ, CLAQUA, CLAMBER, Abg-CoQA, ClarifyMT | Partially solved |
| Canonical interpretation sets | AmbigQA, CondAmbigQA, Saparina & Lapata | Partially explored |
| Explanation of *why* ambiguous | taxonomy labels only (CLAMBER); rewrites (AmbigQA) | Weakly explored |
| Supporting KG evidence per interpretation | not verified in DeepAmbigQA/MARCH | Weakly explored |
| Human verification | standard practice | Solved in general (unknown for KG-CoQA) |
| **Controlled minimal pairs at passage/QA scale** | AmbiCoref (sentence-level coref), Ambig-DS (agent tasks) | **Weakly explored, strong opportunity** |
| **Counterfactual evidence/history edits with known interpretation-set effect** | none found | **Strong dataset opportunity** |
| **Harmless (answer-invariant) ambiguity as hard negative** | Ambig-DS requires "decision-relevant consequences"; otherwise unaddressed | **Weakly explored, strong opportunity** |
| **History-resolved vs history-unresolved status of the same question** | Correct-Detect (resolvable vs not, sentence-level); not in graph-grounded QA | **Weakly explored, strong opportunity** |
| Gold clarification utility computable from structure | EIG in text-to-SQL, CLARINET | Partially explored; open for text QA with gold interpretation sets |
| Effect of KG extraction recall on ambiguity labels | unmeasured in ambiguity datasets | **Open; strong opportunity** (needs exhaustive human enumeration) |
| Contamination of CoQA stories | not addressed | Weakly explored |
| Modern-LLM evaluation | extensive (CLAMBER, ClarifyMT, AbstentionBench, …) | **Already solved**; not a contribution |
| Larger / more stories | trivial | Not novel |

### 8.2 The three papers/datasets most threatening to novelty

1. **DeepAmbigQA (arXiv 2511.01323) + MARCH (ACL 2026 Findings).** Systematic, KG-linked ambiguity generation and multi-hop ambiguity with multi-LLM + human verification. Kills "first to generate ambiguity systematically from graph structure" and "first multi-hop ambiguity".
2. **AmbigQA / CAmbigNQ / Abg-CoQA, plus Saparina & Lapata's interpretation–answer-pair training.** Interpretation sets, clarification annotations and conversational QA on CoQA already exist. "Interpretation set" and "clarification layer" are not novel in themselves.
3. **AMBROSIA / SQUAB / CLARITY (+ AmbiCoref, Ambig-DS).** Controlled generation from formal structure with interpretation sets, paired ambiguous/unambiguous variants, multi-turn continuations. Different modality, but a reviewer will say "same pattern".

Runner-ups: ClarifyMT-Bench and RegretBench (multi-turn clarification policy evaluation), CLAMBER (taxonomy plus clarification), the EMNLP 2025 disambiguation survey (any framing claim).

### 8.3 What *is* missing (the real gap)

Existing resources test whether a model **detects** or **resolves** ambiguity on items whose ambiguity labels are confounded with, or at best loosely tied to, the interpretation structure of the evidence. None found offers **causal control**: the same question with the evidence, history or surface form edited so that the interpretation set changes (or doesn't) in a known way.

Without that control one cannot say whether a model's ambiguity behaviour tracks structure (candidate count, history, answer divergence) or cues (pronouns, vague words, question length). §1 shows a pronoun rule recovers 98% of a published-style graph-generated label.

---

## 9. Candidate dataset ideas

Ratings: novelty Low/Moderate/High/Very High; P = subjective probability.

### P1 — CF-Ambig: counterfactual factorial benchmark (surface × multiplicity × history × answer-divergence)
- **Core idea:** natural CoQA turns as seeds; KG-defined interpretation sets; typed edit operators that change |B|, history resolution, answer divergence or surface form with *known* effect; matched families of 4–12 variants per seed.
- **Gap:** counterfactual control, harmless ambiguity, history-resolved vs unresolved (§8.1).
- **vs Abg-CoQA:** adds formal interpretation sets, graph evidence, counterfactual families, harmless class; model-agnostic ground truth.
- **vs KG-CoQA:** breaks the label–surface confound by design; adds history, answers, interpretations.
- **vs AmbigQA/closest:** AmbigQA gives one interpretation set per natural question; here the *same* question appears across edits. Closest: AmbiCoref (sentence-level), Ambig-DS (tasks), AMBROSIA (SQL).
- **Construction:** §12. **Annotations:** interpretation set, per-interpretation answer + evidence spans, resolvability, decision relevance, gold clarification. **Tasks:** detection, enumeration, answer-set, clarification, selective answering, counterfactual sensitivity.
- **Novelty:** **Moderate–High.** Difficulty: Moderate-High. Cost: Moderate-High. Risks: LLM-edited passages look unnatural; extraction recall; reviewers say "synthetic". P(reviewers call incremental) ≈ 30%. P(strong paper) ≈ 40–45%.

### P2 — History-conditioned ambiguity status in passage-grounded dialogue
- **Core idea:** per-turn labels {unambiguous, resolved-by-history, unresolved, harmless}; clarification answers change later turns' status; dialogues from graph walks over CoQA stories.
- **Gap:** cross-turn propagation over passage/graph evidence. **Closest:** ClarifyMT-Bench, RegretBench, DiscoBench (not passage/graph-grounded).
- **Construction:** generate turns from KG walks, LLM verbalisation, human verification of status labels.
- **Novelty:** High. Difficulty: High (multi-turn validity, salience is graded). Cost: High. Risk: status labels depend on defeasible salience judgements. P(incremental) ≈ 35%. P(strong paper) ≈ 35%.
- **Verdict:** fold a restricted version into P1 (history factor) instead of standing alone.

### P3 — Interpretation-probability benchmark (ambiguity as distribution)
- **Core idea:** ≥10 human raters per item give salience distributions over interpretations; KG supplies candidates; evaluate model-induced distributions vs human (JS divergence, calibration).
- **Gap:** weak severity/probability annotation; MAQA* and interpretation-space UQ need gold distributions.
- **Novelty:** Moderate–High. Difficulty: High (rater load, validity of "probability"). Cost: High. Risk: raters disagree for reasons unrelated to ambiguity. P(incremental) ≈ 30%. P(strong) ≈ 30%.
- **Verdict:** include as a subset (≈500 items) of P1, not a standalone.

### P4 — Gold clarification utility from structure
- **Core idea:** compute expected information gain (EIG) of every candidate clarification as a partition of the interpretation set; compare model-asked clarifications; human responses validate deterministic simulation.
- **Closest:** EIG for text-to-SQL (arXiv 2507.06467), CLARINET, RegretBench.
- **Novelty:** Moderate. Difficulty: Moderate. Cost: Low-Moderate. Risk: reviewers see it as application of known EIG. P(incremental) ≈ 50%. P(strong) ≈ 25%.
- **Verdict:** an annotation layer and metric inside P1.

### P5 — Multi-hop path ambiguity in conversation
- **Closest:** MARCH, DeepAmbigQA. **Novelty:** Low–Moderate. P(incremental) ≈ 60%. P(strong) ≈ 15%.
- **Verdict:** include multi-hop only as one source category.

### P6 — KG-CoQA-large (scale-up with LLM verbalisation + human check)
- **Novelty:** Low. P(incremental) ≈ 80%. P(strong) ≈ 5–10%. Reject.

### P7 — Natural-ambiguity mining benchmark across CoQA/QuAC/TopiOCQA with KG-assisted candidate retrieval
- **Core idea:** use the KG to find *natural* ambiguous turns in existing conversational QA corpora, human-verify; report prevalence.
- **Gap:** ambiguity unlabelled in CoQA/QuAC/TopiOCQA/QReCC. **Closest:** Abg-CoQA (crowd-flagged).
- **Novelty:** Moderate (prevalence + type distribution at scale is genuinely informative). Difficulty: Moderate. Cost: Moderate. Risk: low yield, so small positives. P(incremental) ≈ 45%. P(strong) ≈ 25%.
- **Verdict:** serves as the seed source in P1.

### P8 — Evidence-conflict + ambiguity disentanglement
- **Core idea:** separate question ambiguity from evidence conflict. **Closest:** RAMDocs. **Novelty:** Moderate. P(incremental) ≈ 50%. P(strong) ≈ 15%.
- **Verdict:** include as distractor class only.

---

## 10. Novelty ranking

| Rank | Proposal | Novelty | Feasibility | Paper potential | Notes |
|---|---|---|---|---|---|
| 1 | **P1 CF-Ambig** (absorbing P2-lite, P3-subset, P4, P7) | Moderate–High | Moderate | **Highest** | tests a falsifiable hypothesis |
| 2 | P2 history-conditioned status | High | Low-Moderate | Medium | evaluator validity hardest |
| 3 | P3 interpretation probabilities | Moderate–High | Low | Medium | expensive; validity argument needed |
| 4 | P7 natural mining | Moderate | Moderate | Medium-Low alone | needed as seed |
| 5 | P4 clarification utility | Moderate | High | Low alone | cheap add-on |
| 6 | P8 conflict vs ambiguity | Moderate | High | Low | |
| 7 | P5 multi-hop | Low–Moderate | High | Low | MARCH preempts |
| 8 | P6 scale-up | Low | High | Very low | reject |

Items flagged probably-incremental by ACL/EMNLP reviewers: P4 alone, P5, P6, P7 alone, and any pitch of the form "KG + LLM generation pipeline" as the main contribution.

---

## 11. Recommended dataset design: CF-Ambig

### Why this one

- **It tests a hypothesis, not just supplies data.** "Models' ambiguity behaviour is driven by surface form more than by the interpretation structure of the evidence." The KG-CoQA audit shows a benchmark where that is trivially true of the *classifier*.
- **A KG is necessary, not decoration.** The graph is the oracle that enumerates candidates, certifies exhaustiveness (up to extraction recall), defines edits with known effect, and yields answer sets and EIG deterministically. A purely LLM- or human-built benchmark gives no ground truth for exhaustiveness or counterfactual control.
- **It survives the reviewer attack "this is synthetic."** Seeds are natural CoQA turns; edits are small; a natural-transfer experiment (train on synthetic, test on Abg-CoQA/CAmbigNQ) bounds ecological validity.
- **It is cheaper than P2/P3 and less pre-empted than P5/P6.**

### Is the L0–L5 hierarchy justified? No.

It mixes three axes: source (entity/relation/path), resolvability (history vs none) and multiplicity. A linear scale asserts ordinal difficulty nobody has measured. L4 ("history resolves") and L5 ("even with history, clarify") are values of *resolution condition*; L1–L3 are *sources*. **Use the three orthogonal axes of §5.** Derive "difficulty" empirically (human accuracy, model disagreement) and report it as a post-hoc statistic. If a single ordinal is wanted, define it from measured model/human behaviour.

### Factorial design

| Factor | Levels |
|---|---|
| Surface form of the referring expression | pronoun · definite description · bare name |
| Evidence multiplicity \|B\| | 1 · 2 · 3+ |
| History | absent · neutral · resolving · misleading (recent mention of wrong candidate) |
| Answer divergence | divergent · invariant (harmless) · nested (granularity) |

Not all cells are crossed. A **seed family** has a natural seed plus 3–8 variants from typed edits. Each cell is populated with items whose labels were *not* predictable from surface features; a held-out acceptance test: question-only and pronoun-only classifiers must be at chance *within each stratum* (§16).

### Typed edit operators (effect on the interpretation set is known by construction)

| Op | Effect | Realisation |
|---|---|---|
| E-add | add a type-compatible competitor: \|B\|: 1→2 | LLM inserts one sentence introducing the competitor |
| E-del | remove competitor: 2→1 | delete or rename (sentence-level) |
| H-resolve | history turn that selects one candidate (explicit or salient) | LLM-written turn; or take the real CoQA history |
| H-erase / H-mislead | remove history / place wrong-candidate mention last | |
| S-pron / S-name | change surface form, bindings unchanged | rule + LLM |
| A-collapse | make interpretations' answers equal (e.g., same city) | edit facts |
| A-diverge | the reverse | |
| Q-para | paraphrase; invariance control | |

---

## 12. Complete generation pipeline

```
CoQA (+ fresh post-cutoff stories) ──► KG extraction (≥2 extractors, evidence spans)
   ──► seed mining on natural turns: compute B(q,G,H), |B|, ans(B)
   ──► edit planning (spec JSON: operator, target bindings, surface constraints)
   ──► LLM realisation under hard constraints (insert-sentence / history-turn / question)
   ──► structural check (re-extract edited passage; diff must reproduce intended B)
   ──► answerability/NLI check per interpretation (≥2 verifiers)
   ──► locality check (unchanged spans preserved; no new unintended candidates)
   ──► artifact filter (question-only / pronoun-only probes; adversarial filtering)
   ──► LLM critics (≥3 families: ambiguous? enumerate interpretations)
   ──► human verification (all test items; audit sample of train)
   ──► gold clarification + EIG (partition of B; verbalised by LLM; validated)
```

Stage notes:

1. **Sources.** CoQA train/dev stories (check per-domain licences: CoQA's domains carry different licences [M]). Add a *fresh-source split* from post-2025 text to bound pretraining contamination; add entity-renaming probes.
2. **KG extraction.** Run two extractors from different model families, take the union with confidence, normalise to ≈30–50 relation types (KG-CoQA's 670 strings in 1,000 rows shows what happens otherwise), keep character-offset evidence for every node/edge, and perform coreference clustering over passage *and* history. Measure precision/recall on an exhaustively human-annotated audit of ≈200 stories (§16).
3. **Seed mining.** For each CoQA turn with a referring expression or ellipsis, compute admissible bindings given history; keep turns with |B_0|>1 (natural candidates) and a random sample of |B|=1 (natural negatives). Humans verify before use.
4. **Spec → realisation.** The KG spec is deterministic; the LLM only verbalises under constraints: forbid disambiguating cues, fix register, fixed names. This is where the pipeline is *not* novel (KQA Pro, KGConv, DeepAmbigQAGen, SQUAB, CLARITY do the same). It is the substrate, not the contribution.
5. **Validation:** §16.
6. **Clarification gold.** For each interpretation set derive candidate clarifications as attribute partitions (role, location, relation). Compute EIG (§13). LLM verbalises; humans rate naturalness and unambiguity. User responses are **deterministic functions of the gold interpretation** (no LLM simulator), validated on a human-response subset.

---

## 13. Mathematical formulation

Everything below is a standard construction assembled for this setting. Nothing is claimed as new; each piece has a cited ancestor.

**Setting.** Passage D, extracted graph G(D), history H, question q.

**Bindings.** A semantic frame f(q) with open slots (referent, relation, time, granularity). B(q,H,G) = set of assignments consistent with hard constraints (type/selectional restrictions, agreement, explicit antecedents, history). An *interpretation* is a binding b ∈ B; its answer is ans(b;G).

**Status** (the resolvability axis):

- |B_0| = 1, with B_0 = bindings without history ⇒ **unambiguous**
- |B_0| > 1 and |B_H| = 1 ⇒ **resolved by history**
- |B_H| > 1 and |ans(B_H)| = 1 (up to equivalence) ⇒ **harmless**
- |ans(B_H)| > 1 ⇒ **decision-relevant ambiguity** (needs clarification)

Candidate count |I| > 1 alone is insufficient: it conflates harmless and decision-relevant cases and ignores probability.

**Interpretation prior** π(b | q,H,D). Three estimators, reported side by side: uniform; human salience distribution (≥10 raters on a subset); model-induced. Answer distribution P(a) = Σ_{b: ans(b)=a} π(b).

**Ambiguity score.** A(q) = H(P) = −Σ_a P(a) log P(a): semantic entropy at the interpretation level (cf. Kuhn et al. 2023; Hou et al. 2024; "From Answers to Interpretations"). Related quantities:

- Normalised: A / log|ans(B)|.
- Effective interpretations: N_eff = exp(A).
- **Premature-commitment risk:** 1 − max_a P(a) = error rate of a commit-to-one policy.
- **Decision rule:** clarify iff (1 − max_a P(a)) × loss_wrong > cost_ask.
- Optional semantic-distance weighting via Rao's quadratic entropy Q = Σ_{a,a'} P(a)P(a')d(a,a') [M: C. R. Rao, 1982; not searched this session]. d may use graph edit distance between subgraphs or answer-embedding distance. Use only if it improves agreement with human severity ratings.

**Clarification utility.** For clarification c with response variable R_c = f_c(I):
EIG(c) = H(P) − Σ_r P(r|c) H(P|r).
For deterministic f_c this equals H(f_c(I)). Normalised utility = EIG(c)/max_c' EIG(c'). Cost-aware: EIG(c)/cost(c). Expected turns to resolve ≥ ⌈log_k |I|⌉ for k-way questions. Closest: the EIG framework in interactive text-to-SQL and RegretBench's regret.

**Edit-consistency axioms** (pipeline validation and behavioural tests):

- M1: E-add increases |B| by exactly 1 and weakly increases A under uniform π.
- M2: H-resolve decreases |B_H| to 1 (when the history is resolving).
- M3: S-name/S-pron leaves B unchanged.
- M4: A-collapse sets |ans(B)| = 1, so A = 0.

Models are evaluated on **monotonicity violation rate**; the pipeline is evaluated on whether re-extraction reproduces the intended B.

---

## 14. Dataset schema

Changes from the draft schema: separate `interpretations[]` as objects with evidence; explicit `history`; `family_id` and `edit` for counterfactuals; `resolvability`; `decision_relevant`; priors from three estimators; per-annotator labels (do not collapse disagreement); provenance and contamination fields; clarification options keyed to interpretation IDs.

```json
{
  "id": "cfa-000412-v03",
  "family_id": "cfa-000412",
  "source": {"corpus": "coqa", "story_id": "...", "turn_index": 4, "fresh_split": false},
  "passage": "Tomás and Idris shared a flat in Porto. Tomás worked nights at a bakery; Idris repaired bicycles. ...",
  "history": [{"role": "user", "text": "What does Ade do?"}, {"role": "assistant", "text": "..."}],
  "question": "What does he do for a living?",
  "surface": {"referring_expression": "pronoun", "form": "he"},
  "edit": {"op": "H-erase", "parent_id": "cfa-000412-v01", "edited_spans": [[0, 0]]},
  "knowledge_graph": {
    "nodes": [{"id": "n1", "name": "Tomás", "spans": [[0, 5]]}, {"id": "n2", "name": "Idris", "spans": [[14, 19]]}],
    "edges": [{"id": "e1", "s": "n1", "r": "works_at", "o": "n3", "spans": [[58, 82]], "conf": 0.93}],
    "extractors": ["A", "B"]
  },
  "labels": {
    "ambiguity": true,
    "resolvability": "unresolved",
    "decision_relevant": true,
    "source_type": "referent",
    "n_interpretations": 2
  },
  "interpretations": [
    {"id": "i1", "binding": {"referent": "n1"}, "paraphrase": "What does Tomás do for a living?",
     "answers": ["works nights at a bakery"], "evidence_spans": [[58, 82]], "kg_path": ["e1"],
     "prior": {"uniform": 0.5, "human": 0.55, "model": 0.50}},
    {"id": "i2", "binding": {"referent": "n2"}, "paraphrase": "What does Idris do for a living?",
     "answers": ["repairs bicycles"], "evidence_spans": [[84, 106]], "kg_path": ["e2"],
     "prior": {"uniform": 0.5, "human": 0.45, "model": 0.50}}
  ],
  "ambiguity_score": {"entropy_uniform": 0.693, "entropy_human": 0.688, "n_eff": 2.0, "p_max": 0.55},
  "clarification": {
    "question": "Do you mean Tomás or Idris?",
    "options": [{"text": "Tomás", "interpretation": "i1"}, {"text": "Idris", "interpretation": "i2"}],
    "eig_bits": 1.0, "eig_normalised": 1.0, "cost": 1
  },
  "resolved": [{"question": "What does Idris do for a living?", "answer": "repairs bicycles"}],
  "validation": {
    "structural": {"b_reproduced": true},
    "nli": {"i1": 0.97, "i2": 0.96},
    "locality": {"unchanged_ratio": 0.98},
    "critics": [{"model": "...", "label": true, "interpretations": 2}],
    "artifact_probe": {"question_only_p": 0.52}
  },
  "human": {"annotations": [{"annotator": "a1", "ambiguous": true, "interp_ids": ["i1", "i2"]}, {"annotator": "a2", "ambiguous": true, "interp_ids": ["i1", "i2"]}], "adjudicated": true},
  "provenance": {"generator": "...", "spec_version": "1.0", "seed": 17},
  "split": "test"
}
```

---

## 15. Annotation strategy

- **Unit:** one item = (passage, history, question). Annotators see passage + history + question and: (1) enumerate interpretations they consider plausible; (2) give an answer per interpretation with highlighted evidence; (3) state whether history resolves it; (4) state whether the answers differ; (5) (subset) rate each interpretation's likelihood.
- **Do not collapse disagreement.** Keep per-annotator labels (Plank 2022; Pavlick & Kwiatkowski 2019 [S]). Adjudicate for a "gold" view but release raw labels.
- **Agreement statistics:**
  - Binary ambiguity label: Krippendorff's α (and Fleiss κ for comparability with prior work).
  - Interpretation sets: Krippendorff's α with MASI distance [M: Passonneau 2006] or Jaccard.
  - Distributions: Jensen–Shannon distance between rater distributions, and split-half reliability.
  - Report α per stratum (surface × multiplicity), because pooled α hides low agreement on hard cells.
- **Pilot:** 300 items, 3 annotators, 2 guideline iterations before scale-up.
- **Rough cost (order of magnitude, my estimate):** 6,000 test items × 3 annotators × ~2 min × $15/h ≈ $9k; distribution subset (500 items × 10 raters × 1.5 min) ≈ $1.9k. LLM generation/critics for ~20k items: low thousands of dollars at current API prices (check).
- **Qualification:** annotator screening test; gold-trap items; no LLM-generated labels shown to annotators before they label (avoids anchoring).

---

## 16. Quality-control pipeline

| Check | What | Pass criterion |
|---|---|---|
| Structural | re-run extraction on edited passage; intended B must be reproduced | exact match of binding set |
| Answerability / entailment | each interpretation's answer entailed by passage (≥2 NLI/LLM verifiers) | both ≥ threshold; flag disagreement for humans |
| Edit locality | unchanged spans preserved; only intended sentences differ | token-diff + embedding similarity |
| Near-duplicate removal | embedding similarity + n-gram overlap across questions and passages | max sim threshold; per-template cap |
| Linguistic diversity | distinct-n, self-BLEU, MTLD, syntactic-template entropy, question-opener entropy | report vs Abg-CoQA/AmbigQA |
| **Artifact audit** | question-only, pronoun-only, length-only, passage-free, history-free classifiers within each stratum | ≤ chance + margin; adversarial filtering ([M] AFLite, Le Bras et al., 2020) for residual cues |
| Ambiguity verification | ≥3 LLM critics from different families + humans | LLM = filter only, never ground truth |
| Leakage | split by story *and* by entity set; hold out relation types; entity-renaming contamination probe; fresh-source split | accuracy on renamed ≈ original |
| **Extraction-recall audit** | exhaustive human enumeration of referents/events on ≈200 stories vs G | report recall; estimate fraction of "unambiguous" labels that become ambiguous under the human set |
| Difficulty | N_eff, |B|, model disagreement entropy, human accuracy | report as post-hoc stats |

---

## 17. Benchmark tasks

| Task | Include? | Rationale |
|---|---|---|
| 1 Ambiguity detection (3-way: unambiguous / resolved-by-history / ambiguous; + harmless) | **Core** | contrast with binary; distinguishes the hypotheses |
| 2 Source-type classification | Secondary | taxonomy validity |
| 3 Interpretation generation | **Core** | what Abg-CoQA lacks |
| 4 Candidate answer generation / answer-set QA | **Core** | AmbigQA-style F1 |
| 5 Ambiguous QA (answer or ask) | Core | end-to-end |
| 6 Clarification question generation | **Core** | |
| 7 Clarification ranking | Secondary | cheap, gives EIG-based gold |
| 8 Answer after clarification | Core | closes the loop |
| 9–10 KG path / evidence retrieval | Secondary | evidence F1 |
| 11 Severity estimation | Secondary | needs human subset |
| 12 Selective answering / abstention | **Core** | risk–coverage |
| 13 LLM calibration | **Core** | on the ambiguity decision |
| 14 Multi-turn ambiguity resolution | Limited | small subset to avoid P2 scope creep |
| **15 Counterfactual sensitivity** | **Core, distinctive** | the new task |

Strongest benchmark core: tasks 1, 3, 4, 6, 12, 13, 15.

---

## 18. Baselines

| Generation | Systems |
|---|---|
| Heuristics | majority; pronoun rule; |B| oracle from the extracted graph; length rule; random-policy ask/answer |
| Artifact probes (acceptance tests, not competitors) | question-only TF-IDF LR; passage-free encoder; history-free |
| Encoders | RoBERTa/DeBERTa-v3 cross-encoder over (question, history, passage) |
| Generative | T5/Flan-T5 fine-tuned for enumeration |
| Graph | R-GCN/GAT or graph transformer over the extracted KG + question embedding for detection |
| RAG | evidence retrieval + LLM; evidence-first prompting |
| Open LLMs | current open-weight instruction and reasoning models (pick the current Llama/Qwen/DeepSeek-class releases at submission time) |
| Proprietary | current frontier models from at least three vendors; include reasoning vs non-reasoning variants (AbstentionBench reports reasoning fine-tuning degrades abstention) |
| Interventions | prompting variants (enumerate-then-answer, ask-or-answer, CoT); APA-style alignment; ACT/DPO on the training split; the interpretation-RL recipe of Saparina & Lapata |

Behaviours scored for each model: (1) detects ambiguity; (2) blindly guesses; (3) enumerates interpretations; (4) asks; (5) answers correctly after clarification.

---

## 19. Metrics

- **Detection:** macro-F1, AUROC, AUPRC; Brier score and ECE on the ambiguity probability; **per-stratum** metrics (surface × multiplicity × history).
- **Selective answering:** risk–coverage curves, AURC, accuracy at fixed coverage.
- **Interpretation enumeration:** interpretation recall/precision/F1 via ID-based matching (we have gold bindings) plus an NLI matcher for free-text; answer-set F1 (AmbigQA F1_ans); EDIT-F1 for rewrites; evidence F1 (span- and edge-level).
- **Clarification:** (a) EIG-normalised utility of the asked question under the gold interpretation set; (b) option-coverage (does the question's partition separate every interpretation?); (c) turns-to-resolution and **regret** vs the EIG-optimal policy (RegretBench-style); (d) human ratings of naturalness and non-leading. Simulated user responses are deterministic from gold bindings, not an LLM.
- **Post-clarification QA:** EM / token-F1 against the resolved answer.
- **Counterfactual sensitivity (new, deliberately simple):**
  - *Structure sensitivity:* P(flag | E-add, \|B\|: 1→2) − P(flag | S-pron on \|B\|=1).
  - *Surface dependence index:* flag-rate difference between S-pron / S-name variants with identical B.
  - *History sensitivity:* flag-rate drop under H-resolve; misleading-history error under H-mislead.
  - *Over-clarification:* ask-rate on harmless items.
  - *Monotonicity violation rate* (M1–M4).
- **Consistency:** answer stability under Q-para.

---

## 20. Experiments

1. **Artifact audit across benchmarks** (KG-CoQA, plus any of Abg-CoQA/CAmbigNQ/CLAMBER you can obtain): question-only classifier accuracy. Expected: KG-CoQA ≈ 98% (already measured); CF-Ambig ≈ chance within strata.
2. **Main evaluation** of §18 systems on core tasks.
3. **Counterfactual sensitivity** across factor cells (the hypothesis test).
4. **History experiments:** H-resolve vs H-erase vs H-mislead.
5. **Harmless ambiguity:** over-clarification rates; does CoT/reasoning make it worse?
6. **Calibration vs human priors** on the distribution subset.
7. **Ecological validity:** train on CF-Ambig synthetic, test on Abg-CoQA/CAmbigNQ natural; and the reverse. If transfer fails, the benchmark measures a synthetic skill.
8. **Human baseline** on a test subset.
9. **Contamination probes:** original vs entity-renamed vs fresh-source.
10. **Training value:** counterfactual-paired fine-tuning vs data-matched unpaired control.

## 21. Ablation studies

- Single vs dual KG extractor (effect on recall of interpretations and on false "unambiguous" labels).
- KG-assisted mining vs LLM-only candidate mining (does the KG add anything?).
- With vs without LLM critics; with vs without human verification (label-noise sensitivity).
- Template realisation vs LLM realisation (naturalness vs artifact rate).
- Edit-locality filter on/off.
- Uniform vs human-prior ambiguity score (correlation with model/human behaviour).
- Fresh-source vs CoQA split.
- Remove each factor (history, answer-divergence) to see which drives model behaviour.

## 22. Human evaluation

- **Validity of interpretation sets:** do independent annotators accept each interpretation? Do they propose interpretations the KG missed? (recall estimate, with confidence intervals).
- **Naturalness:** blind comparison of CF-Ambig questions/history turns against real CoQA turns and Abg-CoQA questions; report the rate at which raters can tell which is generated.
- **Clarification quality:** rating of fluency, unambiguity, non-leadingness; match to EIG-optimal partition.
- **Human behaviour on counterfactuals:** do humans' ambiguity ratings move the way the edit operators predict (M1–M4)? This validates the operators.
- **Agreement:** per §15.

---

## 23. Example records (12)

Passages are short and written for illustration; real records use CoQA text and fresh sources. KG fragments use (subject, relation, object).

**E1. Coreference / referent · local · unresolved.**
Passage: "Tomás and Idris shared a flat in Porto. Tomás worked nights at a bakery; Idris repaired bicycles."
KG: (Tomás, works_at, bakery) · (Idris, repairs, bicycles)
Question: "What does he do for a living?"
Why ambiguous: two male antecedents, no history, different answers.
A: Tomás → works nights at a bakery. B: Idris → repairs bicycles.
Clarification: "Do you mean Tomás or Idris?" Resolved: "What does Idris do for a living?"
Type: referent · R3 · decision-relevant.

**E2. Entity name collision (full name, no pronoun) · unresolved.**
Passage: "Sam Reyes joined the school as librarian in 2008. Sam Whitaker, who teaches chemistry, joined in 2015."
KG: (Sam Reyes, joined_in, 2008) · (Sam Whitaker, joined_in, 2015)
Question: "Which year did Sam join the school?"
A: Sam Reyes → 2008. B: Sam Whitaker → 2015.
Clarification: "Sam Reyes, the librarian, or Sam Whitaker, the chemistry teacher?"
Type: entity · R3 · decision-relevant. **The cell KG-CoQA never populates.**

**E3. Name collision resolved by type constraints (hard negative).**
Same passage. Question: "What does Sam teach?"
Why *not* ambiguous: "teach" selects Whitaker (the librarian does not teach).
|B_0| = 2 lexically, |B| = 1 after constraints.
Answer: chemistry. Type: entity · R1 · unambiguous.

**E4. Relation ambiguity.**
Passage: "Hana volunteers at the city museum under the curator Ilya. Her cousin Ruth runs the gift shop there."
KG: (Hana, works_under, Ilya) · (Hana, cousin_of, Ruth) · (Ruth, runs, gift shop)
Question: "Who is Hana connected to at the museum?"
A (supervisory): Ilya. B (kin): Ruth.
Clarification: "Do you mean who she works under, or a relative who works there?"
Type: relation · R3 · decision-relevant.

**E5. Multi-hop: ambiguity at hop 1 propagates.**
Passage: "Lena wrote the safety report and Omar wrote the budget report. Lena reports to Fiona; Omar reports to Greg."
KG: (Lena, wrote, safety report) · (Omar, wrote, budget report) · (Lena, reports_to, Fiona) · (Omar, reports_to, Greg)
Question: "Who does the report's author report to?"
A: safety report → Lena → Fiona. B: budget report → Omar → Greg.
Clarification: "Which report: the safety report or the budget report?"
Type: path (hop 1) · R3 · decision-relevant.

**E6. Event / temporal.**
Passage: "The Harbor Theatre broke ground in March 1987 and opened in October 1989. It closed for renovations in 2004 and reopened in 2007."
Question: "When did the Harbor Theatre open?"
A: first opening → October 1989. B: reopening → 2007.
Clarification: "The original opening, or the reopening after the renovation?"
Type: event/time · R3 · decision-relevant.

**E7. Granularity (answer-compatible, typically harmless).**
Passage: "Aarav was born in Whitefield, a suburb of Bengaluru, in the Indian state of Karnataka."
KG chain: (Aarav, born_in, Whitefield) · (Whitefield, located_in, Bengaluru) · (Bengaluru, located_in, Karnataka)
Question: "Where was Aarav born?"
Interpretations: neighbourhood / city / state, which are nested. Any more specific answer entails the less specific.
Expected behaviour: answer with the most specific, no clarification. Type: granularity · R1 · harmless (nested).

**E8. Harmless ambiguity (answer-invariant, pronoun + 2 candidates).**
Passage: "Ben and his brother Carl both grew up in Leeds. Later Ben moved to Oslo."
Question: "Where did he grow up?"
|B| = 2 (Ben, Carl); ans(B) = {Leeds}.
Expected behaviour: answer "Leeds". Asking is **over-clarification**.
Type: referent · R1 (by invariance) · harmless. **KG-CoQA by construction has none of these.**

**E9a/E9b. History resolves vs does not (counterfactual pair).**
Passage: "Kofi and Ade run a sound studio. Kofi studied economics in Accra; Ade studied engineering in Kumasi."
E9a history: U: "What does Ade do?" / A: "Ade runs the studio with Kofi." Question: "And where did he study?"
→ |B_0| = 2, |B_H| = 1 (Ade, recency/focus). Answer: Kumasi. Status: resolved-by-history (defeasible).
E9b history: empty. Question: "Where did he study?"
→ |B| = 2, answers differ → ambiguous. Clarification: "Kofi or Ade?"
Type: referent · R2 vs R3. **This pair is the minimal-pair unit.** Edit: H-erase.

**E10. Missing argument / ellipsis.**
Passage: "The Aster line runs every 10 minutes on weekdays and every 20 on weekends. The last train leaves at 23:30 on weekdays and at 01:00 on Fridays and Saturdays."
History: U: "When does the Aster line run?" / A: "Every 10 minutes on weekdays, every 20 on weekends."
Question: "And the last one?"
A: weekdays → 23:30. B: Fridays and Saturdays → 01:00.
Clarification: "On a weekday, or on a Friday/Saturday night?"
Type: missing argument · R3 · decision-relevant.

**E11. Numerical / scope.**
Passage: "Of the 120 people registered for the workshop, 85 came on the first day and 97 on the second."
Question: "How many attended?"
A: first day → 85. B: second day → 97. (C: total unique unspecified in the passage, so the question is not fully answerable.)
Clarification: "On which day?"
Type: numerical/scope · R3 · decision-relevant.

**E12. Evidence conflict (boundary case, excluded from the ambiguity core).**
Passage: "The town's website says the museum opens at 9 a.m. The brochure printed last year says 10 a.m."
Question: "What time does the museum open?"
Interpretation set: one (|B| = 1). Answer set: two conflicting sources.
Expected behaviour: report both with sources. This is evidence conflict, not question ambiguity: a distractor class that keeps the two axes apart.

---

## 24. Dataset statistics to report

- Counts: stories, seed turns, families, items, per split; fresh-source share.
- Per factor cell: item counts; \|B\| distribution; answer-divergence distribution; harmless share.
- Source-type and resolution-condition distributions (§5 axes).
- Natural-seed prevalence: % of mined CoQA turns with |B_0| > 1 (informative on its own).
- KG statistics: nodes/edges per story, relation-type counts, extractor agreement, **recall/precision on the audit set**.
- Linguistic diversity (§16), question length, opener entropy, template reuse.
- Agreement (§15) per stratum; adjudication rate; LLM-critic vs human agreement.
- Artifact probe results per stratum; contamination probe gaps.
- Ambiguity score distributions under uniform vs human priors; correlation.
- Clarification: options per question, EIG distribution.

---

## 25. Research questions and hypotheses

- **RQ1:** Does LLM ambiguity detection track evidence structure or surface form? **H1:** flag rates differ more across S-pron/S-name (same B) than across E-add/E-del (different B).
- **RQ2:** Do models use dialogue history to resolve ambiguity? **H2:** flag rate does not drop under H-resolve, and models over-trust a recent wrong-candidate mention (H-mislead).
- **RQ3:** Do models distinguish decision-relevant from harmless ambiguity? **H3:** over-clarification is substantial on harmless items, and worse for reasoning-tuned models.
- **RQ4:** Does interpretation enumeration degrade with |B| and N_eff? **H4:** recall falls monotonically with |B|.
- **RQ5:** Is a pronoun-driven benchmark solvable without evidence? **H5:** question-only classifiers reach ~98% on KG-CoQA but ≈ chance within CF-Ambig strata. (First half already measured.)
- **RQ6:** Does counterfactual-paired training improve structure sensitivity beyond data volume? **H6:** yes vs an unpaired, size-matched control.
- **RQ7:** Does synthetic training transfer to natural ambiguity (Abg-CoQA, CAmbigNQ)? Falsifiable either way.

---

## 26. Paper contributions (each tied to its closest prior work)

1. **An audit showing a graph-generated ambiguity label is recoverable from surface form** (pronoun rule ≈98%; question-only LR 97.6%) and a stratified artifact-probe protocol. *Closest:* Gururangan et al. 2018 / Poliak et al. 2018 (hypothesis-only baselines in NLI); no ambiguity-QA audit found.
2. **A taxonomy with three orthogonal axes** (source, resolution condition, consequence) replacing flat type lists, with *harmless* ambiguity formalised. *Closest:* CLAMBER, Abg-CoQA, ClarifyMT-Bench taxonomies, AmbiGraph-Eval.
3. **Typed counterfactual edit operators over passage, history and surface form with known effect on the interpretation set**, plus edit-consistency axioms used for validation and for behavioural tests. *Closest:* AmbiCoref (sentence-level minimal pairs), Ambig-DS (controlled edits for agent tasks), AMBROSIA/CLARITY (controlled generation from formal structure).
4. **Interpretation-level annotations with evidence spans and KG paths,** and a recall audit of KG-derived interpretation sets against exhaustive human enumeration. *Closest:* AmbigQA, CondAmbigQA; DeepAmbigQA/MARCH (no recall audit known).
5. **Gold clarification utility from the interpretation partition** (EIG-based) with deterministic simulated responses validated on human responses. *Closest:* EIG for text-to-SQL, CLARINET, RegretBench.
6. **A multi-task evaluation with counterfactual-sensitivity metrics,** with findings about modern LLMs. *Closest:* CLAMBER, ClarifyMT-Bench, AbstentionBench (evaluation of modern LLMs alone is **not** a contribution).

Not claimed: "first KG-based ambiguity generation", "first multi-hop ambiguity", "first LLM ambiguity benchmark", "first clarification dataset", "first minimal pairs for ambiguity".

---

## 27. Reviewer criticisms and defences

| # | Reviewer 2 | Defence / experiment |
|---|---|---|
| 1 | Isn't this synthetic augmentation? | Natural CoQA seeds; edits are single-sentence; Exp 7 (natural transfer); naturalness human study |
| 2 | Why CoQA? | Natural coreference/ellipsis, conversational structure; plus fresh-source split and contamination probes; claims don't depend on CoQA specifically |
| 3 | Why is KG extraction reliable? | It isn't assumed: dual extractors, evidence spans, exhaustive-human recall audit, ablation on extraction quality |
| 4 | Multi-parent topology ≠ linguistic ambiguity | Agreed (§7). We use topology for *enumeration*; probability and resolvability come from human priors and history. Report disagreement between topology-based and human labels |
| 5 | Are questions natural? | Seeds are real; rater discrimination test vs real CoQA turns; template-vs-LLM ablation |
| 6 | Different from Abg-CoQA? | Abg-CoQA: 994 crowd-flagged positives, no interpretation set/graph/counterfactuals. Here: formal sets, evidence, counterfactual families, harmless class, 3-axis taxonomy |
| 7 | Different from AmbigQA? | AmbigQA: open-domain single-turn, per-question sets. Here: conversational, passage-grounded, matched families, graph evidence |
| 8 | Why is an LLM needed? | Surface realisation only; ablation shows template-only realisation has higher artifact rate |
| 9 | Could this be built without a KG? | LLM-only mining gives no exhaustiveness guarantee or controlled edits; ablation: KG-assisted vs LLM-only mining recall and edit validity |
| 10 | Is ambiguity natural or artificial here? | Both are measured: natural seed prevalence reported; synthetic edits labelled as such; transfer experiment |
| 11 | Are candidate interpretations exhaustive? | Recall audit vs human enumeration; limitation reported with CIs; items where humans find extra interpretations are added to the gold, not dropped |
| 12 | Large enough? | Targets ≈3k seeds / ≈12–15k items; power analysis on counterfactual contrasts; test fully human-verified |
| 13 | Human validation cost? | Quantified (§15); verification concentrated on test; train audited by sample with LLM-critic filtering |
| 14 | Can models exploit generation artifacts? | Stratified artifact probes as acceptance criteria; adversarial filtering; paraphrase invariance |
| 15 | Train/test leakage? | Split by story and entity set; renaming probe; fresh-source split |
| 16 | What hypothesis does the benchmark test? | H1–H7 (§25); each falsifiable; negative results informative |
| 17 | The clarification simulator is an LLM | It is not: responses are deterministic from gold bindings, validated on human responses |
| 18 | Edits make the passage odd | locality check; naturalness study; report % rejected |

---

## 28. Recommended paper title

- "**Same Words, Different Evidence: A Counterfactual Benchmark for Interpretation-Aware Conversational Question Answering**"
- Alternatives: "Does the Model See the Ambiguity, or the Pronoun? Counterfactual Evaluation of Clarification in Conversational QA"; "CF-Ambig: Graph-Grounded Counterfactuals for Ambiguity and Clarification in Conversational QA".

## 29. Suitable venues

ACL / EMNLP / NAACL (resources & evaluation track, via ARR); COLING / LREC-COLING; NeurIPS Datasets & Benchmarks; COLM; TACL for the audit-heavy version. Abg-CoQA appeared at AKBC 2021; check whether AKBC is still held before targeting it. A short version of the KG-CoQA artifact audit could also go to a workshop (e.g., a dataset-quality or evaluation workshop) for early feedback.

## 30. Implementation roadmap

| Step | Work | Deliverable | Rough time (1–2 people) |
|---|---|---|---|
| 0 | Read the six threat papers fully; run citation tracing; decide scope | notes + related-work table | 1–2 wk |
| 1 | Re-run the KG-CoQA audit on any other benchmarks you can obtain; write the audit as a standalone section/preprint | audit results | 1 wk |
| 2 | KG extraction pipeline (2 extractors, evidence spans, relation normalisation); build 200-story exhaustive human audit | recall/precision numbers | 3–4 wk |
| 3 | Seed mining on CoQA; compute B, ans(B), status; human verification of 300 seeds | seed set + prevalence | 2–3 wk |
| 4 | Edit operators + spec→LLM realisation + validation stack | generation code + rejection stats | 3–4 wk |
| 5 | Pilot annotation (300 items), guideline v2, agreement per stratum | guidelines + α | 2–3 wk |
| 6 | Scale generation; human-verify test; audit train; distribution subset | v1 dataset | 4–6 wk |
| 7 | Baselines and counterfactual-sensitivity experiments; ecological-transfer; contamination probes | results | 3–4 wk |
| 8 | Writing, ablations, human evaluation, release (licences per CoQA domain) | paper + artifacts | 4 wk |

Gate after step 2: if extraction recall on the audit is poor enough that "unambiguous" labels are unreliable, restrict the benchmark to precision-oriented ambiguous items plus human-verified negatives, or re-scope.

---

## 31. Final research-gap statement

**Existing work can do:**
- Annotate natural ambiguity with human interpretation sets and clarification questions (AmbigQA, Abg-CoQA, CAmbigNQ, CondAmbigQA).
- Generate ambiguity from structured sources at scale (DeepAmbigQA, MARCH, AMBROSIA, SQUAB, CLARITY).
- Benchmark LLM overconfidence and clarification policy (CLAMBER, ClarifyMT-Bench, RegretBench, AbstentionBench).
- Provide sentence-level minimal pairs for coreference ambiguity (AmbiCoref).

**Existing work cannot adequately do:**
- Isolate whether a model's ambiguity behaviour tracks the interpretation structure of the evidence or surface cues, because labels are confounded with form or no counterfactual control exists.
- Supply matched families of conversational, passage-grounded questions where history, evidence multiplicity, surface form and answer divergence vary independently with known interpretation-set effects.
- Include answer-invariant (harmless) ambiguity as a hard negative.

**The unresolved research problem is:**
Measuring, causally, whether LLMs respond to the *structure of ambiguity in evidence and dialogue* rather than to its linguistic signals, and whether they ask when, and only when, clarification is decision-relevant.

**We propose:**
CF-Ambig, a counterfactual, graph-grounded benchmark for conversational QA with natural CoQA seeds, KG-derived interpretation sets, typed edit operators, a three-axis taxonomy (source × resolution × consequence), interpretation-level annotations with evidence, and gold clarification utility.

**Why a new dataset is necessary:**
The control must be built into the data: the same question has to appear under edits whose effect on the interpretation set is known. This cannot be retrofitted onto existing labels.

**Why existing datasets cannot simply be reused:**
Abg-CoQA and AmbigQA offer natural but uncontrolled items with small positive sets and no counterfactuals. DeepAmbigQA/MARCH lack conversation and matched families. KG-CoQA's label is recoverable from the pronoun (≈98%). AmbiCoref/Ambig-DS are in other settings (sentence-level; agent tasks).

**Key novelty (honest scope):**
Counterfactual factorial control over evidence, history, surface and answer-divergence in passage-grounded conversational QA, with a recall-audited KG as oracle, and the surface-confound audit that motivates it. Not claimed: KG-based generation, multi-hop ambiguity, clarification annotation, or modern-LLM evaluation.

**Strongest scientific hypothesis:**
H1: existing LLMs' ambiguity decisions depend more on surface form than on evidence multiplicity, history resolution, or answer divergence. Models over-clarify on harmless ambiguity and under-use resolving history.

**Minimum experiments required:**
(1) the cross-benchmark artifact audit with stratified probes; (2) the extraction-recall audit against exhaustive human enumeration; (3) the counterfactual-sensitivity evaluation of ≥6 models across ≥3 families, with human baseline; (4) the natural-transfer experiment on Abg-CoQA or CAmbigNQ; (5) the contamination probe (renamed entities + fresh-source split); (6) ablations on KG vs LLM-only mining and on paired vs unpaired training.

---

## Appendix A. Citation list (all surfaced in search this session unless tagged [M])

Peer-reviewed / archival:
- Guo et al. 2021, Abg-CoQA, AKBC. doi 10.24432/C5F30Z
- Min et al. 2020, AmbigQA, EMNLP. 2020.emnlp-main.466; arXiv 2004.10645
- Stelmakh et al. 2022, ASQA, EMNLP. 2022.emnlp-main.566
- Xu et al. 2019, CLAQUA, EMNLP-IJCNLP. D19-1172
- Lee et al. 2023, CAmbigNQ, Findings EMNLP. 2023.findings-emnlp.772
- Zhang et al. 2024, CLAMBER, ACL. 2024.acl-long.578; arXiv 2405.12063
- Liu et al. 2023, AmbiEnt, EMNLP. 2023.emnlp-main.51; arXiv 2304.14399
- Yuan, Malaviya, Yatskar 2023, AmbiCoref, Findings EACL. 2023.findings-eacl.75; arXiv 2302.00762
- Chen et al. 2021, AmbER, ACL. 2021.acl-long.345; arXiv 2106.06830
- Lee, Ye, Choi 2024, AmbigDocs, COLM. arXiv 2404.12447
- CondAmbigQA, EMNLP 2025. arXiv 2502.01523; 2025.emnlp-main.115
- Papicchio, Cagliero, Papotti 2025, SQUAB, EMNLP. 2025.emnlp-main.906
- Saparina & Lapata 2024, AMBROSIA, NeurIPS D&B. arXiv 2406.19073
- Sarwar et al. 2026, CLARITY, ACL Industry. arXiv 2604.22313; 2026.acl-industry.86
- Park et al. 2026, MARCH, Findings of ACL. arXiv 2509.22750; 2026.findings-acl.1352
- Kirichenko et al. 2025, AbstentionBench, NeurIPS D&B. arXiv 2506.09038
- Tanjim et al. 2025, Disambiguation in Conversational QA survey, EMNLP. 2025.emnlp-main.482; arXiv 2505.12543
- Reddy, Chen, Manning 2019, CoQA, TACL. Q19-1016; arXiv 1808.07042
- Choi et al. 2018, QuAC, EMNLP. D18-1241
- Anantha et al. 2021, QReCC, NAACL. arXiv 2010.04898; 2021.naacl-main.44
- TopiOCQA. arXiv 2110.00768
- Saha et al. 2018, CSQA, AAAI. arXiv 1801.10314
- Christmann et al. 2019, ConvQuestions/CONVEX, CIKM. arXiv 1910.03262
- Cao et al. 2022, KQA Pro, ACL. 2022.acl-long.422
- Brabant et al. 2024, KGConv, LREC-COLING. arXiv 2308.15298; 2024.lrec-main.850
- Liu et al. 2023, PReFNet, WWW. par.nsf.gov/biblio/10428936
- Cole et al. 2023, Selectively Answering Ambiguous Questions, EMNLP. arXiv 2305.14613
- Kuhn, Gal, Farquhar 2023, Semantic Uncertainty, ICLR. arXiv 2302.09664
- Farquhar et al. 2024, semantic entropy, Nature. PMC11186750
- Hou et al. 2024, Input Clarification Ensembling, ICML (PMLR 235:19023). arXiv 2311.08718
- Kim et al. 2024, APA, EMNLP. arXiv 2404.11972; 2024.emnlp-main.119
- Zhang & Choi, Clarify When Necessary, Findings NAACL 2025. arXiv 2311.09469
- Zhang, Knox, Choi, Modeling Future Conversation Turns, ICLR 2025. arXiv 2410.13788
- Chen et al., ACT, ICLR 2025. arXiv 2406.00222
- Gururangan et al. 2018, Annotation Artifacts in NLI, NAACL. N18-2017
- Poliak et al. 2018, Hypothesis-Only Baselines, *SEM. S18-2023
- Plank 2022, "The Problem of Human Label Variation", EMNLP. 2022.emnlp-main.731 (title from search summary)
- Pavlick & Kwiatkowski 2019, Inherent Disagreements in Human Textual Inferences, TACL [S via summary]
- Tang, Soulier, Guigue 2025, AT-CoT, SIGIR. arXiv 2504.12113

Preprints (as of this search):
- DeepAmbigQA, arXiv 2511.01323
- RAMDocs / MADAM-RAG, arXiv 2504.13079
- AmbiGraph-Eval, arXiv 2508.09631
- CLEAR-KGQA, arXiv 2504.09665
- ClarifyMT-Bench, arXiv 2512.21120
- RegretBench ("One More Turn, Less Regret"), arXiv 2607.21143
- DiscoBench, arXiv 2606.27669
- Ambig-DS, arXiv 2605.09698
- CLAIM, arXiv 2608.11631
- The Illusion of Certainty, arXiv 2511.04418
- From Answers to Interpretations, arXiv 2609.04543 (reported accepted to Findings of EMNLP 2026; verify)
- Interactive Text-to-SQL via EIG, arXiv 2507.06467
- CLARINET, arXiv 2405.15784
- Correct-Detect, arXiv 2509.14456
- Saparina & Lapata, Reasoning about Intent for Ambiguous Requests, arXiv 2511.10453
- Qulac / ClariQ (ClariQ: arXiv 2009.11352)

Memory-only, not re-verified [M]: Rao 1982 quadratic entropy; Passonneau 2006 MASI; Le Bras et al. 2020 AFLite; CoQA per-domain licences.

Seen in results but **not examined**, do not cite without reading: ECLAIR (arXiv 2503.15739), AmbiSQL (2508.15276), ConfuseBench, SCICONVBENCH (2605.18630), the agricultural KGQA-CQG paper (journal listing only).
