# TODO

## 1. Finish designing the principle generation mechanism
- [ ] Define the knowledge-tree levels: goal → principles → policies/criteria → decisions → details/evidence
- [ ] Decide how nodes are built: bottom-up (decisions from chunks) + top-down (principles from theme descriptions/review feedback) + linking
- [ ] Define node status: stated in docs / inferred from several decisions / missing
- [ ] Define gap types by tree position: missing principle, missing reason, missing boundary, orphan principle, contradiction
- [ ] Decide the principle format (situation type, principle, why, exceptions/signals, examples) and how it relates to `distill()`
- [ ] Decide how the expert reviews/corrects the tree (viewer page, ~30 min)
- Refs: `Knowledge_Capture_PoC_Report.docx` §8.1, `kcap-public/docs/gap_finder.md` §9

## 2. Add explicit and implicit feedback extraction for the tree search
- [ ] Explicit: optional one-click buttons per answer in the GUI: useful / already documented / unclear / repeated (+ existing "I don't know")
- [ ] Implicit: "I don't know"/hedging detection, realized principle gain (incremental distill), redundancy vs earlier answers, new specifics mentioned
- [ ] Combine into a turn reward r_t (beware comfort bias: "already documented" counts negative)
- [ ] Use r_t to update per-action and per-topic priors, answerability p_ans, and simulator calibration before the next search
- [ ] Offline replay on existing transcripts + tags to check the learned priors would have picked better questions
- [ ] Implement reward v2 behind `strategy_d.reward` config switch (keep v1 for comparability)
- Refs: `kcap-public/docs/tree_search.md` §6–7, `ab_mcts_interview_reward_design.md`

## 3. Improve the gap finder clustering to implement principle generation
- [ ] Replace flat entity clustering with building the knowledge tree (depends on 1)
- [ ] Gap finding per tree node instead of per document cluster; gap cards carry their gap type and tree position
- [ ] Opening questions anchored to concrete past decisions (no internal codes, enough context)
- [ ] Generate hidden test scenarios from tree nodes the documents do not resolve (fairer test)
- [ ] Compare flat vs tree-based gap finding (Experiment 2 precision/recall; offline on Exp 3 corpus)
- Refs: `kcap-public/kcap/components/gap_common.py`, `gap_miner.py`
