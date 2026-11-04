# Answer grading rubric

Use the benchmark's original `gold_answer` and `answer_facts` for each question. Grade each fact independently against the candidate answer, not against the retrieved context. Context helps diagnose a failure afterwards, but the answer must itself state or clearly imply a required fact. Preserve the original facts and flag questionable references for review rather than silently changing the gold set.

## Fact labels

- **Satisfied (1):** the answer states or clearly implies the fact, including its necessary conditions, numbers, and relationships. Equivalent wording counts.
- **Not satisfied (0):** the answer omits the fact, gives insufficient detail, or contradicts it. Do not infer a missing fact from general plausibility.
- **Needs review:** the fact or reference is ambiguous or appears unsupported. Record the reason; resolve it before calculating the final score. Do not silently drop the fact or modify the reference for one system only.

Some benchmark facts are constraints, such as “must not claim X”. Treat a prohibition as satisfied when the answer does not make the forbidden claim, and not satisfied when it does. This does not substitute for separately required positive facts. Where a fact says “may mention”, do not penalize omission; if present, check that the stated detail is accurate. Record any ambiguity for joint review. Extra incorrect claims should be recorded separately, including contradictions not covered by a fact; the proposal's fact-fraction score alone may not capture them.

For a successfully generated answer with resolved labels, **correctness = satisfied facts / total gold facts**. **Fully correct** means all gold facts are satisfied. An abstention that omits required facts earns zero for those facts; a failed generation is a failure, not a successful answer with an invented score. Report failed attempts separately and define the treatment of failures in any aggregate before comparison. Questions without usable facts or unresolved labels are N/A pending review.

## Independent review and calibration

Blind reviewers to the retriever identity and independently label the same sample: the plan specifies 40 answer attempts, ten per project type, by two human graders. Save question and anonymous answer IDs, the fact labels, reviewer, and notes. Resolve disagreements after independent grading, preserving both original labels. Compare the local judge with human grading and report human–human and judge–human Cohen's kappa, stating the unit of comparison and handling of N/A labels. The proposal calls for manual grading of types with kappa below 0.6; apply that rule after agreeing how the fact labels are aggregated for the per-type comparison. Freeze the judge and rubric before comparison runs.
