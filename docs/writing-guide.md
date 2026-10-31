# Jute study writing guide

## Use case

A focused empirical ML paper about jute disease image classification for PCSC reviewers. A skeptical reader should identify what was compared, what the evidence supports and where uncertainty remains. Keep the existing seven sections: Introduction; Related work; Data and related-image controls; Experimental design; Results; Discussion and limitations; Conclusion. PCSC 2027 is tentative, with no fixed deadline.

## Precedence

`requirement` > `authoritative guideline` > `empirical` > `convention` > `preference`. A higher grade wins a conflict within its actual scope. This guide owns writing and reporting; the study's protocol and canonical issues own scientific decisions. For manuscript editing/builds, read `paper/README.md` when present for build and source-preservation procedures. That private file may be absent from a fresh checkout.

## Rules

### Requirement

No PCSC 2027 requirement is verified. Before submission, verify the applicable call, track, template, anonymity, page-count treatment, mandatory sections, disclosures and deadlines. Historical 2026 instructions establish no future requirement.

### Authoritative guideline

1. **Bound claims by the reported comparison (authoritative guideline).** Keep abstract, introduction and conclusion consistent with the tested data, evaluation conditions, observed results and limitations. Report stable, null, negative and inconclusive findings according to the evidence. NeurIPS guidance is transferred for reporting; its submission obligations apply to NeurIPS.
   Source: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), Claims. "Claims in the paper should match theoretical and experimental results."

2. **Name what uncertainty captures (authoritative guideline).** State the statistic, calculation, assumptions and number of underlying observations or runs. Distinguish variation over seeds/runs, splits and sampled evaluation cases; identify what remained fixed or unassessed. Report the analysis actually performed; this guide sets no repetition count or interval method.
   Source: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), Experiment Statistical Significance. "The factors of variability that the error bars are capturing should be clearly stated."

3. **Separate statistical significance from magnitude (authoritative guideline).** If tests are reported, give the comparison magnitude and its relevant uncertainty. Interpret a non-significant result according to the analysis; it alone establishes neither equality nor practical equivalence. A threshold alone cannot establish practical importance.
   Source: [ASA statement](https://www.amstat.org/docs/default-source/amstat-documents/p-valuestatement.pdf), principle 5. "A p-value, or statistical significance, does not measure the size of an effect."

### Empirical

4. **Treat saliency appearance as limited evidence (empirical).** Visually plausible highlighting alone does not validate symptom-based reasoning, localization or generalization. This empirical finding motivates the optional diagnostic-reporting pattern below; project protocol choices remain separate.
   Source: [Adebayo et al.](https://papers.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf), abstract. "reliance, solely, on visual assessment can be misleading."
   Scope: ImageNet, MNIST and Fashion-MNIST experiments randomized parameters and labels. [Yona and Greenfeld](https://arxiv.org/pdf/2110.14297) demonstrate task confounding of parameter-randomization interpretations using semi-synthetic partial/multi-object tasks. Neither paper validates this project's diagnostics; the follow-up does not address every original check.

5. **Uncertainty encoding can change judgments (empirical).** The experimental finding concerns encoding-dependent judgments in the tasks described below. The project application is a preference in the display-choice rule.
   Source: [Correll and Gleicher](https://vis.csail.mit.edu/classes/6.859/readings/pdfs/Corell-ErrorBarsConsideredHarmful.pdf), section 4.2.1. "more likely than those above the bar."
   Scope: 240 North American online participants across experiments; a 96-participant experiment found asymmetric likelihood judgments for outcomes equally distant from a bar-chart mean. Symmetric encodings tested did not show that significant asymmetry. Text-only uncertainty also had costs. This does not establish a universally best format.

6. **Display performance depends on the reading task (empirical).** The experimental finding concerns task-dependent speed and accuracy, not a universal table/graph ranking. The project application is a preference in the display-choice rule.
   Source: [Hink, Wogalter and Eustace](https://www.safetyhumanfactors.org/wp-content/uploads/2020/07/96%29HinkWog%2CEustace%281996%29.pdf), p. 1159. "response time also depended on question and display type."
   Scope: 63 undergraduates answered numerical, trend and comparison questions using seven printed display types. Numerical answers for graphs had a tolerance, and the study observed speed/accuracy tradeoffs. Transfer to manuscript reading is an inference.

### Convention

7. **Describe data provenance and related-image knowledge (convention).** Identify dataset/version, origin, admission and partition policies, class/partition counts and relevant preprocessing. State known relationships, grouping-control coverage, unresolved relationships and historical test exposure from the owning evidence. Bounded controls establish no plant/session independence or zero leakage.
   Source: [REFORMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11092361/), items 2a, 5c and 6b. "dependencies or duplicates between the training and test datasets."
   Scope: published consensus among 19 researchers in ML-based science; no professional-body endorsement was identified. This guide conservatively grades its transfers as convention. Patient-related observations motivate disclosure; jute image relationships must be established by project evidence.

8. **Make evaluation and selection policy inspectable (convention).** State what information each policy can use, partitions/metrics, configuration and reported-model selection, tuning effort, and whether test evidence influenced selection. Describe material differences and their implications for comparability. Comparable reporting does not mean identical hyperparameters or an invented equal-budget rule.
   Source: [REFORMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11092361/), items 5c to 5f, evaluation, selection, tuning and baselines. The checklist supports disclosure; scientific decisions stay in their owning protocol/issues.

9. **State the reach of the evaluated pipeline (convention).** Distinguish classification of prepared images from end-to-end field use. Ground claims about new populations or acquisition conditions in evidence for those conditions; otherwise identify the limitation.
   Source: [REFORMS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11092361/), items 8a and 8b, external validity. Scope transfer is a reporting convention, not evidence of external validity for this study.

### Preference

10. **Use a flexible context-evidence-implication progression (preference).** Within the existing seven sections, orient the reader, present the relevant evidence and explain the local implication. Organize Results around comparisons. Add material that benefits the argument; no paragraph formula, sentence count or favorable outcome is required.
    Source: [Mensh and Kording](https://pmc.ncbi.nlm.nih.gov/articles/PMC5619685/), Rule 3. "Stick to the context-content-conclusion (C-C-C) scheme."

11. **Edit for continuity and parallel comparison (preference).** Put familiar context before new details, keep subjects near verbs, and use consistent terms and parallel wording for comparable descriptions. Preserve necessary definitions, meaningful uncertainty and deliberate repetition.
    Source: [Gopen and Swan](https://www.cs.tufts.edu/comp/150FP/archive/george-gopen/sci.html), sentence structure. "grammatical subjects should be followed as soon as possible by their verbs."
    Expert editing advice, not a controlled comprehension result.

12. **Use exemplars as optional reporting patterns (preference).** Borrow techniques only where they fit the actual evidence: disclose the coverage of relationship information, name evaluation-policy inputs and sources of variation, and connect diagnostics to concrete errors without treating proposed causes as observations. For a diagnostic example, report the actual method, input/settings and sample-selection policy, separating observations from proposed explanations and stating the limits of any checks performed. These are reporting applications, not findings tested by the saliency studies. Examples impose neither experimental controls nor mandatory figures/tables; the owning project protocol determines frozen inputs and prespecified samples.
    Source: the three field exemplars in Techniques; author confirmed flexible adoption on 2026-10-31. Their individual techniques are conventions, and this project's optional use is preference.

13. **Choose and explain displays for the reading task (preference).** Use tables for numerical lookup and figures where they help inspect a comparison. Label uncertainty displays and explain the supported reading. Use either or both where beneficial; no fixed display set is required. These applications are inferences from the bounded display studies, not experimentally established interventions for ML reviewers.
    Source: Correll and Gleicher; Hink, Wogalter and Eustace, cited above. Author confirmed flexible presentation patterns on 2026-10-31.

## Techniques

- Mohanty, Hughes and Salathe, [Using Deep Learning for Image-Based Plant Disease Detection](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full), Frontiers (2016), CC BY, Methods / Measurement of performance: report the coverage of same-leaf mappings, as the authors do for their partial mappings.
- Xiang, Shi and Zhu, [Quantifying the reliability gap](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1826962/full), Frontiers (2026), CC BY, Methods section 2.1: name each policy's information access and repeated variation. The authors distinguish access to labeled target calibration data, recalibrate per repeat and bound their task to cropped-image classification.
- Wei et al., [A Large-Scale In-the-wild Dataset for Plant Disease Segmentation](https://www.nature.com/articles/s41597-025-06513-4), Scientific Data (2026), CC BY-NC-ND 4.0 with possible separate third-party terms, Technical Validation / Failure case analysis: tie metrics to concrete failures. Spatially correct predictions can score zero IoU after misclassification. This segmentation example illustrates diagnostic reporting; its performance findings do not transfer to classification.

## Coverage and open questions

Coverage: `partial`. The PCSC 2027 submission gate above remains open: only historical 2026 pages were verified, and the organizer site failed secure access. This does not establish that a 2027 call is absent.

No recognized plant-disease ML checklist, universal equal-tuning-budget rule or direct test of these writing interventions on PCSC reviewers was verified. Apply the general guidance within each rule's stated scope. The bank entry retains the search and access-failure record; inaccessible sources support no retained rule.

## Humanizer

Activate Humanizer only when the user invokes `follow-writing-guide` or explicitly requests Humanizer. Once activated, it is on for manuscript prose. Apply its editing patterns while preserving scientific meaning, quotations, citations, equations, metric names and definitions. If the skill is unavailable, use this local fallback: remove empty staging, inflated claims, redundant recaps and decorative phrasing while preserving necessary content.

Collisions: the uncertainty disclosures and bounded claims in the authoritative rules take precedence over deleting qualifiers or repetition. The flexible progression must not force Humanizer pattern 2, One-line closers and dramatic fragments. Parallel comparison language may deliberately repeat technical terms; preserve the meaning while removing empty repetition. Meaningful qualifications survive pattern 9, Stacked qualifiers. Preference/convention wording yields to Humanizer where there is a stylistic conflict; scientific facts remain intact.

Protection: Humanizer never rewrites the wording or effect of a requirement or authoritative guideline rule. It may edit the surrounding manuscript prose only while preserving that rule's effect.

## Deviation rule

The author may deviate from a convention or preference with a one-line reason in the draft. A deviation from a requirement or authoritative guideline needs explicit author approval for that passage. Empirical rules apply within their documented scope; a display or editing recommendation inferred beyond a study is a flexible preference, not a new empirical obligation.

Author confirmed flexible narrative, prose and presentation defaults on 2026-10-31. Add what benefits the paper; display choices require no additional experiments.

## Provenance

Bank entry: pcsc-jute-empirical-ml-paper.
Revision: 2026-10-31
Wiki locator: qu1r0raOS-wikis/writing/guides/pcsc-jute-empirical-ml-paper.md
The project guide is self-contained; the wiki is maintenance provenance.

## Existing instructions

| Existing instruction | Disposition | Authority |
| --- | --- | --- |
| Direct professional prose in public docs, commits, PRs and comments; credited to owner alone, without attribution boilerplate or emojis | kept: delivery process | docs/agents/coding.md |
| Preserve historical submissions and corrected evidence; ignore private manuscripts and runtime artifacts | kept: preservation process | docs/agents/coding.md |
| Bounded review does not certify zero leakage, independence or expert labels; disclose historical test exposure | kept: scientific evidence boundary; reflected in reporting | docs/agents/research-context.md |
| Diagnostics use frozen checkpoints/prespecified samples and settings; avoid unsupported localization/generalization/leakage claims | kept: protocol process; reflected in reporting | docs/agents/research-context.md |
| Research index and active private manuscript workflow are optional on-demand context | kept: routing/process | docs/agents/research-context.md |

Verify any external scientific disclosure obligations independently of delivery prose conventions.
