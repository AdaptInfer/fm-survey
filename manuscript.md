---
title: Do Biomedical Tasks Require Biomedical Foundation Models?
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2026-05-04'
author-meta:
- Jingyun Jia
- Zhiyuan Li
- Ben Lengerich
header-includes: |
  <!--
  Manubot generated metadata rendered from header-includes-template.html.
  Suggest improvements at https://github.com/manubot/manubot/blob/main/manubot/process/header-includes-template.html
  -->
  <meta name="dc.format" content="text/html" />
  <meta property="og:type" content="article" />
  <meta name="dc.title" content="Do Biomedical Tasks Require Biomedical Foundation Models?" />
  <meta name="citation_title" content="Do Biomedical Tasks Require Biomedical Foundation Models?" />
  <meta property="og:title" content="Do Biomedical Tasks Require Biomedical Foundation Models?" />
  <meta property="twitter:title" content="Do Biomedical Tasks Require Biomedical Foundation Models?" />
  <meta name="dc.date" content="2026-05-04" />
  <meta name="citation_publication_date" content="2026-05-04" />
  <meta property="article:published_time" content="2026-05-04" />
  <meta name="dc.modified" content="2026-05-04T15:49:27+00:00" />
  <meta property="article:modified_time" content="2026-05-04T15:49:27+00:00" />
  <meta name="dc.language" content="en-US" />
  <meta name="citation_language" content="en-US" />
  <meta name="dc.relation.ispartof" content="Manubot" />
  <meta name="dc.publisher" content="Manubot" />
  <meta name="citation_journal_title" content="Manubot" />
  <meta name="citation_technical_report_institution" content="Manubot" />
  <meta name="citation_author" content="Jingyun Jia" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0009-0006-3241-3485" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Zhiyuan Li" />
  <meta name="citation_author_institution" content="Department of Computer Sciences, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0009-0006-6016-7381" />
  <meta name="twitter:creator" content="@None" />
  <meta name="citation_author" content="Ben Lengerich" />
  <meta name="citation_author_institution" content="Department of Statistics, University of Wisconsin-Madison" />
  <meta name="citation_author_orcid" content="0000-0001-8690-9554" />
  <meta name="twitter:creator" content="@ben_lengerich" />
  <link rel="canonical" href="https://AdaptInfer.github.io/fm-survey/" />
  <meta property="og:url" content="https://AdaptInfer.github.io/fm-survey/" />
  <meta property="twitter:url" content="https://AdaptInfer.github.io/fm-survey/" />
  <meta name="citation_fulltext_html_url" content="https://AdaptInfer.github.io/fm-survey/" />
  <meta name="citation_pdf_url" content="https://AdaptInfer.github.io/fm-survey/manuscript.pdf" />
  <link rel="alternate" type="application/pdf" href="https://AdaptInfer.github.io/fm-survey/manuscript.pdf" />
  <link rel="alternate" type="text/html" href="https://AdaptInfer.github.io/fm-survey/v/c85cb449469b31b6b728eed65453617f7993d4ce/" />
  <meta name="manubot_html_url_versioned" content="https://AdaptInfer.github.io/fm-survey/v/c85cb449469b31b6b728eed65453617f7993d4ce/" />
  <meta name="manubot_pdf_url_versioned" content="https://AdaptInfer.github.io/fm-survey/v/c85cb449469b31b6b728eed65453617f7993d4ce/manuscript.pdf" />
  <meta property="og:type" content="article" />
  <meta property="twitter:card" content="summary_large_image" />
  <link rel="icon" type="image/png" sizes="192x192" href="https://manubot.org/favicon-192x192.png" />
  <link rel="mask-icon" href="https://manubot.org/safari-pinned-tab.svg" color="#ad1457" />
  <meta name="theme-color" content="#ad1457" />
  <!-- end Manubot generated metadata -->
bibliography:
- content/manual-references.json
manubot-output-bibliography: output/references.json
manubot-output-citekeys: output/citations.tsv
manubot-requests-cache-path: ci/cache/requests-cache
manubot-clear-requests-cache: false
...






<small><em>
This manuscript
([permalink](https://AdaptInfer.github.io/fm-survey/v/c85cb449469b31b6b728eed65453617f7993d4ce/))
was automatically generated
from [AdaptInfer/fm-survey@c85cb44](https://github.com/AdaptInfer/fm-survey/tree/c85cb449469b31b6b728eed65453617f7993d4ce)
on May 4, 2026.
</em></small>



## Authors



+ **Jingyun Jia**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0006-3241-3485](https://orcid.org/0009-0006-3241-3485)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [Clouddelta](https://github.com/Clouddelta)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
  </small>

+ **Zhiyuan Li**
  <br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0009-0006-6016-7381](https://orcid.org/0009-0006-6016-7381)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [LZYEIL](https://github.com/LZYEIL)
    <br>
  <small>
     Department of Computer Sciences, University of Wisconsin-Madison
  </small>

+ **Ben Lengerich**
  ^[✉](#correspondence)^<br>
    ![ORCID icon](images/orcid.svg){.inline_icon width=16 height=16}
    [0000-0001-8690-9554](https://orcid.org/0000-0001-8690-9554)
    · ![GitHub icon](images/github.svg){.inline_icon width=16 height=16}
    [blengerich](https://github.com/blengerich)
    · ![Twitter icon](images/twitter.svg){.inline_icon width=16 height=16}
    [ben_lengerich](https://twitter.com/ben_lengerich)
    <br>
  <small>
     Department of Statistics, University of Wisconsin-Madison
  </small>


::: {#correspondence}
✉ — Correspondence possible via [GitHub Issues](https://github.com/AdaptInfer/fm-survey/issues)
or email to
Ben Lengerich \<lengerich@wisc.edu\>.


:::


## Abstract {.page_break_before}

Foundation models are increasingly used across biomedical domains, including clinical text, medical imaging, genomics, and protein modeling. At the same time, rapidly advancing general-purpose foundation models can often be adapted to biomedical tasks through prompting, fine-tuning, and tool use. This raises a central question: **do biomedical applications require domain-specific foundation models, or can general models be adapted effectively?**

In this review, we examine the landscape of biomedical foundation models and compare domain-specific pretraining with adaptation of general models. We discuss how these approaches interact with biomedical data systems and the challenges of evaluation, reliability, and deployment. We conclude by outlining key open problems that will shape the future of foundation models in biomedical research and healthcare.


## Introduction


## The Current State of Domain-Specific Biomedical Foundation Models

Overview of foundation models trained directly on biomedical data across major modalities such as clinical text, imaging, genomics, proteins, and EHRs.


### Adapting General Foundation Models to Biomedical Tasks

How general-purpose foundation models (e.g., large language and vision models) are adapted to biomedical applications through prompting, fine-tuning, and tool use.


## Overview
An important theme in biomedical AI is that model performance depends not only on the model itself, but also on how effectively it interacts with biomedical data systems. In practice, foundation models often need to connect unstructured inputs such as clinical text, medical images, or biological sequences with structured and semi-structured resources such as electronic health records (EHRs), ontologies, and curated biomedical databases [@doi:10.1038/sdata.2016.35; @doi:10.1093/nar/gkh061; @doi:10.1093/nar/gkaa1113]. EHRs are especially challenging in this respect because they are longitudinal, sparse, noisy, and irregularly timed, with measurements collected according to clinical need rather than on a fixed schedule [@doi:10.1038/sdata.2016.35].

This perspective is closely tied to the central question of this review: whether biomedical tasks are better addressed by domain-specific foundation models or by adapting general foundation models. Domain-specific models are often designed around biomedical data structures, whereas general models usually require prompting, retrieval, or tool use to interact effectively with structured biomedical information. Integration with biomedical data systems therefore provides a useful lens for comparing the strengths and limitations of these two approaches [@doi:10.1093/jamia/ocae074; @doi:10.1093/jamia/ocae202].

## Exsiting models
Existing work on integrating foundation models with biomedical data systems can be grouped into several paradigms. These categories are not rigid, and many practical systems combine multiple strategies [@doi:10.1093/jamia/ocae074].

#### Native integration in domain-specific models

One approach builds domain-specific models that directly operate on biomedical structures, especially EHRs. BEHRT adapts the Transformer architecture to longitudinal patient records by modeling diagnosis and treatment codes together with visit and demographic information [@doi:10.1038/s41598-020-62922-y]. Med-BERT similarly uses large-scale structured EHR data with pretraining objectives inspired by masked language modeling to learn reusable patient representations [@doi:10.1038/s41746-021-00455-y]. Related work such as CLMBR also learns representations from longitudinal clinical records while preserving temporal structure [@doi:10.1016/j.jbi.2020.103637].

The advantage of this paradigm is strong alignment with biomedical structure: temporal patterns, code co-occurrence, and patient history can be modeled more naturally than through generic text interfaces [@doi:10.1038/s41598-020-62922-y; @doi:10.1038/s41746-021-00455-y]. Its limitation is reduced portability, since these models are often tied to specific coding systems, institutions, or data formats and require substantial domain-specific pretraining [@doi:10.1016/j.jbi.2020.103637; @doi:10.1093/jamia/ocae074].

#### Adaptation of general foundation models

A second approach adapts general-purpose foundation models to biomedical data systems instead of designing new architectures from scratch. Structured biomedical inputs are often serialized into natural language or otherwise reformatted into model-compatible forms. Med-PaLM is a representative example, showing how a general large language model can be adapted to medical reasoning through instruction tuning rather than fully domain-specific pretraining [@doi:10.1038/s41586-023-06291-2].

In practice, this adaptation may involve prompt-based serialization of patient records, biomedical fine-tuning, or tool-mediated access to external biomedical systems [@doi:10.1038/s41586-023-06291-2; @doi:10.1093/jamia/ocae074]. The main benefit is flexibility, since the same model can support multiple tasks and modalities. However, irregular time series, coded variables, and hierarchical schemas are not naturally captured through prompt text alone, so performance may depend heavily on input formatting and orchestration [@doi:10.1093/jamia/ocae202].

#### Retrieval-augmented integration

A third paradigm combines foundation models with external biomedical knowledge sources at inference time. Instead of relying only on parametric memory, these systems retrieve relevant papers, database entries, clinical guidelines, or ontology-linked documents and condition their outputs on that evidence [@doi:10.1093/jamia/ocaf008; @doi:10.1371/journal.pdig.0000877].

Several variants already exist. Some systems use literature-grounded retrieval for biomedical question answering or clinical decision support [@doi:10.1093/jamia/ocaf008]. Others augment large language models with external medical knowledge bases, as in MKRAG [@arxiv:2309.16035]. Retrieval can also be structured rather than purely textual; for example, KG-RAG uses relations from the SPOKE biomedical knowledge graph to guide prompt generation [@doi:10.1093/bioinformatics/btae560]. These systems are attractive because they are more transparent and easier to update, but they introduce new bottlenecks when retrieval is incomplete, noisy, or poorly used by the downstream model [@doi:10.1093/jamia/ocaf008; @doi:10.1371/journal.pdig.0000877].

#### Graph- and ontology-aware integration

Another paradigm integrates foundation models with biomedical graphs or ontologies, especially in settings where diseases, genes, proteins, drugs, and phenotypes are linked through explicit semantic or biological relationships [@doi:10.1093/nar/gkh061; @doi:10.1093/nar/gkaa1113].

Biomedical knowledge graphs can support reasoning over gene-disease-drug associations, while ontologies can constrain concept normalization and improve consistency across datasets [@doi:10.1093/bioinformatics/btae560; @doi:10.2196/62924]. In some systems, graph structure is used as an external source of biomedical facts during generation, as in KG-RAG [@doi:10.1093/bioinformatics/btae560]. In others, large language models help construct or extend graph resources from clinical text and biomedical literature [@arxiv:2301.12473]. These approaches preserve domain structure that generic sequence models often ignore, but graph resources are often incomplete, heterogeneous, and difficult to integrate cleanly with large pretrained models [@doi:10.1093/bioinformatics/btae560; @doi:10.2196/62924].

#### Multimodal integration

Many biomedical applications require models to combine multiple modalities, including EHR data, clinical notes, imaging, pathology slides, molecular profiles, and literature-derived knowledge [@pmid:39321458; @pmid:40754135].

In clinical settings, multimodal systems may combine imaging with reports or EHR trajectories with free-text notes [@pmid:39321458]. In research settings, they may connect molecular measurements, pathology images, and knowledge-graph information for tasks such as drug discovery or precision medicine [@doi:10.1016/j.drudis.2024.104254]. More recent medical multimodal language-model work similarly aims to place visual and textual biomedical evidence into a shared reasoning framework [@pmid:39321458]. This paradigm is attractive because biomedical decision-making is rarely unimodal, but it is also one of the most difficult to implement because alignment is imperfect, modalities are often missing, and evaluation becomes more complex as the system grows [@pmid:40754135].


## Evaluation, Reliability, and Deployment of Biomedical Foundation Models

### Evaluation
#### Task-Centered Utility over Generic Benchmarks
Evaluation of biomedical foundation models should be task-centered rather than model-centered. The central question is not whether a model scores highly on a generic leaderboard, but whether it improves performance on the biomedical task that actually matters, such as diagnosis, prognosis, survival prediction, retrieval, report generation, or biomarker discovery [@agrawal2025evaluation; @wornow2023shaky]. This requires clearly specifying the intended use case, the target population, the relevant comparator, and the consequences of model errors. Recent clinical benchmarking studies, including evaluations of general-purpose models on clinical decision-support tasks, illustrate the need for task-specific assessment rather than reliance on general-purpose leaderboards [@sandmann2025deepseek; @agrawal2025evaluation; @wornow2023shaky].

#### Modality-Aware Evaluation

Biomedical foundation models span heterogeneous modalities, including clinical text, electronic health records, radiology and pathology images, molecular structures, protein sequences, genomics, transcriptomics, perturbation screens, and single-cell measurements [@moor2023foundation; @agrawal2025evaluation]. Evaluation therefore cannot rely on a single benchmark format. Clinical language models may be assessed through question answering, summarization, and decision-support tasks; imaging models through classification, segmentation, retrieval, and external validation; EHR models through prognosis, phenotyping, and risk stratification; and molecular or protein models through structure prediction, binding prediction, variant effect prediction, target prioritization, or downstream experimental utility.

This diversity means that biomedical foundation model evaluation should distinguish among clinical utility, biological validity, and experimental actionability. A model that performs well on retrospective clinical prediction may not generate useful biological hypotheses, while a model that captures molecular regularities may not be directly deployable in patient-facing settings.

#### Moving Beyond Discriminative Metrics
Evaluation must also go beyond discrimination alone. Metrics such as AUROC, AUPRC, F1, concordance index, and segmentation overlap are informative, but they capture only part of model performance. AUROC summarizes how well a model ranks positive cases above negative cases across decision thresholds, whereas AUPRC is often more informative when clinically relevant positive cases are rare. Methodological reviews in medical imaging emphasize that proper assessment must also incorporate calibration, uncertainty, and other task-specific criteria. A model can appear accurate on average while still producing poorly calibrated or clinically misleading outputs [@kocak2025evaluationmetrics]. The choice of metric should therefore follow the downstream use case: prediction tasks require calibration and risk stratification, retrieval and generation tasks demand expert judgment of quality, and discovery-oriented applications necessitate reproducibility and biological plausibility.

#### Addressing Benchmark Contamination
Benchmark design directly impacts the validity of results. If test sets are contaminated by publicly available pretraining data, reported performance may exaggerate true progress because the model may partially recall previously seen cases, reports, images, or benchmark answers rather than generalize to genuinely unseen biomedical scenarios. Pathology benchmarking initiatives have responded to related validity concerns by using automated evaluation pipelines for external model assessment, reducing the need to publicly circulate sensitive test cohorts [@campanella2025clinicalbenchmark]. This issue is especially acute for foundation models, whose massive pretraining corpora make overlap difficult to rule out. Fair evaluation generally requires comparing foundation models against robust task-specific baselines, smaller domain-adapted models, and clinically realistic alternatives under contamination-aware protocols [@campanella2025clinicalbenchmark; @agrawal2025evaluation]. Often, the most relevant comparator is not the largest available model, but the strongest clinically realistic alternative.

#### Prospective Assessment and the Cost of Expert Review
Retrospective benchmarking should not be treated as the endpoint of evaluation. Evidence becomes substantially stronger when evaluation extends toward workflow-aware analysis and prospective study. A prospective trial of open-source melanoma AI demonstrated that while the algorithmic system had the potential to improve dermatologist decision-making in clinical practice, larger randomized studies remained necessary before routine adoption [@marchetti2023proveai]. Furthermore, human-in-the-loop evaluation introduces significant financial and temporal bottlenecks due to the high cost of expert clinician annotation. While rigorous expert assessment is indispensable, exploring scalable solutions—such as LLM-as-a-judge frameworks validated by human spot-checking—may become necessary to sustain thorough evaluation pipelines. Ultimately, the value of an AI system is established only when predictive performance is connected to usability, workflow fit, and patient-relevant outcomes [@rosen2025clinicalimplementation].

### Reliability
#### Defining High-Stakes Dependability
Reliability is a stricter requirement than benchmark performance. In the biomedical setting, it concerns whether model outputs remain trustworthy when deployment conditions depart from the development setting, and whether failures are sufficiently visible, bounded, and manageable. This matters critically for foundation models, which are often promoted as flexible, cross-modal systems—a flexibility that paradoxically complicates validation and oversight [@moor2023foundation; @agrawal2025evaluation]. Reliability should therefore be understood as a combination of robustness under shift, rigorous uncertainty quantification, and protection against clinically consequential failure modes.

#### Distribution Shift and Shortcut Learning
A defining reliability problem in biomedicine is distribution shift. Clinical data vary across hospitals, scanners, laboratories, note-writing styles, demographic groups, and calendar time, frequently causing strong internal performance to degrade upon external deployment. Ly et al. [@ly2024shortcut] demonstrated across 13 medical datasets that apparent model performance was overestimated by up to 20% on average because models exploited hidden data-acquisition biases rather than disease-relevant signals [@ong2024shortcut]. Related findings in medical imaging indicate that fairness assessed in the original distribution can break down under external generalization, warning that subgroup reliability cannot be assumed [@yang2024limits]. Therefore, evaluation must explicitly examine subgroup stability and resistance to shortcut learning, rather than presuming that massive scale or domain pretraining automatically resolves these vulnerabilities.

#### Uncertainty Quantification and Strategic Deferral
Reliability inherently requires models to signal when they are likely to fail. In clinical settings, an incorrect answer delivered with high confidence is significantly more dangerous than no answer at all. Banerji et al. argue that clinical AI tools should communicate predictive uncertainty at the level of the individual patient, rather than relying solely on population-level summary statistics [@banerji2023uncertainty]. Similarly, frameworks for responsible deployment emphasize selective prediction, wherein models abstain, defer, or restrict outputs in settings where generalization is uncertain [@goetz2024generalization]. For biomedical foundation models, true reliability incorporates robust uncertainty quantification (UQ), abstention behaviors, and escalation pathways to human clinicians. Models that cannot express uncertainty remain difficult to justify in routine high-stakes biomedical use.

#### Generative Faithfulness and Safety
These issues become acute for generative models. In tasks such as note summarization, report drafting, or clinical question answering, the central requirement is not surface-level plausibility, but factual faithfulness and clinical safety. Tam et al. reviewed 142 studies on human evaluation of healthcare LLMs, identifying substantial gaps in generalizability and advocating for standardized expert-centered assessment [@tam2024human]. More directly, Asgari et al. analyzed hallucinations and omissions in medical text summarization, finding that while omissions were more common, hallucinations constituted major errors with severe potential for downstream clinical harm [@asgari2025clinicalsafety]. Consequently, reliability in generative tasks cannot be established through automated metrics alone; it mandates structured failure-mode analysis and explicit assessment of potential patient harm.

### Deployment
#### The Sociotechnical Transition
Deployment is where the central question of this review becomes most concrete. Even a highly accurate and reliable biomedical foundation model is not genuinely useful unless it seamlessly integrates into real biomedical systems and workflows. Implementation science underscores that adoption depends on workflow fit, comparison against existing practice, and implementation outcomes rather than mere model accuracy [@vandesande2024multifaceted; @you2025clinicaltrials; @wells2025fairai]. Deployment must be treated as a multifaceted sociotechnical challenge rather than the final technical stage of software engineering.

#### Workflow Integration and Human-AI Collaboration
A primary requirement is that biomedical foundation models must actively support human work. Successful deployment depends on embedding the model into existing decision pathways without increasing cognitive or administrative burdens. A recent Nature Medicine study on decision support for colorectal cancer surgery utilized a stepwise design—moving from registry-based development to prospective clinical implementation with risk-tailored intervention bundles—explicitly linking predictions to actionable treatment workflows [@rosen2025clinicalimplementation]. Clinician-led deployment studies consistently show more meaningful impacts than technologist-led initiatives, underscoring the necessity of domain leadership [@li2025leadership]. Evaluative focus is shifting toward assessing human-AI collaboration through metrics like time savings and task completion rather than isolated accuracy [@wang2025leads].

#### Deployment Beyond the Clinic: Closed-Loop Biomedical Discovery

Deployment of biomedical foundation models is not limited to clinical decision support. In discovery-oriented settings, foundation models may be integrated into closed-loop experimental pipelines, where models propose hypotheses, prioritize molecules or perturbations, guide robotic experiments, and update predictions based on new assay results. These systems connect model inference with laboratory action, making deployment a problem of both computation and experimental operations.

Closed-loop biomedical discovery introduces challenges that differ from clinical AI, including experimental batch effects, assay variability, robotic failure modes, reagent constraints, sample availability, cost-aware experimental design, and lab-specific protocols. Evaluation should therefore extend beyond retrospective prediction accuracy to include experimental yield, reproducibility across batches, reduction in search cost, feasibility of suggested experiments, and biological plausibility.

#### Operational Feasibility and Regulatory Compliance
Foundation models may appear theoretically attractive but practically infeasible due to infrastructure demands, latency, privacy constraints, and stringent regulatory requirements. Large multimodal systems require substantial computing power and complex interfacing with existing EHR, LIS, or PACS infrastructure. Utilizing external APIs may violate institutional data governance, while local deployment remains computationally expensive. Furthermore, deployment strategies must align with regulatory frameworks, such as the FDA's guidelines for Software as a Medical Device (SaMD) or equivalent international standards. Model updates create ongoing compliance challenges, as behavior shifts post-retraining require renewed local validation and regulatory review.

#### Staged Implementation Strategies
Because biomedical applications carry high stakes, implementation frameworks increasingly advocate for controlled, staged introductions. You et al. propose a clinical-trials-inspired framework spanning safety, efficacy, effectiveness, and monitoring, explicitly detailing silent-mode testing—where predictions are observed without affecting patient care—as a crucial early phase [@you2025clinicaltrials]. Rosenthal et al. similarly argue that adaptive AI systems require "dynamic deployment" with continuous real-time monitoring rather than linear, one-time approvals [@rosenthal2025dynamicdeployment]. In pathology, Campanella et al. successfully deployed an AI-assisted workflow using a four-month silent trial to prove noninferiority under real clinical operating conditions before full integration [@campanella2025realworlddeployment].

#### Continuous Post-Deployment Governance
Deployment is an ongoing lifecycle. Post-deployment monitoring, governance, and maintenance are essential because biomedical data, workflows, and institutional practices experience drift over time. A review of monitoring methods for clinical AI highlighted that practical guidance remains sparse despite universal agreement on the necessity of ongoing surveillance [@andersen2024monitoring]. Healthcare organizations urgently need structural frameworks to address transparency, accountability, equity, privacy, and robustness long after the initial launch [@saenz2024responsibleuse; @wells2025fairai]. The deployment case for biomedical foundation models is compelling only when they are integrated thoughtfully, introduced progressively, and rigorously monitored throughout their operational lifespan.


## Open Problems and Future Directions

Despite substantial progress in biomedical foundation models, several fundamental challenges remain unresolved. Building on the central themes of this review—namely the tension between domain-specific and general models, the importance of system-level integration, and the need for reliable evaluation—we highlight three core open problems that are likely to shape the future of the field. These challenges are deeply interconnected and reflect limitations not only in model design, but also in data, training objectives, and deployment environments.

### Domain-Specific Modeling versus General Adaptation

A central unresolved question is whether biomedical tasks inherently require domain-specific foundation models, or whether increasingly capable general-purpose models can be adapted effectively. Domain-specific models, such as BioBERT and ClinicalBERT [@doi:10.1093/bioinformatics/btz682; @doi:10.48550/arXiv.1904.05342], as well as highly specialized systems like AlphaFold [@doi:10.1038/s41586-021-03819-2], demonstrate that incorporating domain knowledge into pretraining can yield substantial gains, particularly when the underlying structure of the problem is well understood. These successes suggest that inductive biases tailored to biomedical data remain highly valuable.

However, recent advances in large-scale general foundation models have challenged this view. Models such as GPT-4 and PaLM have shown promising performance on some biomedical reasoning tasks with minimal domain-specific training [@doi:10.48550/arXiv.2303.08774; @doi:10.48550/arXiv.2204.02311; @doi:10.1038/s41586-023-06291-2], raising the possibility that sufficiently large models may implicitly learn transferable structure. At the same time, these models often exhibit hallucinations and lack reliable grounding in biomedical knowledge [@doi:10.48550/arXiv.2301.12787], indicating that scale alone may not resolve domain-specific challenges. For instance, a general model might fail to distinguish between clinically important subtypes of a disease that require different treatment strategies.

The absence of a principled framework for resolving this trade-off remains a major limitation. Without such understanding, model development is driven largely by empirical performance on isolated benchmarks rather than by a deeper characterization of task requirements. This motivates a critical direction for future research: identifying the conditions under which domain-specific structure is essential versus when it can be effectively approximated through adaptation. Progress in this area is necessary not only for improving performance, but also for reducing the substantial computational and data costs associated with training specialized models from scratch. In particular, approaches that combine general representations with domain-specific grounding, such as retrieval-augmented systems [@doi:10.48550/arXiv.2005.11401], are promising precisely because they attempt to reconcile these competing paradigms rather than choosing between them.

### Misalignment Between Pretraining Objectives and Biomedical Utility

A second major challenge arises from the mismatch between standard pretraining objectives and the requirements of real-world biomedical applications. Most foundation models are trained using self-supervised objectives, such as next-token prediction, which encourage the learning of general-purpose representations from large-scale data [@doi:10.48550/arXiv.1909.05858; @doi:10.48550/arXiv.2005.14165]. While this paradigm has been remarkably successful, it does not directly optimize for the types of decisions that biomedical systems are expected to support, such as diagnosis, risk prediction, or treatment planning. This limitation arises because next-token prediction is designed to model the distribution of observed text rather than to capture decision-relevant or causal structure. As a result, models may generate fluent and contextually plausible outputs without ensuring correctness or reliability in high-stakes biomedical settings.

This misalignment becomes evident during deployment. Models that perform well on standard benchmarks may still produce outputs that are clinically unsafe, poorly calibrated, or difficult to interpret [@doi:10.48550/arXiv.2301.12787; @doi:10.48550/arXiv.1706.04599]. As a result, extensive post-training procedures—including supervised fine-tuning, instruction tuning, and reinforcement learning from human feedback—are required to align model behavior with user expectations [@doi:10.48550/arXiv.2203.02155; @doi:10.48550/arXiv.2305.18290]. However, these approaches rely heavily on curated datasets and human supervision, which are both costly and difficult to scale in biomedical contexts.

The persistence of this gap suggests that current training paradigms are fundamentally incomplete. Future progress therefore depends on developing methods that more directly align model training with downstream utility. This includes not only improved alignment techniques, but also a rethinking of objective functions to incorporate domain-specific notions of correctness, uncertainty, and risk. Such developments are essential because biomedical applications are inherently high-stakes, and models that optimize proxy objectives without capturing real-world utility may fail in ways that are both subtle and consequential. For example, predicting ICD codes is not equivalent to performing diagnosis. ICD codes are primarily designed for administrative and billing purposes and may only imperfectly reflect the underlying clinical reasoning process. This highlights the gap between optimizing models for proxy labels and developing systems that support clinically meaningful decisions, which in turn raises challenges for interpretability and clinical trust.

### Robustness Under Distribution Shift and System-Level Constraints

A third critical challenge is the lack of robustness of foundation models under distribution shift, particularly in the complex environments characteristic of biomedical data. Unlike many standard machine learning benchmarks, biomedical datasets are generated by heterogeneous and evolving processes. Clinical data, for example, reflects a combination of biological variation, patient behavior, clinician decision-making, and institutional practices [@doi:10.1038/s41591-018-0316-z; @doi:10.1038/s41591-020-0789-4]. As a result, models trained on one dataset often fail to generalize reliably to others. For example, models trained on data from one hospital may perform poorly when applied to another due to differences in patient populations, clinical practices, or data collection protocols, such as variation across imaging scanners or electronic health record systems.

This problem is not merely statistical but fundamentally structural. The underlying data-generating processes are often unstable and context-dependent, meaning that standard assumptions of independent and identically distributed data do not hold. In addition, deployment environments impose constraints that are rarely captured during model development, including integration with electronic health records, regulatory requirements, and the need for interpretability and user trust. In particular, interpretability plays a central role in biomedical settings, where model outputs must be understood and justified for clinical decision-making, error analysis, and regulatory approval.

These considerations suggest that improving model performance in isolation is insufficient. Instead, future progress requires a shift toward system-level thinking, in which models are designed as components within larger biomedical infrastructures. An important question at this level is whether biomedical AI should rely on a single unified multimodal model, or on modular systems that combine modality-specific models through a downstream decision component. While unified models may enable richer cross-modal representations, modular approaches may offer advantages in debugging and clinical integration. The relative merits of these approaches remain unclear, and identifying when each paradigm is preferable is an important open question.

This architectural perspective naturally motivates research into methods for handling distribution shift, such as domain adaptation and out-of-distribution detection [@doi:10.48550/arXiv.1906.02530], as well as approaches that incorporate causal structure or external knowledge to improve robustness. The importance of these directions stems from the fact that reliability in biomedical AI depends not only on predictive accuracy, but on the ability of systems to function consistently across diverse and changing real-world settings.

### Summary

Taken together, these challenges highlight the need for a more integrated understanding of biomedical foundation models. The key open problems identified here—balancing domain specificity and generality, aligning training objectives with real-world utility, and ensuring robustness under distributional and system-level constraints—are not independent issues, but rather reflect different aspects of the same underlying tension between model capability and practical deployment. Addressing these challenges will be essential for translating recent advances in foundation models into reliable and impactful biomedical applications.


## Conclusions


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>


