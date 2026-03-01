---
title: Do Biomedical Tasks Require Biomedical Foundation Models?
keywords:
- markdown
- publishing
- manubot
lang: en-US
date-meta: '2026-03-01'
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
  <meta name="dc.date" content="2026-03-01" />
  <meta name="citation_publication_date" content="2026-03-01" />
  <meta property="article:published_time" content="2026-03-01" />
  <meta name="dc.modified" content="2026-03-01T07:03:55+00:00" />
  <meta property="article:modified_time" content="2026-03-01T07:03:55+00:00" />
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
  <link rel="alternate" type="text/html" href="https://AdaptInfer.github.io/fm-survey/v/fe04c15aecc5f96a5bcd07614a5066897b404875/" />
  <meta name="manubot_html_url_versioned" content="https://AdaptInfer.github.io/fm-survey/v/fe04c15aecc5f96a5bcd07614a5066897b404875/" />
  <meta name="manubot_pdf_url_versioned" content="https://AdaptInfer.github.io/fm-survey/v/fe04c15aecc5f96a5bcd07614a5066897b404875/manuscript.pdf" />
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
([permalink](https://AdaptInfer.github.io/fm-survey/v/fe04c15aecc5f96a5bcd07614a5066897b404875/))
was automatically generated
from [AdaptInfer/fm-survey@fe04c15](https://github.com/AdaptInfer/fm-survey/tree/fe04c15aecc5f96a5bcd07614a5066897b404875)
on March 1, 2026.
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



## Integrating Foundation Models with Biomedical Data Systems

How foundation models interact with structured biomedical data and knowledge sources, including electronic health records, ontologies, and databases.


## Evaluation, Reliability, and Deployment of Biomedical Foundation Models

Challenges in deploying and evaluating biomedical foundation models and ensuring reliability, including benchmarking, dataset bias, interpretability, and clinical validation.


## Open Problems and Future Directions

Key unresolved questions and research opportunities that will shape the development and deployment of foundation models in biomedicine.


## Conclusions


## References {.page_break_before}

<!-- Explicitly insert bibliography here -->
<div id="refs"></div>


