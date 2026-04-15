---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:77578
- loss:CoSENTLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: Efficient sequential Bayesian inference for state-space epidemic
    models using ensemble data assimilation. Estimating latent epidemic states and
    model parameters from partially observed, noisy data remains a major challenge
    in infectious disease modeling. State-space formulations provide a coherent probabilistic
    framework for such inference, yet fully Bayesian estimation is often computationally
    prohibitive because evaluating the observed-data likelihood requires integration
    over all latent trajectories. The Sequential Monte Carlo squared (SMC$^2$) algorithm
    offers a principled approach for joint state and parameter inference, combining
    an outer SMC sampler over parameters with an inner particle filter that estimates
    the likelihood up to the current time point. Despite its theoretical appeal, this
    nested particle filter imposes substantial computational cost, limiting routine
    use in near-real-time outbreak response. We propose Ensemble SMC$^2$ (eSMC$^2$),
    a scalable variant that replaces the inner particle filter with an Ensemble Kalman
    Filter (EnKF) to approximate the incremental likelihood at each observation time.
    While this substitution introduces bias via a Gaussian approximation, we mitigate
    finite-sample effects using an unbiased Gaussian density estimator and adapt the
    EnKF for epidemic data through state-dependent observation variance. This makes
    our approach particularly suitable for overdispersed incidence data commonly encountered
    in infectious disease surveillance. Simulation experiments with known ground truth
    and an application to 2022 United States (U.S.) monkeypox incidence data demonstrate
    that eSMC$^2$ achieves substantial computational gains while producing posterior
    estimates comparable to SMC$^2$. The method accurately recovers latent epidemic
    trajectories and key epidemiological parameters, providing an efficient framework
    for sequential Bayesian inference from imperfect surveillance data.
  sentences:
  - 'Why Some Models Resist Unlearning: A Linear Stability Perspective. Machine unlearning,
    the ability to erase the effect of specific training samples without retraining
    from scratch, is critical for privacy, regulation, and efficiency. However, most
    progress in unlearning has been empirical, with little theoretical understanding
    of when and why unlearning works. We tackle this gap by framing unlearning through
    the lens of asymptotic linear stability to capture the interaction between optimization
    dynamics and data geometry. The key quantity in our analysis is data coherence
    which is the cross sample alignment of loss surface directions near the optimum.
    We decompose coherence along three axes: within the retain set, within the forget
    set, and between them, and prove tight stability thresholds that separate convergence
    from divergence. To further link data properties to forgettability, we study a
    two layer ReLU CNN under a signal plus noise model and show that stronger memorization
    makes forgetting easier: when the signal to noise ratio (SNR) is lower, cross
    sample alignment is weaker, reducing coherence and making unlearning easier; conversely,
    high SNR, highly aligned models resist unlearning. For empirical verification,
    we show that Hessian tests and CNN heatmaps align closely with the predicted boundary,
    mapping the stability frontier of gradient based unlearning as a function of batching,
    mixing, and data/model alignment. Our analysis is grounded in random matrix theory
    tools and provides the first principled account of the trade offs between memorization,
    coherence, and unlearning.'
  - Bridge2AI Recommendations for AI-Ready Genomic Data. Rapid advancements in technology
    have led to an increased use of artificial intelligence (AI) technologies in medicine
    and bioinformatics research. In anticipation of this, the National Institutes
    of Health (NIH) assembled the Bridge to Artificial Intelligence (Bridge2AI) consortium
    to coordinate development of AI-ready datasets that can be leveraged by AI models
    to address grand challenges in human health and disease. The widespread availability
    of genome sequencing technologies for biomedical research presents a key data
    type for informing AI models, necessitating that genomics data sets are AI-ready.
    To this end, the Genomic Information Standards Team (GIST) of the Bridge2AI Standards
    Working Group has documented a set of recommendations for maintaining AI-ready
    genomics datasets. In this report, we describe recommendations for the collection,
    storage, identification, and proper use of genomics datasets to enable them to
    be considered AI-ready and thus drive new insights in medicine through AI and
    machine learning applications.
  - 'Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion
    in Malaysia. Despite accounting for 96.1% of all businesses in Malaysia, access
    to financing remains one of the most persistent challenges faced by Micro, Small,
    and Medium Enterprises (MSMEs). Newly established businesses are often excluded
    from formal credit markets as traditional underwriting approaches rely heavily
    on credit bureau data. This study investigates the potential of bank statement
    data as an alternative data source for credit assessment to promote financial
    inclusion in emerging markets. First, we propose a cash flow-based underwriting
    pipeline where we utilise bank statement data for end-to-end data extraction and
    machine learning credit scoring. Second, we introduce a novel dataset of 611 loan
    applicants from a Malaysian lending institution. Third, we develop and evaluate
    credit scoring models based on application information and bank transaction-derived
    features. Empirical results show that the use of such data boosts the performance
    of all models on our dataset, which can improve credit scoring for new-to-lending
    MSMEs. Finally, we will release the anonymised bank transaction dataset to facilitate
    further research on MSME financial inclusion within Malaysia''s emerging economy.'
- source_sentence: Double $q$-Wigner Chaos and the Fourth Moment. In this paper, we
    prove the Fourth Moment Theorem for sequences of (noncommutative) random variables
    given as sums of two stochastic integrals in two different parity orders of chaos,
    both in the free Wigner chaos setting and a $q$-Gaussian generalization. Specifically,
    we prove that convergence to the appropriate central limit distribution is mediated
    entirely by the behavior of the first four (mixed) moments of the two stochastic
    integrals, which in turn controls the $L^2$ norms of partial integral contractions
    of those kernels. The key step in both the free and $q$-Gaussian settings is a
    polarization identity for fourth cumulants of sums which holds only when the two
    terms have differing parities. These results are analogous to the recent preprint
    Fourth-Moment Theorems for Sums of Multiple Integrals by Basse-O'Connor, Kramer-Bang,
    and Svedsen in the classical Wiener-Itô chaos setting.
  sentences:
  - Scaling limit of the step-reinforced and stochastic Levy--Lorentz model on weakly
    entangled integer lattice. This paper describes the stochastic Levy--Lorentz gas
    driven by general long-range reference random walk on correlated and entangled
    random medium. Further consideration has been laid on the stochastic reinforcement
    of the underlying random walk, where it now possesses memory. Central limit theorems
    are obtained in both cases.
  - Quantum Computers, Predictability, and Free Will. This article focuses on the
    connection between the possibility of quantum computers, the predictability of
    complex quantum systems in nature, and the issue of free will.
  - 'ProFlow: Zero-Shot Physics-Consistent Sampling via Proximal Flow Guidance. Inferring
    physical fields from sparse observations while strictly satisfying partial differential
    equations (PDEs) is a fundamental challenge in computational physics. Recently,
    deep generative models offer powerful data-driven priors for such inverse problems,
    yet existing methods struggle to enforce hard physical constraints without costly
    retraining or disrupting the learned generative prior. Consequently, there is
    a critical need for a sampling mechanism that can reconcile strict physical consistency
    and observational fidelity with the statistical structure of the pre-trained prior.
    To this end, we present ProFlow, a proximal guidance framework for zero-shot physics-consistent
    sampling, defined as inferring solutions from sparse observations using a fixed
    generative prior without task-specific retraining. The algorithm employs a rigorous
    two-step scheme that alternates between: (\romannumeral1) a terminal optimization
    step, which projects the flow prediction onto the intersection of the physically
    and observationally consistent sets via proximal minimization; and (\romannumeral2)
    an interpolation step, which maps the refined state back to the generative trajectory
    to maintain consistency with the learned flow probability path. This procedure
    admits a Bayesian interpretation as a sequence of local maximum a posteriori (MAP)
    updates. Comprehensive benchmarks on Poisson, Helmholtz, Darcy, and viscous Burgers''
    equations demonstrate that ProFlow achieves superior physical and observational
    consistency, as well as more accurate distributional statistics, compared to state-of-the-art
    diffusion- and flow-based baselines.'
- source_sentence: 'The complexity of downward closures of indexed languages. Indexed
    languages are a classical notion in formal language theory, which has attracted
    attention in recent decades due to its role in higher-order model checking: They
    are precisely the languages accepted by order-2 pushdown automata. The downward
    closure of an indexed language -- the set of all (scattered) subwords of its members
    -- is well-known to be a regular over-approximation. It was shown by Zetzsche
    (ICALP 2015) that the downward closure of a given indexed language is effectively
    computable. However, the algorithm comes with no complexity bounds, and it has
    remained open whether a primitive-recursive construction exists. We settle this
    question and provide a triply (resp.\ quadruply) exponential construction of a
    non-deterministic (resp.\ deterministic) automaton. We also prove (asymptotically)
    matching lower bounds. For the upper bounds, we rely on recent advances in semigroup
    theory, which let us compute bounded-size summaries of words with respect to a
    finite semigroup. By replacing stacks with their summaries, we are able to transform
    an indexed grammar into a context-free one with the same downward closure, and
    then apply existing bounds for context-free grammars.'
  sentences:
  - Weak Charge Form Factor Determination at the Electron-Ion Collider. Determining
    the weak charge form factor, $F_W(Q^2)$, of nuclei over a continuous range of
    momentum transfers, $0\lesssim Q^2 \lesssim 0.1$ GeV$^2$, is essential for mapping
    out the distribution of neutrons in nuclei. The neutron density distribution has
    significant implications for a broad range of areas, including studies of nuclear
    structure, neutron stars, and physics beyond the Standard Model. Currently, our
    knowledge of $F_W(Q^2)$ comes primarily from fixed target experiments that measure
    the parity-violating longitudinal electron spin asymmetry in coherent elastic
    electron-ion scattering. Fixed target experiments, such as CREX and PREX-1,2,
    have provided high-precision weak charge form factor extractions for the $^{48}{\rm
    Ca}$ and $^{208}{\rm Pb}$ nuclei, respectively. However, a major limitation of
    fixed target experiments is that they each provide data only at a single value
    of $Q^2$. With the proposed Electron-Ion Collider (EIC) on the horizon, we explore
    its potential to impact the determination of the weak charge form factor. While
    it cannot compete with the precision of fixed target experiments, it can provide
    data over a wide and continuous range of $Q^2$ values, and for a wide variety
    of nuclei. We show that for integrated luminosities of $\mathcal{L} > $ 200/$A$
    fb$^{-1}$, where $A$ denotes the nucleus atomic weight, the EIC can be complementary
    to fixed target experiments, and can significantly impact constraints from CREX
    and PREX-1,2 by lifting degeneracies in theoretical models of the neutron density
    distribution.
  - Extended branching Rauzy induction. Branching Rauzy induction is a two-sided form
    of Rauzy induction that acts on regular interval exchange transformations (IETs).
    We introduce an extended form of branching Rauzy induction that applies to arbitrary
    standard IETs, including non-minimal ones. The procedure generalizes the branching
    Rauzy method with two induction steps, merging and splitting, to handle equal-length
    cuts and invariant components respectively. As an application, we show, via a
    stepwise morphic argument, that all return words in the language of an arbitrary
    IET cluster in the Burrows-Wheeler sense.
  - A unified approach to the Dirac fine structures on the $S$-spectrum and a connection
    with Jacobi polynomials. This paper contributes to the recently introduced theory
    of fine structures on the $S$-spectrum. We study, in a unified way, the functional
    calculi for axially Poly-Analytic-Harmonic functions on the $S$-spectrum. Axially
    Poly-Analytic-Harmonic functions of type $(β, m)$, for $β, m \in \mathbb{N}_0$
    belong to the kernel of the Dirac-Laplace operators $D^βΔ^m_{n+1}$ of type $(β,
    m)$ and contain as particular cases Poly-Analytic and Poly-Harmonic functions
    of axial type. By applying these operators to the Cauchy kernels $S^{-1}_L(s,x)$
    of (left) slice hyperholomorphic functions, we obtain an integral representation
    for axially Poly-Analytic-Harmonic functions. We point out that the kernels $D^βΔ^m_{n+1}S^{-1}_L(s,x)$
    have a remarkable connection with Jacobi polynomials. By replacing the paravector
    operator $T$ with commuting components in the kernels $D^βΔ^m_{n+1} S^{-1}_L(s,x)$,
    we obtain the associated resolvent operators. With these resolvent operators,
    denoted by $S^{-1}_{L, D^βΔ^m}(s,T)$, we define the associated functional calculi
    based on the $S$-spectrum and study their properties.
- source_sentence: 'SSNAPS: Audio-Visual Separation of Speech and Background Noise
    with Diffusion Inverse Sampling. This paper addresses the challenge of audio-visual
    single-microphone speech separation and enhancement in the presence of real-world
    environmental noise. Our approach is based on generative inverse sampling, where
    we model clean speech and ambient noise with dedicated diffusion priors and jointly
    leverage them to recover all underlying sources. To achieve this, we reformulate
    a recent inverse sampler to match our setting. We evaluate on mixtures of 1, 2,
    and 3 speakers with noise and show that, despite being entirely unsupervised,
    our method consistently outperforms leading supervised baselines in \ac{WER} across
    all conditions. We further extend our framework to handle off-screen speaker separation.
    Moreover, the high fidelity of the separated noise component makes it suitable
    for downstream acoustic scene detection. Demo page: https://ssnapsicml.github.io/ssnapsicml2026/'
  sentences:
  - 'LALM-as-a-Judge: Benchmarking Large Audio-Language Models for Safety Evaluation
    in Multi-Turn Spoken Dialogues. Spoken dialogues with and between voice agents
    are becoming increasingly common, yet assessing them for their socially harmful
    content such as violence, harassment, and hate remains text-centric and fails
    to account for audio-specific cues and transcription errors. We present LALM-as-a-Judge,
    the first controlled benchmark and systematic study of large audio-language models
    (LALMs) as safety judges for multi-turn spoken dialogues. We generate 24,000 unsafe
    and synthetic spoken dialogues in English that consist of 3-10 turns, by having
    a single dialogue turn including content with one of 8 harmful categories (e.g.,
    violence) and on one of 5 grades, from very mild to severe. On 160 dialogues,
    5 human raters confirmed reliable unsafe detection and a meaningful severity scale.
    We benchmark three open-source LALMs: Qwen2-Audio, Audio Flamingo 3, and MERaLiON
    as zero-shot judges that output a scalar safety score in [0,1] across audio-only,
    transcription-only, or multimodal inputs, along with a transcription-only LLaMA
    baseline. We measure the judges'' sensitivity to detecting unsafe content, the
    specificity in ordering severity levels, and the stability of the score in dialogue
    turns. Results reveal architecture- and modality-dependent trade-offs: the most
    sensitive judge is also the least stable across turns, while stable configurations
    sacrifice detection of mild harmful content. Transcription quality is a key bottleneck:
    Whisper-Large may significantly reduce sensitivity for transcription-only modes,
    while largely preserving severity ordering. Audio becomes crucial when paralinguistic
    cues or transcription fidelity are category-critical. We summarize all findings
    and provide actionable guidance for practitioners.'
  - 'FutureX-Pro: Extending Future Prediction to High-Value Vertical Domains. Building
    upon FutureX, which established a live benchmark for general-purpose future prediction,
    this report introduces FutureX-Pro, including FutureX-Finance, FutureX-Retail,
    FutureX-PublicHealth, FutureX-NaturalDisaster, and FutureX-Search. These together
    form a specialized framework extending agentic future prediction to high-value
    vertical domains. While generalist agents demonstrate proficiency in open-domain
    search, their reliability in capital-intensive and safety-critical sectors remains
    under-explored. FutureX-Pro targets four economically and socially pivotal verticals:
    Finance, Retail, Public Health, and Natural Disaster. We benchmark agentic Large
    Language Models (LLMs) on entry-level yet foundational prediction tasks -- ranging
    from forecasting market indicators and supply chain demands to tracking epidemic
    trends and natural disasters. By adapting the contamination-free, live-evaluation
    pipeline of FutureX, we assess whether current State-of-the-Art (SOTA) agentic
    LLMs possess the domain grounding necessary for industrial deployment. Our findings
    reveal the performance gap between generalist reasoning and the precision required
    for high-value vertical applications.'
  - A note on approximation in weighted Korobov spaces via multiple rank-1 lattices.
    This paper studies the multivariate approximation of functions in weighted Korobov
    spaces using multiple rank-1 lattice rules. It has been shown by Kämmerer and
    Volkmer (2019) that algorithms based on multiple rank-1 lattices achieve the optimal
    convergence rate for the $L_{\infty}$ error in Wiener-type spaces, up to logarithmic
    factors. While this result was translated to weighted Korobov spaces in the recent
    monograph by Dick, Kritzer, and Pillichshammer (2022), the analysis requires the
    smoothness parameter $α$ to be greater than $1$ and is restricted to product weights.
    In this paper, we extend this result for multiple rank-1 lattice-based algorithms
    to the case where $1/2<α\le 1$ and for general weights, covering a broader range
    of periodic functions with low smoothness and general relative importance of variables.
    We also provide a summability condition on the weights to ensure strong polynomial
    tractability for any $α>1/2$. Furthermore, by incorporating random shifts into
    multiple rank-1 lattice-based algorithms, we prove that the resulting randomized
    algorithm achieves a nearly optimal convergence rate in terms of the worst-case
    root mean squared $L_2$ error, while retaining the same tractability property.
- source_sentence: 'Bringing the economics of biodiversity into policy and decision-making:
    A target and cost-based approach to pricing biodiversity. Given ongoing, human-induced,
    loss of wild species we propose the Target and Cost Analysis (TCA) approach as
    a means of incorporating biodiversity within government appraisals of public spending.
    Influenced by how carbon is priced in countries around the world, the resulting
    biodiversity shadow price reflects the marginal cost of meeting government targets
    while avoiding disagreements on the use of willingness to pay measures to value
    biodiversity. Examples of how to operationalize TCA are developed at different
    scales and for alternative biodiversity metrics, including extinction risk for
    Europe and species richness in the UK. Pricing biodiversity according to agreed
    targets allows trade-offs with other wellbeing-enhancing uses of public funds
    to be sensibly undertaken without jeopardizing those targets, and is compatible
    with international guidelines on Cost Benefit Analysis.'
  sentences:
  - 'A converse of Berndtsson''s theorem on the positivity of direct images. Berndtsson''s
    famous theorem asserts that, for a compact Kähler fibration $p:X\to Y$, the direct
    image bundle $p_*(K_{X/Y}\otimes L)$ of a semi-positive Hermitian holomorphic
    line bundle $L\to X$ is Nakano semi-positive. As a continuation of our previous
    work, we prove a converse of Berndtsson''s theorem in the case of a projective
    fibration: if $p_*(K_{X/Y}\otimes L\otimes E)$ is Griffiths semi-positive for
    every semi-positive Hermitian holomorphic line bundle $E\to X$, then the curvature
    of $L$ must be semi-positive.'
  - 'NoLBERT: A No Lookahead(back) Foundational Language Model. We present NoLBERT,
    a lightweight, timestamped foundational language model for empirical research
    -- particularly for forecasting in economics, finance, and the social sciences.
    By pretraining exclusively on text from 1976 to 1995, NoLBERT avoids both lookback
    and lookahead biases (information leakage) that can undermine econometric inference.
    It exceeds domain-specific baselines on NLP benchmarks while maintaining temporal
    consistency. Applied to patent texts, NoLBERT enables the construction of firm-level
    innovation networks and shows that gains in innovation centrality predict higher
    long-run profit growth.'
  - 'JWST Advanced Deep Extragalactic Survey (JADES) Data Release 5: MIRI Coordinated
    Parallels in GOODS-S and GOODS-N. Medium to ultra-deep mid-infrared imaging surveys
    with the James Webb Space Telescope (JWST)''s Mid-Infrared Instrument (MIRI) are
    reframing our view of the early Universe, from the emergence of ultra-red dusty
    and quiescent galaxies to the epoch of reionization to the first galaxies. Here
    we present the MIRI coordinated parallels component of the JADES program, which
    obtained ultra-deep (155 ks) imaging at $7.7 μ$m over $\sim10$ arcmin$^2$ as well
    as medium depth ($\sim5-15$ ks) imaging at $7.7, 12.8$, and $15 μ$m over $\sim36$,
    25, and 22 arcmin$^2$, respectively, in the GOODS-S and GOODS-N fields. This paper
    describes the data reduction, which combines the official JWST Calibration Pipeline
    with custom steps to optimize flagging of warm/hot pixels and optimize background
    subtraction. We further introduce a new step to address artifacts caused by persistence
    from saturating sources. The final, fully reduced JADES/MIRI mosaics are being
    released as part of JADES Data Release 5, along with prior-based forced photometry
    using NIRCam detection images, providing critical rest-frame near-infrared and
    optical constraints on early galaxy populations.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on BAAI/bge-base-en-v1.5
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: arxiv val
      type: arxiv-val
    metrics:
    - type: cosine_accuracy
      value: 0.9402507543563843
      name: Cosine Accuracy
---

# SentenceTransformer based on BAAI/bge-base-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Bringing the economics of biodiversity into policy and decision-making: A target and cost-based approach to pricing biodiversity. Given ongoing, human-induced, loss of wild species we propose the Target and Cost Analysis (TCA) approach as a means of incorporating biodiversity within government appraisals of public spending. Influenced by how carbon is priced in countries around the world, the resulting biodiversity shadow price reflects the marginal cost of meeting government targets while avoiding disagreements on the use of willingness to pay measures to value biodiversity. Examples of how to operationalize TCA are developed at different scales and for alternative biodiversity metrics, including extinction risk for Europe and species richness in the UK. Pricing biodiversity according to agreed targets allows trade-offs with other wellbeing-enhancing uses of public funds to be sensibly undertaken without jeopardizing those targets, and is compatible with international guidelines on Cost Benefit Analysis.',
    'NoLBERT: A No Lookahead(back) Foundational Language Model. We present NoLBERT, a lightweight, timestamped foundational language model for empirical research -- particularly for forecasting in economics, finance, and the social sciences. By pretraining exclusively on text from 1976 to 1995, NoLBERT avoids both lookback and lookahead biases (information leakage) that can undermine econometric inference. It exceeds domain-specific baselines on NLP benchmarks while maintaining temporal consistency. Applied to patent texts, NoLBERT enables the construction of firm-level innovation networks and shows that gains in innovation centrality predict higher long-run profit growth.',
    "JWST Advanced Deep Extragalactic Survey (JADES) Data Release 5: MIRI Coordinated Parallels in GOODS-S and GOODS-N. Medium to ultra-deep mid-infrared imaging surveys with the James Webb Space Telescope (JWST)'s Mid-Infrared Instrument (MIRI) are reframing our view of the early Universe, from the emergence of ultra-red dusty and quiescent galaxies to the epoch of reionization to the first galaxies. Here we present the MIRI coordinated parallels component of the JADES program, which obtained ultra-deep (155 ks) imaging at $7.7 μ$m over $\\sim10$ arcmin$^2$ as well as medium depth ($\\sim5-15$ ks) imaging at $7.7, 12.8$, and $15 μ$m over $\\sim36$, 25, and 22 arcmin$^2$, respectively, in the GOODS-S and GOODS-N fields. This paper describes the data reduction, which combines the official JWST Calibration Pipeline with custom steps to optimize flagging of warm/hot pixels and optimize background subtraction. We further introduce a new step to address artifacts caused by persistence from saturating sources. The final, fully reduced JADES/MIRI mosaics are being released as part of JADES Data Release 5, along with prior-based forced photometry using NIRCam detection images, providing critical rest-frame near-infrared and optical constraints on early galaxy populations.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8976, 0.7760],
#         [0.8976, 1.0000, 0.7545],
#         [0.7760, 0.7545, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Dataset: `arxiv-val`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| **cosine_accuracy** | **0.9403** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 77,578 training samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>score</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                            | sentence2                                                                            | score                                                          |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               | float                                                          |
  | details | <ul><li>min: 28 tokens</li><li>mean: 248.32 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 33 tokens</li><li>mean: 246.24 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.43</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | sentence2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | score                            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code>Continuous approximate roots of polynomial equations via shape theory. We study continuous approximate solutions to polynomial equations over the ring $C(X)$ of continuous complex-valued functions over a compact Hausdorff space $X$. We show that when $X$ is one-dimensional, the existence of such approximate solutions is governed by the behaviour of maps from the fundamental pro-group of $X$ into braid groups.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | <code>Ordinary abelian varieties: isogeny graphs and polarizations. Given an integer $D$ and an ordinary isogeny class of abelian varieties defined over a finite field $\mathbb{F}_q$ with commutative $\mathbb{F}_q$-endomorphism algebra, we provide algorithms for computing all isogenies of degree dividing $D$ and polarizations of degree dividing $D$. We discuss phenomena that arise for higher dimension abelian varieties but not elliptic curves, bounds on the diameter of the graph of minimal isogenies, and decompositions of isogeny graphs into orbits for the Picard group of the Frobenius order.</code>                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>0.33499999999999996</code> |
  | <code>On damage of interpolation to adversarial robustness in regression. Deep neural networks (DNNs) typically involve a large number of parameters and are trained to achieve zero or near-zero training error. Despite such interpolation, they often exhibit strong generalization performance on unseen data, a phenomenon that has motivated extensive theoretical investigations. Comforting results show that interpolation indeed may not affect the minimax rate of convergence under the squared error loss. In the mean time, DNNs are well known to be highly vulnerable to adversarial perturbations in future inputs. A natural question then arises: Can interpolation also escape from suboptimal performance under a future $X$-attack? In this paper, we investigate the adversarial robustness of interpolating estimators in a framework of nonparametric regression. A finding is that interpolating estimators must be suboptimal even under a subtle future $X$-attack, and achieving perfect fitting can substantiall...</code> | <code>NeuralFLoC: Neural Flow-Based Joint Registration and Clustering of Functional Data. Clustering functional data in the presence of phase variation is challenging, as temporal misalignment can obscure intrinsic shape differences and degrade clustering performance. Most existing approaches treat registration and clustering as separate tasks or rely on restrictive parametric assumptions. We present \textbf{NeuralFLoC}, a fully unsupervised, end-to-end deep learning framework for joint functional registration and clustering based on Neural ODE-driven diffeomorphic flows and spectral clustering. The proposed model learns smooth, invertible warping functions and cluster-specific templates simultaneously, effectively disentangling phase and amplitude variation. We establish universal approximation guarantees and asymptotic consistency for the proposed framework. Experiments on functional benchmarks show state-of-the-art performance in both registration and clustering, with robustness to missin...</code> | <code>0.75</code>                |
  | <code>Hyperbolicity and fundamental groups of complex quasi-projective varieties (II): via non-abelian Hodge theories. This is Part II of a series of three papers. We studies the hyperbolicity of complex quasi-projective varieties $X$ in the presence of a big and reductive representation $\varrho: π_1(X)\to {\rm GL}_N(\mathbb{C})$. For any Galois conjugate variety $X^σ$ with $σ\in {\rm Aut}(\mathbb{C}/\mathbb{Q})$, we prove the generalized Green-Griffiths-Lang conjecture. When $\varrho$ is furthermore large, we show that the special subsets of $X^σ$ describing the non-hyperbolicity locus coincide, and that this locus is proper exactly when $X$ is of log general type. Moreover, if the Zariski closure of $ρ(π_1(X))$ is semisimple, we prove that there exists a proper Zariski closed subset $Z \subsetneq X^σ$ such that every subvariety not contained in $Z$ is of log general type and all entire curves in $X^σ$ are contained in $Z$. This result extends the theorems of the third author (2010) and of...</code> | <code>Heights on toric varieties for singular metrics: Local theory. We show that the (toric) local height of a toric variety with respect to a semipositive torus-invariant singular metric is given by the integral of a concave function over a compact convex set. This generalizes a result of Burgos, Philippon, and Sombra for the case of continuous metrics and answers a question raised by Burgos, Kramer, and Kühn in 2016.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | <code>0.5841666666666666</code>  |
* Loss: [<code>CoSENTLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "pairwise_cos_sim"
  }
  ```

### Evaluation Dataset

#### Unnamed Dataset

* Size: 16,247 evaluation samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>score</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                            | sentence2                                                                            | score                                                          |
  |:--------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                               | string                                                                               | float                                                          |
  | details | <ul><li>min: 28 tokens</li><li>mean: 251.73 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 28 tokens</li><li>mean: 251.73 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.42</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | score                            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
  | <code>Optimal illness policy for an unethical daycare center. While businesses are typically more profitable if their workers and communities are minimally exposed to diseases, the same is not true for daycare centers. Here it is shown that a daycare center could maximize its profits by maintaining a population of sick children within the center, with the intention to infect more children who then do not attend. Through a modification of the Susceptible-Infected-Recovered (SIR) model for disease spread we find the optimal number of sick children who should be kept within the center to maximize profits. We show that as disease infectiousness increases, the optimal attendance rate of sick children approaches zero, while the potential profit increases.</code>                                                                                                                                                                                                                                              | <code>Neutrino production mechanisms in strongly magnetized quark matter: Current status and open questions. We review the main neutrino emission mechanisms operating in dense quark matter under strong magnetic fields, with particular emphasis on conditions expected in the interiors of compact stars. We discuss the direct Urca and neutrino synchrotron processes in unpaired quark matter, incorporating the effects of Landau-level quantization. For the direct Urca process, the quantization of the electron energy spectrum plays a critical role, whereas quark quantization can often be neglected at sufficiently high baryon densities. The resulting field-dependent neutrino emissivity is anisotropic and exhibits an oscillatory behavior as a function of magnetic-field strength. We explore the implications of these effects for magnetar cooling and for possible anisotropic neutrino emission that could contribute to pulsar kicks. In addition, we review the $ν\barν$ synchrotron emission process, which, a...</code> | <code>0.0</code>                 |
  | <code>Polygon Containment and Translational Min-Hausdorff-Distance between Segment Sets are 3SUM-Hard. The 3SUM problem represents a class of problems conjectured to require $Ω(n^2)$ time to solve, where $n$ is the size of the input. Given two polygons $P$ and $Q$ in the plane, we show that some variants of the decision problem, whether there exists a transformation of $P$ that makes it contained in $Q$, are 3SUM-Hard. In the first variant $P$ and $Q$ are any simple polygons and the allowed transformations are translations only; in the second and third variants both polygons are convex and we allow either rotations only or any rigid motion. We also show that finding the translation in the plane that minimizes the Hausdorff distance between two segment sets is 3SUM-Hard.</code>                                                                                                                                                                                                                         | <code>Rigid-Invariant Sliced Wasserstein via Independent Embeddings. Comparing probability measures when their supports are related by an unknown rigid transformation is an important challenge in geometric data analysis, arising in shape matching and machine learning. Classical optimal transport (OT) distances, including Wasserstein and sliced Wasserstein, are sensitive to rotations and reflections, while Gromov-Wasserstein (GW) is invariant to isometries but computationally prohibitive for large datasets. We introduce \emph{Rigid-Invariant Sliced Wasserstein via Independent Embeddings} (RISWIE), a scalable pseudometric that combines the invariance of NP-hard approaches with the efficiency of projection-based OT. RISWIE utilizes data-adaptive bases and matches optimal signed permutations along axes according to distributional similarity to achieve rigid invariance with near-linear complexity in the sample size. We prove bounds relating RISWIE to GW in special cases and empirically demonstrat...</code> | <code>1.0</code>                 |
  | <code>Feasible constructivism. Dummett's argument for intuitionism is well known. There is a concern that the argument proves too much, specifically, that it supports the extreme and apparently incoherent position of strict finitism. The central question is how to explicate the notion that it is possible in practice to construct an arithmetical term or verify a statement. The strict finitist answer is plagued by the sorites paradox. We propose and develop feasibilism as a more plausible view, where computational feasibility, as captured by the class of polynomial-time problems, yields a robust and expedient explication of "possible in practice". In this approach, the complexity is bounded by a polynomial function of the input size, rather than bounded by a constant (as in strict finitism), thus resolving the sorites issues. We show that a system of strictly bounded arithmetic, introduced by Sam Buss, precisely formalizes the feasibilist view so as to satisfy Dummett's requirements.</code> | <code>A fully diagonalized spectral method on the unit ball. Our main objective in this work is to show how Sobolev orthogonal polynomials emerge as a useful tool within the framework of spectral methods for boundary-value problems. The solution of a boundary-value problem for a stationary Schrödinger equation on the unit ball can be studied from a variational perspective. In this variational formulation, a Sobolev inner product naturally arises. As test functions, we consider the linear space of the polynomials satisfying the boundary conditions on the sphere, and a basis of mutually orthogonal polynomials with respect to the Sobolev inner product is provided. The basis of the proposed method is given in terms of spherical harmonics and univariate Sobolev orthogonal polynomials. The connection formula between these Sobolev orthogonal polynomials and the classical orthogonal polynomials on the ball is established. Consequently, the Sobolev Fourier coefficients of a function satisfying the bo...</code> | <code>0.33499999999999996</code> |
* Loss: [<code>CoSENTLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "pairwise_cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `learning_rate`: 1e-05
- `weight_decay`: 0.01
- `num_train_epochs`: 1
- `warmup_ratio`: 0.1
- `warmup_steps`: 0.1
- `fp16`: True
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 1e-05
- `weight_decay`: 0.01
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.1
- `warmup_steps`: 0.1
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch      | Step     | Training Loss | Validation Loss | arxiv-val_cosine_accuracy |
|:----------:|:--------:|:-------------:|:---------------:|:-------------------------:|
| -1         | -1       | -             | -               | 0.9115                    |
| 0.0412     | 50       | 8.0178        | -               | -                         |
| 0.0824     | 100      | 7.5333        | 7.3744          | 0.9034                    |
| 0.1237     | 150      | 7.3440        | -               | -                         |
| 0.1649     | 200      | 7.2780        | 7.2430          | 0.9282                    |
| 0.2061     | 250      | 7.2585        | -               | -                         |
| 0.2473     | 300      | 7.2108        | 7.2278          | 0.9339                    |
| 0.2885     | 350      | 7.2187        | -               | -                         |
| 0.3298     | 400      | 7.2111        | 7.2124          | 0.9355                    |
| 0.3710     | 450      | 7.2271        | -               | -                         |
| 0.4122     | 500      | 7.2050        | 7.1927          | 0.9382                    |
| 0.4534     | 550      | 7.1821        | -               | -                         |
| 0.4946     | 600      | 7.2082        | 7.2076          | 0.9394                    |
| 0.5359     | 650      | 7.1989        | -               | -                         |
| 0.5771     | 700      | 7.1813        | 7.1919          | 0.9400                    |
| 0.6183     | 750      | 7.1879        | -               | -                         |
| 0.6595     | 800      | 7.1708        | 7.1963          | 0.9403                    |
| 0.7007     | 850      | 7.1563        | -               | -                         |
| 0.7420     | 900      | 7.1453        | 7.1956          | 0.9410                    |
| 0.7832     | 950      | 7.1500        | -               | -                         |
| 0.8244     | 1000     | 7.1624        | 7.1953          | 0.9399                    |
| 0.8656     | 1050     | 7.1738        | -               | -                         |
| **0.9068** | **1100** | **7.1821**    | **7.1906**      | **0.9403**                |
| 0.9481     | 1150     | 7.1698        | -               | -                         |
| 0.9893     | 1200     | 7.1543        | 7.1911          | 0.9408                    |
| -1         | -1       | -             | -               | 0.9403                    |

* The bold row denotes the saved checkpoint.

### Framework Versions
- Python: 3.12.12
- Sentence Transformers: 5.2.2
- Transformers: 5.0.0
- PyTorch: 2.9.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### CoSENTLoss
```bibtex
@article{10531646,
    author={Huang, Xiang and Peng, Hao and Zou, Dongcheng and Liu, Zhiwei and Li, Jianxin and Liu, Kay and Wu, Jia and Su, Jianlin and Yu, Philip S.},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    title={CoSENT: Consistent Sentence Embedding via Similarity Ranking},
    year={2024},
    doi={10.1109/TASLP.2024.3402087}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->