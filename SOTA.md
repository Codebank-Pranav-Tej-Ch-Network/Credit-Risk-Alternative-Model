# State-of-the-Art in PII/PHI Detection for Handwritten Medical Documents: A Comprehensive Review

## 1. Overview and Imperative

This report consolidates a comprehensive review of existing datasets, methods, and performance metrics for the detection of Personally Identifiable Information (PII) and Protected Health Information (PHI) in handwritten medical documents. The digitization of healthcare has led to a massive volume of unstructured data, including scanned prescriptions and clinical notes. While invaluable for research and patient care, this data is rich with sensitive information. Manual redaction is unscalable and error-prone, making automated, intelligent privacy region detection a critical enabling technology for secure data sharing and regulatory compliance.

This document synthesizes findings from numerous studies, centered around a complete table of datasets and model performances. It further contextualizes these findings by defining the core concepts of PII/PHI, detailing evaluation methodologies, analyzing state-of-the-art approaches, and identifying the critical research gaps that currently limit progress in achieving true end-to-end de-identification for handwritten medical records.

## 2. Foundational Concepts: Defining the Targets of Detection

Effective detection requires a precise definition of its targets, which are grounded in legal and regulatory frameworks like the Health Insurance Portability and Accountability Act (HIPAA).

**Personally Identifiable Information (PII):** PII is any information that can distinguish or trace an individual's identity, either alone (direct identifiers like a Social Security Number) or when combined with other data (indirect identifiers like ZIP code and date of birth). The scope of PII is constantly evolving as new technologies make it easier to link seemingly anonymous data points to a specific person.

**Protected Health Information (PHI):** PHI is a subset of PII specific to the healthcare context. Under HIPAA, PHI is any individually identifiable health information held or transmitted by a healthcare provider or its associates. PII becomes PHI when it is linked to the provision of healthcare. HIPAA explicitly lists 18 identifiers (such as names, dates, medical record numbers, and biometric data) that must be removed for data to be considered de-identified under the "Safe Harbor" method. The accurate detection of these 18 identifiers is the primary goal of most medical de-identification systems.

## 3. A Framework for Evaluation: Metrics and Methodologies

Evaluating the performance of PII/PHI detection systems requires a rigorous quantitative framework, borrowing metrics from the machine learning fields of Named Entity Recognition (NER) for text and Object Detection for images.

**Core Metrics:**

* **Precision:** Measures the accuracy of the predictions. It answers: "Of all the regions the model identified as private, what fraction was correct?" High precision minimizes over-redaction, preserving data utility.
* **Recall (Sensitivity):** Measures the completeness of the detection. It answers: "Of all the actual private regions, what fraction did the model find?" High recall is paramount for privacy, as it minimizes data leaks.
* **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both concerns.

**Metrics for Visual Detection:**

* **Intersection over Union (IoU):** For image-based tasks, IoU quantifies the overlap between a predicted bounding box and the ground-truth box. A detection is typically considered a "True Positive" if the IoU exceeds a set threshold (e.g., 0.5).
* **mean Average Precision (mAP):** The standard metric for object detection models, mAP averages the precision across all recall values and across all classes, providing a comprehensive measure of a model's performance.

**The Asymmetric Cost of Errors:** A critical observation is the imbalanced cost of errors. A False Negative (missing a piece of PHI) can lead to a privacy breach and severe regulatory penalties. A False Positive (incorrectly redacting benign information) reduces data utility but does not compromise privacy. Therefore, systems in this domain must be optimized with an explicit prioritization of Recall to ensure maximum safety and compliance.

## 4. Consolidated Review of Datasets and Model Performance

The following table consolidates all datasets, models, performance metrics, and citations from the comprehensive review of PII/PHI detection in handwritten medical documents. Each row represents either a dataset with its usage or a standalone dataset available for research.

| Serial No. | Training and Testing Dataset | ML Model Used | Metrics for Evaluation | Link to Relevant Papers | Other Citations | Access to Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | i2b2 2014 De-identification Dataset <br> • 1,304 longitudinal medical records (296 patients) <br> • Training: 790 records (188 patients, 17,045 PHI instances) <br> • Testing: 514 records (109 patients, 11,462 PHI instances) <br> • 7 major PHI categories, 25 subcategories | Various NER models in i2b2 challenge | • Mean F1-score <br> • Maximum F1-score <br> • Entity-based micro-averaged evaluation | i2b2 2014 Challenge | Stubbs, A., Kotfila, C., & Uzuner, O. (2015). Journal of Biomedical Informatics, 58, S11-S19 | 🔒 Data use agreement required <br> HuggingFace |
| 2 | i2b2 2014 De-identification Dataset | Spark NLP Healthcare | • F1-score on PHI categories <br> • F1-score on granular categories <br> • Comparative performance evaluation | (https://www.sciencedirect.com/science/article/pii/S2665963822000793) | Kocaman, V., & Talby, D. (2022). Accurate Clinical and Biomedical Named Entity Recognition at Scale | 🔒 Same as above |
| 3 | i2b2 2014 De-identification Dataset | BiLSTM-CNN-CRF hybrid system | • F1-score (token) <br> • F1-score (strict) <br> • F1-score (binary token) <br> • CEGS N-GRID ranking | (https://www.sciencedirect.com/science/article/pii/S1532046417301223) | Liu, Z., et al. (2017). ScienceDirect | 🔒 Same as above |
| 4 | i2b2 2014 De-identification Dataset | XLNet | • F1-score on PHI recognition <br> • Entity performance comparison <br> • Performance improvement over BERT | (https://e-hir.org/upload/pdf/hir-2022-28-1-16.pdf) | Healthcare Informatics Research (2022) | 🔒 Same as above |
| 5 | i2b2 2006 De-identification Dataset <br> • 889 discharge summaries <br> • 19,498 PHI instances <br> • Partners Healthcare System | Clinical ModernBERT | • F1-score on i2b2-2006 <br> • F1-score on i2b2-2012 <br> • Token sequence support evaluation | (https://pmc.ncbi.nlm.nih.gov/articles/PMC11230489/) | PMC11230489 (2024) | 🔒 Available through n2c2 portal |
| 6 | IAM Handwriting Database <br> • 13,353 handwritten line images <br> • 657 writers, 1,539 pages <br> • 115,320 words, 5,685 sentences <br> • Lancaster-Oslo/Bergen Corpus | - | • Sentence-level annotations <br> • Line-level annotations <br> • Word-level annotations <br> • Image quality metrics | Many places | Marti, U.V., & Bunke, H. (2002). ICDAR 1999 | ✅ Publicly available <br> HuggingFace |
| 7 | IAM On-Line Handwriting Database <br> • Handwritten English text on whiteboard | - | • Writer identification metrics <br> • Text recognition accuracy <br> • Online handwriting evaluation | Have to confirm availability | LiBu05-03 | ✅ Publicly available |
| 8 | Pakistani Medical Prescriptions Dataset <br> • Diverse handwritten prescriptions <br> • Various regions of Pakistan | TrOCR + Mask R-CNN <br> TrOCR-Base-Handwritten <br> Multi-Head Attention <br> Positional Embeddings | • Character Error Rate (CER) <br> • Standard benchmark evaluation <br> • Medicine name extraction accuracy | (https://arxiv.org/html/2412.18199v1) | Ahmad, M., et al. (2024). IEEE Conference Publication | ❓ Availability unclear <br> Research dataset |
| 9 | Bangladeshi Prescription Dataset <br> • 17,431 prescription terms <br> • Poor legibility challenges | TrOCR | • Character recognition accuracy <br> • Medicine name extraction <br> • Legibility assessment | (https://www.researchgate.net/publication/392949112_Recognizing_Medicine_Names_from_Bangladeshi_Handwritten_Prescription_Images_using_TrOCR) | ResearchGate (2024) | ❓ Research use only |
| 10 | Medical-SYN Dataset <br> • Synthetic medical prescriptions <br> • English + Urdu text | ViLanOCR <br> Multilingual transformer model | • Character Error Rate (CER) <br> • Cross-lingual evaluation <br> • Multilingual performance assessment | (https://pmc.ncbi.nlm.nih.gov/articles/PMC11065407/) | Khan, A., et al. (2024). PMC11065407 | ❓ Synthetic dataset <br> Research use |
| 11 | Tobacco-800 Dataset <br> • 878 documents with stamps/signatures <br> • 412 with logos, 878 without <br> • 150-300 DPI resolution | Faster R-CNN <br> VGG-16 backbone <br> ResNet-50 backbone | • mean Average Precision (mAP) <br> • Logo detection accuracy <br> • Signature detection performance <br> • Intersection over Union (IoU) | Multiple object detection papers | Tobacco industry documents research | ✅ Publicly available <br> Research dataset |
| 12 | SignverOD Dataset <br> • 8,022 labeled signatures, dates, redactions <br> • Bank documents and memos | Object detection models | • Signature verification accuracy <br> • Date detection performance <br> • Redaction identification metrics | Technical review citations | Object detection research papers | ✅ Publicly available |
| 13 | SPODS Dataset <br> • 400 scanned documents <br> • Stamps and signatures annotated <br> • Pseudo-official documents | Object detection models | • mean Average Precision (mAP) <br> • Stamp detection evaluation <br> • Signature recognition assessment | Technical review citations | Document analysis research | ✅ Publicly available |
| 14 | StaVer (Stamp Verification) Dataset <br> • Stamp verification focused | YOLOv8-v11 <br> Multiple YOLO variants | • mean Average Precision (mAP) <br> • Real-time processing metrics <br> • Precision evaluation <br> • Frames per second (FPS) | Technical review citations | YOLO stamp detection research | ✅ Publicly available |
| 15 | SynthDoG-Generated Datasets <br> • Synthetic documents <br> • 4 languages (EN, CN, JP, KR) <br> • 0.5M documents per language | Donut (Document Understanding Transformer) <br> Swin Transformer encoder <br> BART decoder | • Language-agnostic processing evaluation <br> • Document parsing accuracy <br> • Cross-lingual performance metrics | (https://arxiv.org/abs/2111.15664) | Kim, G., et al. (2022). ECCV 2022 | ✅ Available with Donut <br> GitHub |
| 16 | CORD Dataset <br> • Receipt parsing dataset <br> • Used for Donut evaluation | Donut <br> OCR-free end-to-end <br> Vision Transformer + BART | • Processing time per document <br> • Document parsing accuracy <br> • Normalized Tree Edit Distance (nTED) <br> • Comparison with OCR pipelines | (https://github.com/clovaai/donut) | Multiple Donut implementations | ✅ Available via HuggingFace |
| 17 | Train Ticket Dataset <br> • Chinese train ticket parsing | Donut <br> Fine-tuned for ticket parsing | • Processing time per document <br> • Parsing accuracy evaluation <br> • Real-time processing capability | (https://github.com/WalysonGO/donut-ocr) | Donut research implementations | ✅ Available with Donut demos |
| 18 | RVL-CDIP Dataset <br> • Document classification <br> • 16 document categories | Donut <br> Document classification variant | • Processing time per document <br> • Classification accuracy <br> • OCR-free processing evaluation | (https://github.com/santoshvutukuri/donut-no-OCR) | Document classification research | ✅ Publicly available |
| 19 | PhysioNet Clinical Datasets <br> • Various clinical text datasets <br> • Ages <89 not treated as PHI | Various clinical NLP models | • Clinical entity extraction metrics <br> • De-identification performance. I don't think the data is available publicly now. <br> • Domain-specific evaluation criteria | PhysioNet | Multiple clinical NLP papers | 🔒 Credentialed access <br> Research use |
| 20 | Clinical Receipt Text Segments <br> • 15,297 clinical receipt segments <br> • Mixed printed/handwritten | ResNet-101T <br> Transformer decoder | • Character Error Rate (CER) <br> • Mixed text type handling <br> • Real medical document performance | Technical review citations | Medical OCR research | ❓ Research dataset |

## 5. Analysis of State-of-the-Art Methodologies

The approaches to PII/PHI detection have evolved significantly, moving from manual rules to sophisticated deep learning pipelines.

**Evolution from Rules to Transformers:** Early systems relied on rule-based methods (e.g., regular expressions) and classical machine learning. While effective for structured data like phone numbers, they lacked the flexibility for unstructured text. The state-of-the-art is now dominated by Transformer-based models like BERT and RoBERTa. These models excel at understanding context, leading to significant performance gains. Domain-specific models like ClinicalBERT, which are pre-trained on large corpora of medical text, demonstrate superior performance on clinical documents. For long documents that exceed the standard input limits of these models, architectures like Longformer and BigBird use sparse attention mechanisms to process thousands of tokens at once.

**The Challenge of Handwritten and Scanned Documents:** Since handwritten medical documents are processed as images, a multi-stage pipeline is required. The state-of-the-art approach combines three distinct AI models:

1.  **Object Detection:** A model like YOLO is used to first detect and localize regions of text within the document image.
2.  **Optical Character Recognition (OCR):** The identified text regions are then passed to an OCR engine, such as EasyOCR, which converts the image of the text into a machine-readable string.
3.  **Named Entity Recognition (NER):** Finally, the extracted text string is analyzed by a Transformer-based NER model to classify whether it contains PHI.

**The Rise of Generative Anonymization:** The newest frontier involves using Large Language Models (LLMs) not just for detection, but for generative anonymization. Instead of simply masking data (e.g., with `[***]`), these models can replace identified PHI with realistic but synthetic surrogate values (e.g., replacing "Jane Doe" with "Gina Smith"). This preserves the document's readability and semantic structure, which is crucial for downstream analysis.

## 6. Comparative Analysis: Insights from Other Domains

Analyzing privacy detection in other regulated domains like finance and law reveals converging trends and highlights the importance of domain-specific adaptation.

* **Financial Documents:** This domain uses a hybrid approach. Highly structured data like credit card numbers are detected with precise, rule-based methods, while unstructured PII like names and addresses are handled by deep learning NER models. The availability of high-quality, multilingual synthetic datasets has been crucial for training models without exposing real financial data.
* **Legal Documents:** The key challenge here is domain adaptation. The specialized vocabulary and syntax of legal text mean that general-purpose NER models perform very poorly. The SOTA approach involves fine-tuning Transformer models on corpora of annotated legal documents. A unique requirement is consistent pseudonymization, ensuring the same individual is replaced with the same pseudonym throughout a document to maintain narrative coherence.
* **General Document Redaction:** Commercial tools like Redactable and Adobe Acrobat Pro define the SOTA for general business use. They integrate AI-powered pattern detection, OCR for scanned documents, and critical security features like metadata scrubbing to ensure redaction is permanent and comprehensive.

## 7. Synthesis of Findings

### Legend and Access Information

**Access Types**

* ✅ **Publicly Available:** Open access, no restrictions
* 🔒 **Restricted Access:** Requires data use agreement, credentials, or institutional approval
* ❓ **Unclear/Research:** Availability unclear, likely limited to research collaborations

### Dataset Categories Summary

| Category | Count | Key Characteristics | Main Limitation |
| :--- | :--- | :--- | :--- |
| Clinical Text + PHI | 4 datasets | Comprehensive PHI annotations | ❌ Typed text only, no handwriting |
| Handwriting Only | 2 datasets | Large-scale handwriting samples | ❌ No medical content or PHI labels |
| Medical Handwriting | 3 datasets | Real medical prescriptions | ❌ No PII/PHI annotations |
| Visual Elements | 4 datasets | Stamps, signatures, logos | ❌ Non-medical domains |
| Synthetic/Mixed | 7 datasets | Artificial or processed data | ❌ Limited real-world applicability |

### Evaluation Metrics Summary

| Metric Type | Application | Common Usage |
| :--- | :--- | :--- |
| **F1-Score** | Text-based PII/PHI detection | Entity-level evaluation, micro/macro averaging |
| **Character Error Rate (CER)** | Handwriting recognition | Character-level accuracy assessment |
| **Word Error Rate (WER)** | Text recognition | Word-level accuracy measurement |
| **mean Average Precision (mAP)**| Object detection | Visual element detection (stamps, signatures) |
| **Intersection over Union (IoU)**| Object localization | Bounding box accuracy for visual elements |
| **Processing Time** | System efficiency | Real-time capability assessment |
| **Normalized Tree Edit Distance (nTED)** | Structured output | Document parsing accuracy |

### Critical Research Gap

🚨 **No dataset combines all four requirements:**
* ✅ Handwritten content
* ✅ Medical domain
* ✅ PII/PHI annotations
* ✅ Sufficient scale for deep learning

This fundamental gap explains why all reported "handwritten medical PII detection" performance actually measures synthetic data, domain transfer, or visual-only detection rather than true end-to-end handwritten medical document de-identification.

### Key Findings

* **Dataset Fragmentation:** The field is characterized by datasets that excel in one dimension (handwriting, medical content, or PHI annotations) but lack integration across all three. This forces researchers to use proxy tasks or datasets from other domains.
* **The Rise of Synthetic Data:** A clear trend is the increasing reliance on high-fidelity synthetic data. Datasets like the MIDI Dataset for medical images solve the "chicken-and-egg" problem of needing private data to build privacy tools. By infusing clean images with synthetic PHI, they create a shareable, risk-free benchmark with a perfect ground truth.
* **Metric Diversity:** Different evaluation approaches (F1-score for text, CER for handwriting recognition, mAP for visual objects) make cross-study comparisons challenging, highlighting the need for standardized evaluation protocols for multi-stage pipelines.
* **Access Barriers:** Most high-quality medical datasets require institutional agreements and are not publicly available, which slows down research. In contrast, open datasets often lack direct medical relevance or PHI annotations.
* **Model Specialization:** Different model architectures are specialized for different tasks—Transformers for contextual text understanding, CNNs for visual element detection, and hybrid, multi-stage pipelines for complex, scanned documents. There is no one-size-fits-all solution.
* **Speed vs. Accuracy Trade-offs:** End-to-end models like Donut offer faster processing for document understanding tasks by avoiding a separate OCR step, but their evaluation can be complex. Traditional pipelines offer more modularity and interpretability at the cost of speed.

## 8. Open Challenges and Future Research Directions

Despite progress, several open challenges define the future of the field:

* **The Privacy-Utility Trade-off:** This is the central challenge. Aggressive redaction ensures privacy but can destroy the scientific value of the data. Future work must focus on utility-preserving techniques like generative anonymization and consistent pseudonymization that allow for longitudinal analysis without revealing identity.
* **Robustness and Generalization (Dataset Shift):** Models trained on a static dataset often fail when deployed in a real-world environment where document formats, sources, and content are constantly evolving. The future lies in an MLOps approach of "Privacy-as-a-Service," with continuous monitoring for data drift and automated re-training to ensure models remain robust over time.
* **Multimodal and Structurally Complex Documents:** Real-world documents are complex artifacts combining typed text, handwriting, tables, and images. Current siloed approaches are insufficient. The development of unified, multimodal models that can understand content and spatial layout holistically is a major research frontier.
* **Explainability and Trustworthiness:** As LLMs are increasingly used for generative anonymization, ensuring their reliability is critical. New evaluation frameworks are needed to verify that these "black box" models have removed all PHI without subtly altering clinical meaning.

## 9. Conclusion and Strategic Recommendations

The state-of-the-art in privacy region detection is defined by a move towards domain-adapted, AI-powered pipelines. For text, Transformer-based models are the standard, while for images, a multi-stage approach combining object detection, OCR, and NER is most effective. The lack of a comprehensive, publicly available dataset for handwritten medical documents with PHI annotations remains the single largest barrier to progress.

Based on this review, the following strategic recommendations are proposed:

* **Prioritize Domain-Specific Models:** Do not use general-purpose models for specialized domains. Invest in solutions that are pre-trained or fine-tuned on high-quality, in-domain data.
* **Optimize for Recall:** Given the severe consequences of a data leak, models must be tuned to maximize recall, ensuring all potential PHI is identified, even at the cost of some over-redaction.
* **Adopt a Pipeline Approach for Complex Documents:** For scanned or handwritten documents, use a modular pipeline that leverages the best available technology for each sub-task (text detection, recognition, and classification).
* **Invest in Synthetic Data Generation:** To overcome the primary research gap, efforts should be directed toward creating a large-scale, realistic, synthetic dataset that combines handwritten text, a medical context, and comprehensive PII/PHI annotations.
* **Plan for Long-Term Maintenance:** Treat de-identification as a continuous operational process, not a one-time project. Implement an MLOps framework to monitor for dataset shift and retrain models to maintain performance and compliance.
