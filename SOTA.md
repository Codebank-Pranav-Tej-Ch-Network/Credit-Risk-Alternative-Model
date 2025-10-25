State-of-the-Art in PII/PHI Detection for Handwritten Medical Documents: A Comprehensive Review1. Overview and ImperativeThis report consolidates a comprehensive review of existing datasets, methods, and performance metrics for the detection of Personally Identifiable Information (PII) and Protected Health Information (PHI) in handwritten medical documents. The digitization of healthcare has led to a massive volume of unstructured data, including scanned prescriptions and clinical notes. While invaluable for research and patient care, this data is rich with sensitive information. Manual redaction is unscalable and error-prone, making automated, intelligent privacy region detection a critical enabling technology for secure data sharing and regulatory compliance.This document synthesizes findings from numerous studies, centered around a complete table of datasets and model performances. It further contextualizes these findings by defining the core concepts of PII/PHI, detailing evaluation methodologies, analyzing state-of-the-art approaches, and identifying the critical research gaps that currently limit progress in achieving true end-to-end de-identification for handwritten medical records.2. Foundational Concepts: Defining the Targets of DetectionEffective detection requires a precise definition of its targets, which are grounded in legal and regulatory frameworks like the Health Insurance Portability and Accountability Act (HIPAA).Personally Identifiable Information (PII): PII is any information that can distinguish or trace an individual's identity, either alone (direct identifiers like a Social Security Number) or when combined with other data (indirect identifiers like ZIP code and date of birth). The scope of PII is constantly evolving as new technologies make it easier to link seemingly anonymous data points to a specific person.Protected Health Information (PHI): PHI is a subset of PII specific to the healthcare context. Under HIPAA, PHI is any individually identifiable health information held or transmitted by a healthcare provider or its associates. PII becomes PHI when it is linked to the provision of healthcare. HIPAA explicitly lists 18 identifiers (such as names, dates, medical record numbers, and biometric data) that must be removed for data to be considered de-identified under the "Safe Harbor" method. The accurate detection of these 18 identifiers is the primary goal of most medical de-identification systems.3. A Framework for Evaluation: Metrics and MethodologiesEvaluating the performance of PII/PHI detection systems requires a rigorous quantitative framework, borrowing metrics from the machine learning fields of Named Entity Recognition (NER) for text and Object Detection for images.Core Metrics:Precision: Measures the accuracy of the predictions. It answers: "Of all the regions the model identified as private, what fraction was correct?" High precision minimizes over-redaction, preserving data utility.Recall (Sensitivity): Measures the completeness of the detection. It answers: "Of all the actual private regions, what fraction did the model find?" High recall is paramount for privacy, as it minimizes data leaks.F1-Score: The harmonic mean of Precision and Recall, providing a single score that balances both concerns.Metrics for Visual Detection:Intersection over Union (IoU): For image-based tasks, IoU quantifies the overlap between a predicted bounding box and the ground-truth box. A detection is typically considered a "True Positive" if the IoU exceeds a set threshold (e.g., 0.5).mean Average Precision (mAP): The standard metric for object detection models, mAP averages the precision across all recall values and across all classes, providing a comprehensive measure of a model's performance.The Asymmetric Cost of Errors: A critical observation is the imbalanced cost of errors. A False Negative (missing a piece of PHI) can lead to a privacy breach and severe regulatory penalties. A False Positive (incorrectly redacting benign information) reduces data utility but does not compromise privacy. Therefore, systems in this domain must be optimized with an explicit prioritization of Recall to ensure maximum safety and compliance.4. Consolidated Review of Datasets and Model PerformanceThe following table consolidates all datasets, models, performance metrics, and citations from the comprehensive review of PII/PHI detection in handwritten medical documents. Each row represents either a dataset with its usage or a standalone dataset available for research.Serial No.Training and Testing DatasetML Model UsedMetrics for EvaluationLink to Relevant PapersOther CitationsAccess to Dataset1i2b2 2014 De-identification Dataset 
1,304 longitudinal medical records (296 patients) 
Training: 790 records (188 patients, 17,045 PHI instances) 
Testing: 514 records (109 patients, 11,462 PHI instances) 
7 major PHI categories, 25 subcategoriesVarious NER models in i2b2 challenge• Mean F1-score 
• Maximum F1-score 
• Entity-based micro-averaged evaluationi2b2 2014 ChallengeStubbs, A., Kotfila, C., & Uzuner, O. (2015). Journal of Biomedical Informatics, 58, S11-S19🔒 Data use agreement required 
HuggingFace2i2b2 2014 De-identification DatasetSpark NLP Healthcare• F1-score on PHI categories 
• F1-score on granular categories 
• Comparative performance evaluation(https://www.sciencedirect.com/science/article/pii/S2665963822000793)Kocaman, V., & Talby, D. (2022). Accurate Clinical and Biomedical Named Entity Recognition at Scale🔒 Same as above3i2b2 2014 De-identification DatasetBiLSTM-CNN-CRF hybrid system• F1-score (token) 
• F1-score (strict) 
• F1-score (binary token) 
• CEGS N-GRID ranking(https://www.sciencedirect.com/science/article/pii/S1532046417301223)Liu, Z., et al. (2017). ScienceDirect🔒 Same as above4i2b2 2014 De-identification DatasetXLNet• F1-score on PHI recognition 
• Entity performance comparison 
• Performance improvement over BERT(https://e-hir.org/upload/pdf/hir-2022-28-1-16.pdf)Healthcare Informatics Research (2022)🔒 Same as above5i2b2 2006 De-identification Dataset 
889 discharge summaries 
19,498 PHI instances 
Partners Healthcare SystemClinical ModernBERT• F1-score on i2b2-2006 
• F1-score on i2b2-2012 
• Token sequence support evaluation(https://pmc.ncbi.nlm.nih.gov/articles/PMC11230489/)PMC11230489 (2024)🔒 Available through n2c2 portal6IAM Handwriting Database 
13,353 handwritten line images 
657 writers, 1,539 pages 
115,320 words, 5,685 sentences 
Lancaster-Oslo/Bergen Corpus-• Sentence-level annotations 
• Line-level annotations 
• Word-level annotations 
• Image quality metricsMany placesMarti, U.V., & Bunke, H. (2002). ICDAR 1999✅ Publicly available 
HuggingFace7IAM On-Line Handwriting Database 
Handwritten English text on whiteboard-• Writer identification metrics 
• Text recognition accuracy 
• Online handwriting evaluationHave to confirm availabilityLiBu05-03✅ Publicly available8Pakistani Medical Prescriptions Dataset 
Diverse handwritten prescriptions 
Various regions of PakistanTrOCR + Mask R-CNN 
TrOCR-Base-Handwritten 
Multi-Head Attention 
Positional Embeddings• Character Error Rate (CER) 
• Standard benchmark evaluation 
• Medicine name extraction accuracy(https://arxiv.org/html/2412.18199v1)Ahmad, M., et al. (2024). IEEE Conference Publication❓ Availability unclear 
Research dataset9Bangladeshi Prescription Dataset 
17,431 prescription terms 
Poor legibility challengesTrOCR• Character recognition accuracy 
• Medicine name extraction 
• Legibility assessment(https://www.researchgate.net/publication/392949112_Recognizing_Medicine_Names_from_Bangladeshi_Handwritten_Prescription_Images_using_TrOCR)ResearchGate (2024)❓ Research use only10Medical-SYN Dataset 
Synthetic medical prescriptions 
English + Urdu textViLanOCR 
Multilingual transformer model• Character Error Rate (CER) 
• Cross-lingual evaluation 
• Multilingual performance assessment(https://pmc.ncbi.nlm.nih.gov/articles/PMC11065407/)Khan, A., et al. (2024). PMC11065407❓ Synthetic dataset 
Research use11Tobacco-800 Dataset 
878 documents with stamps/signatures 
412 with logos, 878 without 
150-300 DPI resolutionFaster R-CNN 
VGG-16 backbone 
ResNet-50 backbone• mean Average Precision (mAP) 
• Logo detection accuracy 
• Signature detection performance 
• Intersection over Union (IoU)Multiple object detection papersTobacco industry documents research✅ Publicly available 
Research dataset12SignverOD Dataset 
8,022 labeled signatures, dates, redactions 
Bank documents and memosObject detection models• Signature verification accuracy 
• Date detection performance 
• Redaction identification metricsTechnical review citationsObject detection research papers✅ Publicly available13SPODS Dataset 
400 scanned documents 
Stamps and signatures annotated 
Pseudo-official documentsObject detection models• mean Average Precision (mAP) 
• Stamp detection evaluation 
• Signature recognition assessmentTechnical review citationsDocument analysis research✅ Publicly available14StaVer (Stamp Verification) Dataset 
Stamp verification focusedYOLOv8-v11 
Multiple YOLO variants• mean Average Precision (mAP) 
• Real-time processing metrics 
• Precision evaluation 
• Frames per second (FPS)Technical review citationsYOLO stamp detection research✅ Publicly available15SynthDoG-Generated Datasets 
Synthetic documents 
4 languages (EN, CN, JP, KR) 
0.5M documents per languageDonut (Document Understanding Transformer) 
Swin Transformer encoder 
BART decoder• Language-agnostic processing evaluation 
• Document parsing accuracy 
• Cross-lingual performance metrics(https://arxiv.org/abs/2111.15664)Kim, G., et al. (2022). ECCV 2022✅ Available with Donut 
GitHub16CORD Dataset 
Receipt parsing dataset 
Used for Donut evaluationDonut 
OCR-free end-to-end 
Vision Transformer + BART• Processing time per document 
• Document parsing accuracy 
• Normalized Tree Edit Distance (nTED) 
• Comparison with OCR pipelines(https://github.com/clovaai/donut)Multiple Donut implementations✅ Available via HuggingFace17Train Ticket Dataset 
Chinese train ticket parsingDonut 
Fine-tuned for ticket parsing• Processing time per document 
• Parsing accuracy evaluation 
• Real-time processing capability(https://github.com/WalysonGO/donut-ocr)Donut research implementations✅ Available with Donut demos18RVL-CDIP Dataset 
Document classification 
16 document categoriesDonut 
Document classification variant• Processing time per document 
• Classification accuracy 
• OCR-free processing evaluation(https://github.com/santoshvutukuri/donut-no-OCR)Document classification research✅ Publicly available19PhysioNet Clinical Datasets 
Various clinical text datasets 
Ages <89 not treated as PHIVarious clinical NLP models• Clinical entity extraction metrics 
• De-identification performance. I don't think the data is available publicly now. 
• Domain-specific evaluation criteriaPhysioNetMultiple clinical NLP papers🔒 Credentialed access 
Research use20Clinical Receipt Text Segments 
15,297 clinical receipt segments 
Mixed printed/handwrittenResNet-101T 
Transformer decoder
