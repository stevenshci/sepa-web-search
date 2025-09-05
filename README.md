# SePA Web-Search Pipeline

Official implementation of the-aware web-search pipeline from "SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching" (IEEE BHI 2025).

## Code Release Status
**Full code will be released by September 26, 2025**

We are currently preparing the code for public release, including:
- Cleaning and documentation
- Removing any sensitive data/credentials  
- Adding usage examples
- Testing installation procedures

## Abstract
This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group k-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff’s $\delta = 0.3$, *p* = 0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems.
