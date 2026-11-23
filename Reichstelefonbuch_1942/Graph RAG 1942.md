# Graphical RAG
## Design Specification
### Historical German Reich (1942) Surname Knowledge Graph
Version: Draft 1.0

---

# Objective

Design a historical surname knowledge graph capable of supporting:

- Graph RAG
- Genealogical reasoning
- Historical administrative geography
- Name normalization
- OCR uncertainty
- Frequency weighting
- Provenance tracking
- Agentic AI ingestion pipelines

The graph becomes the authoritative dictionary for surname validation and historical context retrieval.

---

# Core Design Philosophy

Traditional RAG retrieves documents.

Graph RAG retrieves relationships.

Instead of asking

> "Does Hermann exist?"

the system asks

> "How is Hermann connected to this place, this district, this time period, and how common was it?"

This dramatically reduces hallucination while increasing historical confidence.

---

# Overall Architecture

```
                   OCR Sources
                        │
                        ▼
              Document Processing Agents
                        │
                        ▼
             Name Extraction Agents
                        │
                        ▼
           Name Normalization Agents
                        │
                        ▼
            Historical Validation Agents
                        │
                        ▼
                 Neo4j Knowledge Graph
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
    Graph RAG      Analytics      Interactive Dashboard
```

---

# Data Sources

The graph is designed to integrate multiple historical datasets.

Examples

- Reich Telephone Book (RTB)
- Einwohnerbuch
- Kirchenbücher
- Census
- Civil registrations
- Military records
- Immigration lists
- Gazetteers
- Historical maps
- OCR extracted documents

Each source becomes provenance.

---

# Graph Entity Model

## Geography

```
Country

    ↓

Province / Reichsgau

    ↓

Gau

    ↓

Kreis

    ↓

Ort

    ↓

Street (optional)

    ↓

Building (optional)
```

---

## Person Layer

```
Surname

GivenName

Person

Occupation

Religion

Language
```

---

## Historical Layer

```
Year

Date

Historical Event

Administrative Boundary
```

---

## Provenance Layer

```
Source

Book

Archive

Page

Image

OCR Segment

Human Reviewer
```

---

# Neo4j Labels

```
:Country

:Gau

:Province

:Kreis

:Ort

:Street

:Surname

:Variant

:GivenName

:Source

:Archive

:Page

:OCR

:Document

:Frequency

:Language

:Reviewer
```

---

# Relationships

## Geography

```
PART_OF

CONTAINS

NEIGHBOURS
```

---

## Demographic

```
APPEARS_IN

LIVES_IN

BORN_IN

DIED_IN

MARRIED_IN
```

---

## Frequency

```
REGIONAL_FREQUENCY

LOCAL_FREQUENCY

NATIONAL_FREQUENCY
```

---

## Name Intelligence

```
HAS_VARIANT

LIKELY_VARIANT

OCR_CONFUSION

PHONETIC_MATCH

DERIVED_FROM
```

---

## Provenance

```
FOUND_IN

EXTRACTED_FROM

VALIDATED_BY

REVIEWED_BY

GENERATED_BY
```

---

# Frequency Model

Each surname stores frequency independently at every administrative level.

Example

```
Surname

    │

    ├── Ort Frequency

    ├── Kreis Frequency

    ├── Gau Frequency

    └── Reich Frequency
```

Example

```
Hermann

Pernegg
Count = 1

↓

Bruck an der Mur
Count = 17

↓

Steiermark
Count = 763

↓

Reich
Count = 4,928
```

This allows Bayesian confidence estimation.

---

# Trust Model

Confidence is not binary.

It is computed.

Example

```
Confidence Score

=

Local Evidence

+

Regional Frequency

+

OCR Confidence

+

Variant Similarity

+

Source Reliability

+

Human Validation
```

---

# Bayesian Interpretation

Instead of

```
Surname exists
```

the graph stores

```
Probability
```

Example

```
Hermann

Local

1

Regional

763

Likelihood

High

Noise Risk

Very Low
```

---

# OCR Intelligence

Store confusion edges.

```
Murhammer

↓

Murnhammer

↓

Murhammer

↓

Nurhammer
```

Relationship

```
OCR_CONFUSION
```

Weight

```
0.92

0.84

0.63
```

---

# Variant Model

```
Schmid

↓

Schmidt

↓

Schmied

↓

Schmiedt
```

These become

```
HAS_VARIANT
```

with confidence weights.

---

# Agentic AI Architecture

Instead of one pipeline, use specialized agents.

---

## Agent 1

Document Acquisition

Responsibilities

- scan archives

- import PDFs

- metadata extraction

---

## Agent 2

OCR Agent

Responsibilities

- OCR

- handwriting detection

- segmentation

---

## Agent 3

Name Extraction Agent

Responsibilities

Identify

- surname

- firstname

- occupation

- address

---

## Agent 4

Normalization Agent

Responsibilities

Convert

```
HERMANN

Hermann

Herrmann

Herman
```

into canonical forms.

---

## Agent 5

Historical Geography Agent

Responsibilities

Resolve

```
Pernegg

↓

Kreis

↓

Gau

↓

Historical boundaries
```

---

## Agent 6

Frequency Agent

Computes

Local frequencies

Regional frequencies

National frequencies

Rare names

Hapaxes

---

## Agent 7

Graph Builder

Writes

Neo4j nodes

Neo4j relationships

Indexes

Constraints

---

## Agent 8

Validation Agent

Checks

duplicate people

duplicate surnames

OCR conflicts

variant conflicts

---

## Agent 9

Graph RAG Agent

Retrieves

subgraphs

instead of documents.

Produces

context package for LLM.

---

# Graph RAG Retrieval

Instead of vector search

```
Hermann
```

retrieve

```
Surname

↓

Variant

↓

Ort

↓

Kreis

↓

Gau

↓

Nearby surnames

↓

Historical sources

↓

OCR confidence

↓

Frequency

↓

Maps
```

The LLM receives an entire historical neighbourhood.

---

# Interactive Dashboard

Technology

```
Neo4j

+

Streamlit

+

PyVis

+

Leaflet

+

D3.js

+

Graph Data Science
```

---

# Dashboard Views

## Geographic Map

Surname heatmaps

Historical boundaries

Migration paths

---

## Network Graph

Surname

↓

Places

↓

Variants

↓

Sources

---

## Timeline

Surname frequency

across years

---

## OCR Review

Original scan

↓

OCR text

↓

Corrected text

↓

Confidence

---

## Graph Statistics

Number of

People

Places

Sources

Variants

Relationships

Confidence distributions

---

# Graph Analytics

Use Neo4j Graph Data Science.

Algorithms

```
PageRank

Community Detection

Node Similarity

Betweenness

Weakly Connected Components

Label Propagation

Shortest Path

Louvain

FastRP

KNN
```

Applications

Identify surname clusters.

Detect migration corridors.

Detect OCR anomalies.

Find likely spelling variants.

Suggest missing relationships.

---

# Graph RAG Dictionary

The graph becomes the master dictionary.

Instead of

```
dictionary.json
```

the LLM queries Neo4j.

Example

```
MATCH

Surname

↓

Variant

↓

Regional Frequency

↓

Historical Sources

↓

OCR Confidence

↓

Nearby Names
```

Returned as structured context.

---

# Long-Term Vision

The knowledge graph evolves into a historical surname intelligence platform capable of:

- reconstructing administrative geography
- validating OCR automatically
- identifying probable surname variants
- supporting genealogical research
- providing explainable Graph RAG retrieval
- enabling interactive historical exploration
- serving as the canonical historical surname dictionary for AI-assisted archival analysis

The graph is not merely a storage layer; it functions as a continuously enriched historical knowledge base where every assertion is linked to provenance, weighted by evidence, and available for transparent retrieval by both humans and AI systems.

---

# Document Index

Subsequent documents in the `Graphical RAG` folder:

1. `01-Architecture.md` — overall system architecture.
2. `02-Neo4j-Schema.md` — complete node, relationship, constraints, and indexes.
3. `03-Agentic-Pipeline.md` — detailed multi-agent workflows and orchestration.
4. `04-Ingestion-Pipeline.md` — OCR, ETL, normalization, and provenance.
5. `05-GraphRAG-Retrieval.md` — retrieval algorithms, Cypher patterns, and context assembly.
6. `06-Dashboard.md` — Streamlit/D3/Neo4j visualization design.
7. `07-Confidence-Model.md` — Bayesian trust scoring, OCR confidence, and frequency weighting.
8. `08-Graph-Analytics.md` — Neo4j Graph Data Science algorithms and use cases.
9. `09-API.md` — REST/GraphQL endpoints for external applications.
10. `10-Roadmap.md` — phased implementation plan from proof-of-concept to a complete historical surname knowledge graph.
