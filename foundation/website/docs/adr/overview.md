---
sidebar_position: 0
title: "Architecture Decision Records"
description: "Collection of architectural decisions made during ARES ChronoFabric development"
---

# Architecture Decision Records (ADRs)

This section contains Architecture Decision Records documenting important architectural decisions made during the development of ARES ChronoFabric.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Index

### [ADR 0000: Template](./0000-template.md)
Template for creating new ADRs with consistent structure and format.

### [ADR 0001: Single Infrastructure Components](./0001-one-bus-one-scheduler-one-config.md)
Decision to use single bus, scheduler, and configuration system to avoid duplication and nondeterminism.

## Creating New ADRs

To create a new ADR:

1. Copy the [template](./0000-template.md)
2. Assign the next available number
3. Fill in the context, decision, consequences, and alternatives
4. Submit for review through the normal development process

## ADR Lifecycle

ADRs can have the following statuses:

- **Proposed**: Under discussion and review
- **Accepted**: Decision has been made and is being implemented
- **Superseded**: Replaced by a newer ADR (reference the superseding ADR)
- **Deprecated**: No longer relevant but kept for historical context