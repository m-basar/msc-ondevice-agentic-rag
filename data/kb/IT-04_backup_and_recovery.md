---
id: IT-04
title: Backup and Recovery
category: IT
version: 1.3
effective_date: 2025-05-01
status: current
owner: IT Manager
---

# Backup and Recovery

## What is backed up

All business systems are backed up: the order system, the finance system, the People system, file storage, and email. Data held only on a local device is not backed up, which is why staff are required to save work to company storage rather than the desktop.

## Schedule

| System | Frequency | Retention |
|---|---|---|
| Order system | Nightly, plus hourly transaction logs | 30 days |
| Finance system | Nightly | 30 days daily, 12 monthly |
| File storage | Nightly | 30 days |
| Email | Continuous, provider replication | 30 days deleted item recovery |

Monthly backups are retained for twelve months. An annual backup is retained for seven years to meet the financial record requirements in REG-02.

## Where backups are held

Backups are written to encrypted cloud storage in a UK region, and a second copy is held offline. The offline copy exists specifically so that ransomware which reaches the network cannot also reach the backups.

## Recovery objectives

| Measure | Target |
|---|---|
| Recovery point objective | 1 hour for the order system, 24 hours for other systems |
| Recovery time objective | 4 hours for the order system, 1 working day for other systems |

## Testing

A restore test is performed quarterly. One system is chosen at random, restored to an isolated environment, and verified. Results are recorded and reported to the senior team. A backup that has never been restored is not a backup.

## Requesting a restore

Staff who need a deleted file recovered contact the IT helpdesk. Files deleted within the last 30 days can normally be recovered the same working day.
