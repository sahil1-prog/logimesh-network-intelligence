# Operations Strategy Memo: Optimizing Delivery ETAs with Graph-Based Network Intelligence

**To:** Head of Network Operations, LogiMesh
**From:** Data Science Strategy Team
**Date:** June 14, 2026

## Executive Summary
Our strategic analysis of 144,800+ recent delivery trips reveals that treating the LogiMesh logistics network as a connected graph rather than independent point-to-point estimates drastically improves ETA accuracy. By integrating structural network features, our Graph-Enhanced ETA model **reduces prediction error (MAE) from 54.6 minutes to 43.3 minutes** and increases the percentage of predictions within 15% of actual times from **43.6% to 51.0%**. Furthermore, our bottleneck audit isolated the specific hubs systematically causing downstream network delays, offering a targeted path to reduce SLA breaches.

## Top 5 Structural Bottleneck Hubs
Using network centrality metrics (Betweenness Centrality) weighted against the volume of chronically delayed outgoing trips (where actual time exceeds OSRM estimates by >20%), we have identified the top 5 highest-impact bottlenecks:

1. **Gurgaon_Bilaspur_HB (Haryana)** 
   - *SLA Breach Score:* 4937.3
   - *Recommendation:* **Facility Upgrade & Parallel Routing.** As the highest structural risk by an order of magnitude, congestion here cascades across the network.
2. **Bangalore_Nelmngla_H (Karnataka)** 
   - *SLA Breach Score:* 1229.1
   - *Recommendation:* **Route-Type Shift.** Increase FTL adoption on medium-distance outgoing corridors to bypass carting delays.
3. **Bhiwandi_Mankoli_HB (Maharashtra)**
   - *SLA Breach Score:* 632.5
   - *Recommendation:* **Facility Upgrade.** Expand sorting capacity to reduce dwell times during Evening/Night shifts.
4. **Hyderabad_Shamshbd_H (Telangana)**
   - *SLA Breach Score:* 309.4
   - *Recommendation:* **Parallel Routing.** Reroute lower-priority Carting shipments through alternate secondary hubs.
5. **Kolkata_Dankuni_HB (West Bengal)**
   - *SLA Breach Score:* 236.8
   - *Recommendation:* **Time-of-Day Optimization.** Shift Carting dispatch times from Morning to Afternoon to leverage lower delay factors.

## Business Impact & Revenue-at-Risk Recovery
If targeted interventions (upgrades and parallel routing) are successfully implemented at the **Top 3 Hubs** (Gurgaon, Bangalore, Bhiwandi), we project a significant reduction in late deliveries. 
*   **Volume Impact**: These top 3 hubs account for ~85% of the top-5 network structural risk. Resolving their congestion will reduce network-wide SLA breaches on chronically delayed corridors by an estimated **18-22%**.
*   **Revenue Recovery**: Assuming an industry-standard penalty/cost of ₹1,000 per SLA breach, recovering 20% of the SLA breaches cascading from these top 3 hubs represents an estimated **₹12-15 Lakhs** in recovered revenue-at-risk per month.

## FTL vs. Carting Decision Framework
Our analysis quantifies the time-cost trade-offs for route-type selection based on corridor distance and time of day:
*   **Short Corridors**: Carting is generally efficient, except during Evening hours where FTL shows a significantly lower delay factor (2.37 vs 1.89).
*   **Medium Corridors**: FTL consistently outperforms Carting in predictability across all times of day. Switching from Carting to FTL on Medium/Evening corridors reduces the delay factor from 2.54 to 2.08.
*   **Long Corridors**: Carting is highly unpredictable (Delay factor > 4.7) during Evening hours. **Recommendation**: Mandate FTL for all Long/Evening corridors to ensure SLA compliance, despite the higher unit cost.
