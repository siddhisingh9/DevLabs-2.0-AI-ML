# Langgraph Multi-Agent

## Structure

```bash
User Query
    ↓
 SUPERVISOR (LLM)
  ↙    ↓    ↘
R     C     W        ← Researcher, Coder, Writer
  ↘    ↑    ↗
 SUPERVISOR (LLM)    ← sees all results, decides next step
    ↓
  Answer
```

## Prompt and Output

```bash
########################## 

Enter your Message : Tell me about the Recent Football Match Happen 

 ########################## 

As a technical writer, I have synthesized the recent activity regarding the **2026 FIFA World Cup** and domestic league schedules. Below is a structured summary of the current football landscape.

***

# Football Match Summary Report
**Date:** July 3, 2026
**Subject:** Recent Tournament and League Activity

## 1. FIFA World Cup™ (Round of 32)
The 2026 FIFA World Cup is currently in the **Round of 32** stage. This is a critical knockout phase where tournament favorites are vying for a position in the Round of 16.

*   **Recent Notable Fixture:** On July 2, 2026, a significant match took place between **Spain** and **Austria**.
*   **Tournament Status:** Competition remains high, with teams from various confederations competing in high-stakes knockout matches.

## 2. Domestic League Activity
While the international tournament takes center stage, several domestic leagues are active or entering their scheduling phases. 

*   **European Leagues:** Fixtures for the 2026 season are being tracked across major competitions, including:
    *   **England:** Championship, League One, and League Two.
    *   **Continental Europe:** Various leagues, including Serie A, Bundesliga, Ligue 1, and La Liga, are in the process of finalizing fixtures and pre-season activity.
*   **Regional Leagues:** National competitions such as the Scottish Premiership and Norwegian 1. divisjon (Matchday 14) are also reporting active game logs.

## 3. Data Sources and Tracking
For real-time updates and historical match data, the following platforms provide the most consistent technical data feeds:

| Platform | Scope |
| :--- | :--- |
| **OneFootball** | Tournament-specific match reports and daily recaps. |
| **LiveScore** | Granular data on minute-by-minute scores and fixtures. |
| **Sky Sports** | Broad coverage of global league fixtures and tournament progress. |

***

**Note to Reader:** *To provide a more specific breakdown—such as match statistics, goal scorers, or possession data—please specify the particular match or team you are researching.*

 ########################## 
 ```