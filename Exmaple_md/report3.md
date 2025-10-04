# Rapport: Maximera vinsten för The Daily Grind


## Implementerade funktioner

### 1. calculate_daily_profit(params)
Denna funktion beräknar daglig vinst baserat på följande formler:
- **Total intäkt** = kaffe pris per kopp × genomsnittliga kunder per dag
- **Totala rörliga kostnader** = (bönkostnad + mjölkkostnad) × genomsnittliga kunder per dag  
- **Daglig vinst** = total intäkt - totala rörliga kostnader - dagliga fasta kostnader

### 2. run_scenario(scenario_name, params)
Funktionen kör olika scenarion och presenterar resultaten:
- Scenario-namn
- Alla parametrar (priser, kostnader, antal kunder)
- Beräknade resultat (intäkt, rörliga kostnader, vinst)

### 3. find_customers_for_profit(target_profit, initial_params)
Målvinstsökande funktion som:
- Itererar genom antal kunder från 1 till max 1000
- Beräknar vinsten för varje antal kunder
- Returnerar det minsta antal kunder som behövs för att nå målvinsten
- Returnerar None om målvinsten inte kan nås inom gränserna

## Analys av resultat

### Basfall (Base Case)
- **Kaffe pris:** 3.50€ per kopp
- **Antal kunder:** 200 per dag
- **Bönkostnad:** 0.75€ per kopp
- **Mjölkkostnad:** 0.50€ per kopp
- **Fasta kostnader:** 150€ per dag

**Resultat:**
- Intäkt: 700.00€
- Rörliga kostnader: 250.00€
- **Daglig vinst: 300.00€**

### Dyrt kaffe  scenario
- **Kaffe pris ökad till:** 4.00€ per kopp (+0.50€)
- Alla andra parametrar oförändrade

**Resultat:**
- Intäkt: 800.00€ (+100.00€)
- Rörliga kostnader: 250.00€ (oföränd)
- **Daglig vinst: 400.00€ (+100.00€)**

### Målvinstsökning
- **Målvinst:** 500.00€ per dag
- **Behövda kunder:** 289 kunder per dag

## Reflektioner

- **Kristoffer, du har använt € i exempel output men i koden har du använt $ samt att € är förre siffrorna och int efter (╯°□°)╯︵ ┻━┻**
- Grundberäkningarna var rätt straightforward när jag väl förstod formlerna
- _Ai har använts som bollplank till report.md samt stöd i kodning._