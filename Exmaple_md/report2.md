# Rapport: Maze Runner - Reinforcement Learning Agent

## 1. step() funktionen i Maze klassen

Denna funktion hanterar robotens rörelser och belöningar i labyrinten:

- **Rörelsevalidering:** Kontrollerar om rörelsen är tillåten med `_is_move_allowed()`
- **Stegräkning:** Ökar `self.steps += 1` för varje steg
- **Positionuppdatering:** Beräknar ny position baserat på action och uppdaterar labyrinten
- **Belöningssystem:** 
  - 0 poäng för att nå mål
  - -1 poäng för vanligt steg
  - -10 poäng för att gå in i vägg eller utanför gränser

## 2. Hyperparameter Experiment

Testade 9 olika kombinationer av alpha (learning rate) och random_factor (exploration rate):

### Testade parametrar:
- **Alpha värden:** 0.01, 0.1, 0.5
- **Random factor värden:** 0.1, 0.3, 0.5
- **Episoder:** 3000 per kombination

### Resultat (sorterat efter prestanda):

![Graf of parameters](./resources/Figure_1.png)

<details>
<summary>A better view of the graphs</summary>

### Detailed Results

![Graph](./resources/alpha_comparison.png)
![Graph](./resources/exploration_rate_comparison.png)
![Graph](./resources/individual_learning_curves.png)
![Graph](./resources/normal_minus.png)

</details>

1. **α=0.1, ε=0.1: 10.66 steg (genomsnitt)**
2. α=0.01, ε=0.1: 10.78 steg (genomsnitt)
3. α=0.01, ε=0.3: 13.20 steg (genomsnitt)
4. α=0.01, ε=0.5: 17.98 steg (genomsnitt)
5. α=0.1, ε=0.5: 30.52 steg (genomsnitt)
6. α=0.1, ε=0.3: 105.40 steg (genomsnitt)
7. α=0.5, ε=0.3: 109.27 steg (genomsnitt)
8. α=0.5, ε=0.5: 113.01 steg (genomsnitt)
9. α=0.5, ε=0.1: 162.36 steg (genomsnitt)

## 3. Analys av Resultat

### Bästa kombination: α=0.1, ε=0.1

**Motivering:**
- **Balanserad inlärning:** Alpha=0.1 ger tillräckligt snabb inlärning utan att vara för volatil
- **Optimal utforskning:** Random_factor=0.1 balanserar utforskning och exploatering perfekt
- **Snabb konvergens:** Når optimal prestanda snabbast av alla kombinationer

### Exploration vs Exploitation

**Exploration:**
- Högre random_factor => mer slumpmässiga drag
- Nödvändigt för att upptäcka nya strategier
- För mycket => långsam inlärning (se ε=0.5 resultat)

**Exploitation:**
- Lägre random_factor => fokuserar på bästa kända drag
- Effektivt när man redan vet vad som fungerar
- För lite => kan fastna i lokala optima

### Learning Rate (Alpha) påverkan

- **α=0.01:** För konservativ, långsam inlärning
- **α=0.1:** Optimal balans mellan stabilitet och snabbhet
- **α=0.5:** För aggressiv, kan vara instabil

## 4. Väggstraff Implementation

Väggstraffet (-10) implementerat för att:
- **Snabbare inlärning:** Agenten lär sig snabbt att undvika väggar
- **Bättre konvergens:** Tydlig negativ feedback för dåliga drag
- **Realistisk beteende:** Motsvarar verklighetens konsekvenser

### Hypotes vs Resultat

**Hypotes:** Väggstraffet skulle förbättra inlärningen avsevärt jämfört med bara -1 för steg.

![Graf of parameters](./resources/normal_minus.png)
_Graferna blir mycket “spretigare” – agenten tar längre tid på sig att stabilisera sig._

**Resultat:** Bekräftat! Agenten når optimal prestanda (10.66 steg) mycket snabbare med väggstraffet.

## 5. Slutsatser

1. **Optimal hyperparameter kombination:** α=0.1, ε=0.1 ger bäst prestanda
2. **Väggstraffet är kritiskt:** Utan det skulle inlärningen vara mycket långsammare
3. **Balans är nyckeln:** För mycket exploration eller exploitation försämrar prestandan
4. **Learning rate påverkar stabilitet:** För hög alpha kan göra inlärningen instabil

## 6. Teknisk Implementation

### Agentens inlärningsprocess:
```python
# Epsilon-greedy action selection
if random.random() < self.random_factor:
    return random.choice(moves)  # Explore
else:
    return next_move  # Exploit

# Monte Carlo state-value update (every-visit)
self.G[prev_state] = self.G[prev_state] + self.alpha * (target - self.G[prev_state])

```

### Belöningsstruktur:
- **Mål:** 0 (optimal)
- **Vanligt steg:** -1 (motiverar snabbhet)
- **Väggträff:** -10 (starkt avskräckande)

**Resultatet visar att agenten framgångsrikt lär sig navigera labyrinten med en genomsnittlig prestanda på 10.66 steg, vilket är nära det teoretiska minimumet för denna labyrint.**


- _Ai har använts som bollplank till report.md samt stöd i kodning._

