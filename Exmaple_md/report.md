# Rapport: Tower of Hanoi - Breadth-First Search Implementation

## Implementerade funktioner

### 1. `find_shortest_path(start_state, target_state)`
Denna funktion implementerar Breadth-First Search (BFS) för att hitta den kortaste vägen mellan två tillstånd i Tower of Hanoi.

- **Initialisering:** Skapar en kö (`deque`) för att lagra vägar och en `visited` mängd för att undvika oändliga loopar
- **BFS loop:** Processar vägar i "först in, först ut" ordning för att garantera kortaste väg
- **Tillståndsexpansion:** Använder `get_valid_moves()` för att generera alla möjliga nästa tillstånd
- **Vägkonstruktion:** Bygger nya vägar genom att lägga till nästa tillstånd till befintliga vägar

### 2. Datastrukturer och algoritm

**Kö (Queue):**
- Använder `collections.deque` för effektiv FIFO operation
- Lagrar kompletta vägar (listor av tillstånd) istället för enskilda tillstånd
- Garanterar att kortaste vägar utforskas först

**Besökt mängd (Visited Set):**
- Håller reda på tillstånd som redan har behandlats
- Förhindrar oändliga loopar och redundant beräkning
- O(1) lookup tid för snabb prestanda


### Detaljerad väg för n=3 (7 drag):
1. **('A', 'A', 'A')** - Start: Alla diskar på pinne A
2. **('C', 'A', 'A')** - Flytta disk 1 till C
3. **('C', 'B', 'A')** - Flytta disk 2 till B
4. **('B', 'B', 'A')** - Flytta disk 1 till B
5. **('B', 'B', 'C')** - Flytta disk 3 till C
6. **('A', 'B', 'C')** - Flytta disk 1 till A
7. **('A', 'C', 'C')** - Flytta disk 2 till C
8. **('C', 'C', 'C')** - Flytta disk 1 till C (mål)


## Teknisk implementation

### BFS-algoritm struktur
```python
# Initialisering
queue = deque([start_state])
visited = {start_state}

# Huvudloop
while queue:
    current_path = queue.popleft()  # FIFO
    current_state = current_path[-1]
    
    if current_state == target_state:
        return current_path  # Kortaste väg hittad
    
    # Expandera tillstånd
    for next_state in get_valid_moves(current_state):
        if next_state not in visited:
            visited.add(next_state)
            queue.append(current_path + [next_state])
```

### Tillståndsrepresentation
- **Tupel format:** `('A', 'B', 'C')` för 3 diskar
- **Index-betydelse:** `state[i]` = pinne där disk i+1 befinner sig
- **Giltiga drag:** Kontrolleras av `get_valid_moves()` funktionen

## Varför BFS är optimal för detta problem

### 1. Kortaste väg-garanti
**BFS hittar alltid den kortaste vägen:**
- Utforskar vägar i stigande ordning av längd
- Första lösningen som hittas är garanterat optimal
- Perfekt för optimeringsproblem som Tower of Hanoi

### 2. Komplett sökning
**Undviker lokala optima:**
- Systematisk genomgång av alla möjliga vägar
- Hittar lösning om den existerar
- Ingen risk att missa den optimala lösningen

### 3. Deterministik beteende
**Förutsägbar och reproducerbar:**
- Samma input ger alltid samma output
- Lätt att debugga och verifiera
- Perfekt för automatiserad testning

## Varför inte Depth-First Search (DFS)?

### 1. Ingen kortaste väg-garanti
**DFS kan hitta långa vägar först:**
- Kan gå djupt in i en gren innan den hittar målet
- Fösta lösningen som hittas är inte nödvändigtvis kortast
- Kräver komplett sökning för att garantera optimalitet

### 2. Oändliga loopar
**Risk för stack overflow:**
- Kan gå oändligt djupt i felaktiga grenar
- Kräver explicit djupgräns eller besökt-kontroll
- Mindre robust än BFS för detta problem

### 3. Ineffektiv för optimering
**Inte designad för kortaste väg-problem:**
- DFS är bättre för existensproblem ("finns det en lösning?")
- BFS är designad för optimeringsproblem ("vad är den bästa lösningen?")
- Tower of Hanoi handlar om att hitta den kortaste sekvensen

## Reflektioner

### Styrkor med BFS-implementationen
1. **Matematiskt korrekt:** Följer BFS-algoritmen exakt enligt specifikationen
2. **Effektiv:** Använder `deque` för O(1) enqueue/dequeue-operationer
3. **Robust:** Hanterar edge cases som identiska start- och måltillstånd
4. **Läsbar:** Tydlig kodstruktur med kommentarer

### Möjliga förbättringar
1. **Bidirectional BFS:** Sök från båda hållen för stora problem
2. **Heuristik:** A* med Manhattan distance för snabbare sökning
3. **Memoization:** Cachning av `get_valid_moves()` resultat

### Slutsats
BFS är det perfekta valet för Tower of Hanoi eftersom problemet handlar om att hitta den **kortaste** vägen mellan två tillstånd. Algoritmen garanterar optimalitet och är enkelt att implementera och förstå.

---

- *AI har använts som bollplank för rapporten samt stöd i kodning.*
