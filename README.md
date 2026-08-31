Gruppo 13 -FLuSCo Racing: Guida Autonoma su TORCS attraverso Imitation Learning

Il seguente progetto illustra l'implementazione di un agente di guida autonoma per il simulatore TORCS, sviluppato in occasione della IBM AI Racing League 2026 e ottimizzato per il circuito Corkscrew.
La strategia adottata si fonda sul Behavioral Cloning (BC). Il sistema utilizza una rete neurale di tipo Multi-Head MLP (DrivingMLP) addestrata per mappare i dati dei sensori (26 input) ai comandi fisici del veicolo (sterzo, acceleratore, freno).
Il driver integra, inoltre, meccanismi di sicurezza anti-uscita (*Soft Track-Keeping* e *Edge Braking*) per mitigare il Distributional Shift e garantire una guida fluida e sicura, completando il giro senza danni.

---
Struttura del Progetto
Il progetto è contenuto nella cartella `LavoroFLuSCoRace` ed è organizzato in moduli sequenziali per garantire chiarezza architetturale:

* `dataset/` --> Contiene la telemetria di addestramento generata dal bot (`bot_dataset.csv`).
* `bot.py` --> Script per la Fase 1: Raccolta Dati tramite Pilota Esperto Rule-Based.
* `train.py` --> Script per la Fase 2: Addestramento della rete neurale (DrivingMLP).
* `drive.py` --> Script per la Fase 3: Agente di guida in tempo reale (Inferenza).
* `snakeoil3_*.py` --> Librerie di comunicazione UDP per l'interfacciamento client-server con TORCS.

---

Requisiti e Installazione
Il progetto richiede la versione 3.11 di Python (o compatibile). Si consiglia l'utilizzo di un ambiente virtuale isolato.

1. Posizionarsi nella cartella del progetto da terminale:
```bash
cd Desktop/LavoroFLuSCoRace

```

2. Creare e attivare l'ambiente virtuale: *(Windows)*
```bash
python -m venv venv
.\venv\Scripts\activate

```

*(Linux/Mac)*
```bash
python3 -m venv venv
source venv/bin/activate

```

3. Installare le dipendenze: Eseguire il seguente comando per installare tutte le librerie necessarie (PyTorch, scikit-learn, pandas, ecc.):
```bash
pip install torch pandas numpy scikit-learn pynput

```

---

Modalità di Esecuzione
Per visualizzare correttamente il giro dell'agente autonomo su TORCS è fondamentale rispettare il seguente ordine di esecuzione. 
Il simulatore TORCS funge da Server e deve essere in "attesa" prima di lanciare lo script Python (Client).

PASSO 1: Preparazione e Avvio di TORCS
1. Avviare il simulatore tramite interfaccia grafica.
2. Navigare nel menù principale per configurare la gara:
* Cliccare su Race --> Quick Race --> Configure Race
* Selezionare il tracciato: cercare e selezionare Corkscrew, quindi, cliccare su Accept.
* Configurare i piloti: nella lista dei guidatori, assicurarsi di rimuovere tutti i piloti umani o altri bot e selezionare solo il bot chiamato "scr_server 1" (server UDP che attende il codice in Python) e cliccare su Accept.
3. Avviare la simulazione cliccando su New Race.
4. A questo punto, la schermata di TORCS caricherà la pista e si fermerà mostrando la dicitura "Initializing Driver scr_server 1", indicando che TORCS è in attesa del codice Python.

PASSO 2: Avvio della Rete Neurale (Client Python)
Con TORCS in attesa, aprire il terminale (con l'ambiente virtuale attivato e posizionato nella cartella "LavoroFLuSCoRace"):
1. Per avviare la guida standard (Modalità Consigliata):
Questo comando avvia il modello MLP caricando i pesi pre-addestrati e attivando i sistemi di correzione della traiettoria per completare il giro in sicurezza.
```bash
python drive.py

```
2. Per avviare la guida senza aiuti (Modalità Raw - Opzionale):
Questo comando disabilita ABS, TCS e limitatore di sovrasterzo, permettendo di valutare le decisioni "pure" della rete neurale.
```bash
python drive.py --raw

```
Appena inviato il comando sul terminale, comparirà la scritta "Connesso a TORCS su porta 3001"* e l'auto in TORCS partirà istantaneamente, iniziando a guidare da sola.
(Per fermare la simulazione, premi "ESC" su TORCS e poi "CTRL+C" nel terminale per chiudere lo script).

---

Esecuzione delle Fasi Precedenti
Per creare il dataset o riaddestrare la rete da zero bisogna procedere con i seguenti step:
* Raccolta dati: Avviare una gara in TORCS su porta 3001, quindi lanciare:
```bash
python bot.py

```
* Addestramento Rete: Con il dataset generato, avviare il training (impiegherà un po' di tempo):
```bash
python train.py --csv dataset/bot_dataset.csv --out-dir ./output

```
---

**Gruppo 13 - Luisa Ingenito, Silvia Liguoro, Lucia Monetta, Cosimo Rivellini**
