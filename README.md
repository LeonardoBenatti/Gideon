# Gideon

Assistente chatbot locale basato su LLM, con interfaccia web moderna, ricerca online in tempo reale, esecuzione sandboxed di codice Python e gestione documentale avanzata.

Gideon viene eseguito interamente sul tuo computer: il modello linguistico gira in locale tramite [llama.cpp](https://github.com/ggml-org/llama.cpp), garantendo massima privacy, zero costi di abbonamento e assenza di API cloud obbligatorie.

---

## Caratteristiche principali

* 💬 **Chat multilingue con ragionamento (*thinking*)**: visualizzazione del flusso di pensiero interno del modello prima della risposta.
* 🔍 **Ricerca Web Integrata**: supporto trasparente per DuckDuckGo (zero configurazione), Brave Search API o istanze SearXNG locali.
* 🐍 **Esecuzione Python Sandboxed**: calcoli matematici precisi, statistiche, elaborazione dati e grafici generati in tempo reale in ambiente protetto.
* 📄 **Analisi Documentale & Visione Multimodale**: lettura e analisi di PDF con estrazione testo/OCR, analisi di immagini e file di codice sorgente.
* 📁 **Creazione & Download File**: capacità dell'assistente di generare script, fogli di calcolo o testi e renderli scaricabili con un clic.
* 🌐 **Interfaccia Web Reattiva**: UI in stile dark mode, storico conversazioni, rendering Markdown con syntax highlighting e LaTeX.

---

## Requisiti di Sistema

| Componente | Requisito Minimo | Consigliato |
| :--- | :--- | :--- |
| **Sistema Operativo** | Windows 10/11 (64-bit) | Windows 11 (64-bit) |
| **RAM di Sistema** | 16 GB | 32 GB |
| **Scheda Video (GPU)** | NVIDIA con ≥ 6 GB VRAM (CUDA) | NVIDIA RTX con ≥ 8/12 GB VRAM |
| **Spazio su Disco** | ~8 GB liberi | SSD con 15+ GB liberi |
| **Python** | Python 3.11+ | Python 3.11 o 3.12 |
| **llama-server** | [`llama.cpp`](https://github.com/ggml-org/llama.cpp) con backend CUDA | Installabile automaticamente |

---

## Avvio Rapido (Windows)

Il progetto include uno script di avvio completamente automatizzato (`start.bat`):

1. **Clona la repository**:
   ```bash
   git clone https://github.com/LeonardoBenatti/Gideon.git
   cd Gideon
   ```

2. **Avvia Gideon con un clic**:
   Fai doppio clic su:
   ```
   start.bat
   ```
   nella cartella principale (oppure su `Qwen 3.5-9B-4bit\start.bat`).
   *Al primo avvio, `start.bat` eseguirà in automatico:*
   * La creazione del virtual environment Python (`shared_venv`)
   * L'installazione di tutte le dipendenze necessarie
   * Il download del modello quantizzato `Qwen3.5-9B-Q4_K_M.gguf` e del proiettore visivo `mmproj` da Hugging Face
   * La verifica e il download di `llama-server` (tramite `winget` se non presente)

3. **Apri il browser**:
   Una volta visualizzato il messaggio *"Application startup complete"*:
   * **Da questo computer**: [http://localhost:8000](http://localhost:8000)
   * **Da altri dispositivi nella stessa rete locale**: `http://<IP-DEL-TUO-PC>:8000`

---

## Installazione Manuale Alternativa

Se preferisci preparare l'ambiente manualmente:

1. Crea e attiva l'ambiente virtuale:
   ```powershell
   python -m venv shared_venv
   .\shared_venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Scarica i pesi del modello in `shared_models/qwen3.5-9b/`:
   * [Qwen3.5-9B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
   * [mmproj-F16.gguf](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)

3. Avvia il server backend:
   ```powershell
   cd "Qwen 3.5-9B-4bit"
   python server.py
   ```

---

## Configurazione Avanzata

Copia `.env.example` in `.env` per personalizzare impostazioni opzionali:
* `BRAVE_API_KEY`: per abilitare la ricerca tramite Brave Search API.
* `SEARXNG_URL`: indirizzo di un'istanza locale o privata SearXNG.
* `CTX`: lunghezza della finestra di contesto in token (default: 32768).
* `ENABLE_THINKING`: `1` per abilitare la catena di ragionamento, `0` per disabilitarla.

---

## Struttura del Progetto

```
Gideon/
├── start.bat                  # Launcher principale (avvio rapido con un clic)
├── Qwen 3.5-9B-4bit/
│   ├── server.py              # Backend FastAPI, gestione LLM e tool
│   ├── index.html             # Frontend Web UI interattivo
│   ├── start.bat              # Script di avvio dettagliato
│   └── tests/                 # Test di sicurezza sandbox Python
├── requirements.txt           # Dipendenze Python
├── .env.example               # Template variabili d'ambiente
├── .gitignore                 # Esclusioni da Git (pesi, venv, log, chat)
└── LICENSE                    # Licenza MIT
```

---

## Licenza

Questo progetto è distribuito sotto licenza [MIT](LICENSE).

