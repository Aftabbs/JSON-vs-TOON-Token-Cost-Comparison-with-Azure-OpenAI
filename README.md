# JSON vs TOON – Token & Cost Comparison with Azure OpenAI (gpt-4o)

<img width="281" height="179" alt="image" src="https://github.com/user-attachments/assets/c5957129-6262-437e-b0f0-8eaa833aab73" />


This repo is a **small, self-contained demo** showing how much token usage and cost you can save by switching from **JSON** to **TOON (Token-Oriented Object Notation)** when sending structured data to an LLM.

It runs the **same task** twice:

1. Once with a **JSON** payload
2. Once with a **TOON** payload (a compact, token-efficient notation)

Then it prints:

- Actual **prompt / completion / total tokens**
- An **estimated cost** for each run
- **% reductions** in tokens and cost

Example (real run):

```text
--- JSON INPUT ---
Prompt tokens    : 538
Completion tokens: 265
Total tokens     : 803
Estimated cost   : $0.004394

--- TOON INPUT ---
Prompt tokens    : 364
Completion tokens: 207
Total tokens     : 571
Estimated cost   : $0.003278

======================== COMPARISON SUMMARY ========================
Prompt token reduction : 32.34%
Total token reduction  : 28.89%
Cost reduction (est.)  : 25.41%
===================================================================
````

---

## 1. Project Structure

Minimal setup:

```text
.
├─ t.py                     # or json_vs_toon_demo.py – main demo script
├─ README.md                # this file
└─ .env                     # Azure credentials + model config
```

You can rename `t.py` to something like `json_vs_toon_demo.py` if you prefer.

---

## 2. Prerequisites

* Python 3.9+ (recommended)
* An **Azure OpenAI** resource with a deployed **gpt-4o** model
* A virtual environment (optional but recommended)

---

## 3. Environment Variables (`.env`)

Create a `.env` file in the project root with:

```env
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-05-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_MODEL=gpt-4o
```

> `AZURE_OPENAI_DEPLOYMENT` should match the **deployment name** you configured in the Azure portal for gpt-4o.

---

## 4. Install Dependencies

From the project directory:

```bash
# create venv if you want (optional)
python -m venv venv
venv\Scripts\activate      # on Windows
# source venv/bin/activate # on macOS / Linux

pip install -r requirements.txt
```

If you’re not using a `requirements.txt`, you can install directly:

```bash
pip install openai python-dotenv toon-python
```

---

## 5. Running the Demo

Run the script:

```bash
python t.py
# or
python json_vs_toon_demo.py
```

The script will:

1. Build a small, realistic **buyer + property listings** dataset.
2. Create **two prompts**:

   * One embedding the data as **pretty JSON**
   * One embedding the data as **TOON**
3. Call Azure OpenAI (gpt-4o) for each version.
4. Print metrics and a sample of the model’s output.

---

## 6. How the Script Works (High Level)

Inside `t.py`:

* **Environment & client**

  ```python
  from dotenv import load_dotenv
  from openai import AzureOpenAI

  load_dotenv()

  client = AzureOpenAI(
      api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
  )

  deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
  ```

* **Sample data** (you can replace with your own):

  ```python
  data = {
      "buyer_profile": { ... },
      "listings": [ ... ],
  }
  ```

* **JSON prompt**: embeds `json.dumps(data, indent=2)`.

* **TOON prompt**: embeds `toon_encode(data)` from `toon-python`.

  ```python
  from toon_python import encode as toon_encode
  toon_block = toon_encode(data)
  ```

* **Model call**:

  ```python
  response = client.chat.completions.create(
      model=deployment,
      temperature=0,
      messages=[
          {"role": "system", "content": "You are a concise, expert real estate analyst."},
          {"role": "user", "content": user_prompt},
      ],
  )
  ```

* **Token & cost calculation**:

  ```python
  usage = response.usage
  prompt_tokens = usage.prompt_tokens
  completion_tokens = usage.completion_tokens
  total_tokens = usage.total_tokens

  # configurable pricing assumptions
  INPUT_COST_PER_1K = 0.00275
  OUTPUT_COST_PER_1K = 0.011

  cost = (prompt_tokens / 1000) * INPUT_COST_PER_1K \
       + (completion_tokens / 1000) * OUTPUT_COST_PER_1K
  ```

* Finally, it prints JSON vs TOON stats and the % savings.

---

