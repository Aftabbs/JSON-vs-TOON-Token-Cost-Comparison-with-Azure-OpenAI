import os
import json
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AzureOpenAI
from toon_python import encode as toon_encode  # TOON encoder


# These are just defaults for the demo – adjust to your actual contract/region.
INPUT_COST_PER_1K = 0.00275
OUTPUT_COST_PER_1K = 0.011


@dataclass
class RunResult:
    mode: str  # "JSON" or "TOON"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    preview_text: str


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Rough cost estimate in USD based on token counts."""
    return (
        (prompt_tokens / 1000.0) * INPUT_COST_PER_1K
        + (completion_tokens / 1000.0) * OUTPUT_COST_PER_1K
    )


def build_sample_data() -> dict:
    """
    Small but realistic dataset – you can swap this with your own structure.
    """
    return {
        "buyer_profile": {
            "budget_min": 600_000,
            "budget_max": 900_000,
            "target_areas": ["Coral Gables", "Coconut Grove"],
            "must_haves": ["2+ bedrooms", "walkable", "low hoa", "safe neighborhood"],
        },
        "listings": [
            {
                "mls_id": "A11861233",
                "price": 439_900,
                "beds": 2,
                "baths": 2,
                "sqft": 1180,
                "neighborhood": "Aventura",
                "hoa_monthly": 780,
                "walk_score": 82,
                "safety_score": 7.8,
            },
            {
                "mls_id": "A11543210",
                "price": 795_000,
                "beds": 3,
                "baths": 3,
                "sqft": 1650,
                "neighborhood": "Coral Gables",
                "hoa_monthly": 350,
                "walk_score": 89,
                "safety_score": 9.1,
            },
            {
                "mls_id": "A11498765",
                "price": 720_000,
                "beds": 2,
                "baths": 2,
                "sqft": 1420,
                "neighborhood": "Coconut Grove",
                "hoa_monthly": 420,
                "walk_score": 92,
                "safety_score": 8.7,
            },
            {
                "mls_id": "A11800001",
                "price": 910_000,
                "beds": 3,
                "baths": 2.5,
                "sqft": 1750,
                "neighborhood": "Coral Gables",
                "hoa_monthly": 510,
                "walk_score": 86,
                "safety_score": 9.3,
            },
        ],
    }


def build_json_prompt(data: dict) -> str:
    """Prompt that embeds the data as pretty JSON."""
    json_block = json.dumps(data, indent=2)
    return f"""
You are an AI assistant helping a real estate team evaluate condo listings for a buyer.

You will receive property and buyer data in JSON format.
1. Identify the top 2 listings for this buyer.
2. For each, explain briefly why it is a strong match.
3. Briefly mention any listings to avoid and why.

Respond in 3-5 bullet points, concise and professional.

JSON DATA:
{json_block}
""".strip()


def build_toon_prompt(data: dict) -> str:
    """Prompt that embeds the same data encoded as TOON."""
    toon_block = toon_encode(data)
    return f"""
You are an AI assistant helping a real estate team evaluate condo listings for a buyer.

You will receive property and buyer data in TOON format (Token-Oriented Object Notation).

TOON basics:
- Indentation indicates nesting (like YAML).
- Lines like `listings[4]{{field1,field2,...}}:` declare an array of objects.
- Each subsequent indented line is a row with comma-separated values in that field order.

1. Identify the top 2 listings for this buyer.
2. For each, explain briefly why it is a strong match.
3. Briefly mention any listings to avoid and why.

Respond in 3-5 bullet points, concise and professional.

TOON DATA:
{toon_block}
""".strip()


def call_model(client: AzureOpenAI, model: str, user_content: str, mode: str) -> RunResult:
    """Send one chat.completions request and capture usage + cost."""
    response = client.chat.completions.create(
        model=model,  # deployment name in Azure
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a concise, expert real estate analyst.",
            },
            {"role": "user", "content": user_content},
        ],
    )

    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    cost = calculate_cost(prompt_tokens, completion_tokens)

    preview = response.choices[0].message.content
    if len(preview) > 400:
        preview = preview[:400] + "...\n[truncated]"

    return RunResult(
        mode=mode,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        preview_text=preview,
    )


def main():
    load_dotenv()

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if not all([api_key, endpoint, api_version, deployment]):
        raise RuntimeError(
            "Missing one or more Azure env vars: "
            "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT"
        )

    # Azure-specific client
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

    data = build_sample_data()

    json_prompt = build_json_prompt(data)
    toon_prompt = build_toon_prompt(data)

    print("Running JSON run...")
    json_result = call_model(client, deployment, json_prompt, mode="JSON")

    print("Running TOON run...")
    toon_result = call_model(client, deployment, toon_prompt, mode="TOON")

    print("\n==================== JSON vs TOON (gpt-4o on Azure) ====================")

    def report(r: RunResult):
        print(f"\n--- {r.mode} INPUT ---")
        print(f"Prompt tokens    : {r.prompt_tokens}")
        print(f"Completion tokens: {r.completion_tokens}")
        print(f"Total tokens     : {r.total_tokens}")
        print(f"Estimated cost   : ${r.cost_usd:.6f}")
        print("Sample output    :")
        print(r.preview_text)

    report(json_result)
    report(toon_result)

    # Savings calculations
    prompt_savings_pct = (
        100.0
        * (json_result.prompt_tokens - toon_result.prompt_tokens)
        / json_result.prompt_tokens
        if json_result.prompt_tokens > 0
        else 0.0
    )
    total_savings_pct = (
        100.0
        * (json_result.total_tokens - toon_result.total_tokens)
        / json_result.total_tokens
        if json_result.total_tokens > 0
        else 0.0
    )
    cost_savings_pct = (
        100.0 * (json_result.cost_usd - toon_result.cost_usd) / json_result.cost_usd
        if json_result.cost_usd > 0
        else 0.0
    )

    print("\n======================== COMPARISON SUMMARY ========================")
    print(f"Prompt token reduction : {prompt_savings_pct:.2f}%")
    print(f"Total token reduction  : {total_savings_pct:.2f}%")
    print(f"Cost reduction (est.)  : {cost_savings_pct:.2f}%")
    print("===================================================================")
    print(
        "\nNote: Pricing numbers are approximate. "
        "Update INPUT_COST_PER_1K and OUTPUT_COST_PER_1K to match your actual Azure pricing."
    )


if __name__ == "__main__":
    main()
