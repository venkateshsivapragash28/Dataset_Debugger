import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.env_loader import GOOGLE_API_KEY
from tools.structure_analyzer import df_to_column_samples
from tools.loader import load_data


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


SYSTEM_PROMPT = """
You are a dataset column label validation agent.

You will receive a dictionary:

{
 "column_name": [sample_values],
 "column_name": [sample_values]
}

Your job is to detect ALL column label inconsistencies.

Detect these issue types:

missing_name
noise
mismatch
swap
duplicate
ambiguous
inconsistent_style
reserved_name

Return ONLY JSON.

Output schema:

[
 {
  "column": "<original column name>",
  "issue": "<issue type>",
  "action": "rename | review | none",
  "suggested_name": "<suggested column name>",
  "confidence": 0-1,
  "reason": "<short explanation>"
 }
]

Rules:
- If column is correct → issue = "correct", action="none"
- Return a result for EVERY column
- Only return valid JSON
"""


def validate_dataset(column_samples):

    prompt = f"""
Dataset Columns and Sample Values:

{json.dumps(column_samples, indent=2)}

Return a validation result for EVERY column in the dataset.
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])

    content = response.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        llm_report = json.loads(content)

        if not isinstance(llm_report, list):
            llm_report = []

    except Exception:
        llm_report = []

    # Convert LLM output into dictionary
    llm_map = {}

    for item in llm_report:
        if isinstance(item, dict) and "column" in item:
            llm_map[item["column"]] = item

    final_report = []

    for column in column_samples.keys():

        if column in llm_map:

            item = llm_map[column]

            # Ensure required fields exist
            item.setdefault("issue", "correct")
            item.setdefault("action", "none")
            item.setdefault("suggested_name", column)
            item.setdefault("confidence", 0.9)
            item.setdefault("reason", "")

            final_report.append(item)

        else:

            final_report.append({
                "column": column,
                "issue": "correct",
                "action": "none",
                "suggested_name": column,
                "confidence": 1.0,
                "reason": "no label issues detected"
            })

    return final_report


def generate_label_report(file_path):

    df = load_data(file_path)[0]

    column_samples = df_to_column_samples(df)

    report = validate_dataset(column_samples)

    return report


if __name__ == "__main__":

    file_path = r"C:\Users\VENKAT\Desktop\GRIND\Dataset_Debugger\Automobile_data.csv"

    report = generate_label_report(file_path)

    print("\n===== COLUMN LABEL INCONSISTENCY REPORT =====\n")

    for r in report:

        print(f"column : {r['column']}")
        print(f"issue : {r['issue']}")
        print(f"suggested name : {r['suggested_name']}")
        print(f"action : {r['action']}")
        print(f"confidence : {r['confidence']}")
        print(f"reason : {r['reason']}\n")
