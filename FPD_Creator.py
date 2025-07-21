import os
import re
import csv
import pdfplumber
from openai import OpenAI

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

PDF_FILE_PATH = ""
OUTPUT_DIR    = ""

# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts and returns all text from the given PDF file."""
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                full_text.append(txt)
    return "\n".join(full_text)

def call_gpt(messages, model="o4-mini") -> str:
    """Helper to call OpenAI chat completion and return the content."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

def parse_numbered_list(text: str) -> list[str]:
    """
    Parses lists like:
      1. Alice
      2 Bob
    and returns ['Alice','Bob'].
    """
    items = []
    for line in text.splitlines():
        # allow "1. Foo" or "1 Foo"
        m = re.match(r"^\s*\d+\.?\s+(.+)$", line)
        if m:
            items.append(m.group(1).strip())
    return items

def parse_markdown_table(md: str) -> list[dict]:
    """Parses a Markdown table into a list of dicts."""
    lines = [l.strip() for l in md.splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for row in lines[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows

def write_csv(rows: list[dict], headers: list[str], out_path: str):
    """Writes rows (list of dicts) to a CSV with given headers."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ------------------------------------------------------------------------------
# GPT PROMPT TEMPLATES (refactored)
# ------------------------------------------------------------------------------

def get_participants_prompt(sop_text: str) -> list[dict]:
    system = "You are a process modelling expert."
    user   = (
        "Identify the main participants that execute the processes mentioned the Standard Operating Procedure given below.\n"
        "Return a simple, single-level numbered list (e.g. “1. Alice”, “2. Bob”), with no extra text using only alphanumeric characters and spaces (no special characters):\n\n"
        "```\n"
        f"{sop_text}\n"
        "```"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

def get_bpmn_prompt(participant: str, sop_text: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are an expert in BPMN 2.0 process modelling."},
        {
            "role": "user",
            "content": (
                f"As the participant **{participant}** trying to understand what you need to do within this standard operating procedure and how you collaborate with the other participants based on the Standard Operating Procedure given below. Create a single Formalised Process Data table using the BPMN 2.0 Standard relating to all processes that you participate in, ensuring that all steps are represented."
                "The tasks must fit within a single swim lane with a single start event."
                "The result must be given in Markdown format, as a table with exactly these columns(nothing additional):\n\n"
                "| Object Type         | Object Name                           | Predecessor                           | Successor                                                               | Input Data                                    | Output Data                                  | Additional Information                                                |\n"
                "| ------------------- | ------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------- |\n"
                "• Start Event         | Start Condition                       | N/A                                   | Name of next Object                                                     | N/A                                           | N/A                                          |                                                                       |\n"
                "• Receive Task        | Task X (where a message is recieved)  |                                       | Name of next Object                                                     |                                               | Name of Data Object 1; Name of Data Object 2 |                                                                       |\n"
                "• Gateway (Exclusive) | Condition for splitting sequence flow |                                       | Name of next Object (Condition 1) OR Name of next Object (Condition 2)  |                                               |                                              |                                                                       |\n"
                "• Send Task           | Task Y (where a message is sent)      | Only Object Name of Exclusive Gateway |                                                                         | Name of Data Object 1                         | Name of Data Object 3                        |                                                                       |\n"
                "• Task                | Task Z                                | Only Object Name of Exclusive Gateway |                                                                         | Name of Data Object 1                         | Name of Data Object 4                        | Additional information if any (special conditions, constraints, etc.) |\n"
                "• Gateway (Parallel)  | Parallel Gateway name                 |                                       | Name of Successor 1 AND Name of Successor 2.                            |                                               |                                              |                                                                       |\n"
                "• Task                | Task A                                | Only Object Name of Parallel Gateway  |                                                                         |                                               |                                              |                                                                       |\n"
                "• Task                | Task B                                | Only Object name of Parallel Gateway  |                                                                         | Name of Data Object 2; Name of Data Object 3  |                                              |                                                                       |\n"
                "• Intermediate Event  | Event Name                            | Task B                                | Task C                                                                  | N/A                                           | N/A                                          |                                                                       |\n"
                "• Task                | Task C                                | Event Name                            | End Condition                                                           | Name of Data Object 2                         |                                              |                                                                       |\n"
                "• End Event           | End Condition                         |                                       | N/A                                                                     | N/A                                           | N/A                                          |                                                                       |\n\n"
                "Use **only** these column headers and rows that reflect your own processes – no extra section headings, labels or comments.  \n\n"
                "```\n"
                f"{sop_text}\n"
                "```"
            )
        }
    ]


def get_message_flows_prompt(bpmn_tables: dict[str, str]) -> list[dict]:
    # System prompt stays the same
    system_content = (
        "You are a process modeling expert trying to understand interactions between participants. Given below are markdown tables which specify the send and receive tasks as well as the associated data for each participant."
        "Analyze all the participant’s sending and receiving tasks to identify all the message flows between them."
        "Return a single Markdown table with exactly these columns:\n\n"
        "| Message Sent             | Sending Task            | Sending Participant          | Receiving Task          | Receiving Participant         |\n"
        "| ------------------------ | ----------------------- | ---------------------------- | ----------------------- | ----------------------------- |\n"
        "| Name of Message Sent     | Name of Send Task       | Name of Sending Participant  | Name of Receive Task    | Name of Receiving Participant |\n\n"
        "Only list Send → Receive flows that logically belong together."
    )

    # Build user content by extracting only Activity rows and three columns
    user_sections = []
    for participant, md_table in bpmn_tables.items():
        # parse the full table
        rows = parse_markdown_table(md_table)
        # keep only activities
        activities = [r for r in rows if r.get("Object Type") in ("Send Task","Receive Task")]

        # build a tiny markdown table with Task Type, Task Name, Output Data
        mini_table = (
            "| Task Type | Task Name | Output Data | Input Data |\n"
            "| --------- | --------- | ----------- | ---------- |\n"
        )

        for act in activities:
            task_type   = act.get("Object Type", "").strip()
            task_name   = act.get("Object Name", "").strip()
            output_data = act.get("Output Data","").strip() or "—"
            input_data = act.get("Input Data","").strip() or "—"
            mini_table += f"| {task_type} | {task_name} | {output_data} | {input_data} |\n"

        user_sections.append(
            f"Tasks for {participant}:\n\n{md_table}\n"
        )

    user_content = "\n\n".join(user_sections)

    return [
        {"role": "system",  "content": system_content},
        {"role": "user",    "content": user_content},
    ]



# ------------------------------------------------------------------------------
# MAIN PROCESS
# ------------------------------------------------------------------------------

def main(pdf_path: str, output_dir: str):
    print(f"Reading PDF from {pdf_path} ...")
    sop_text = extract_text_from_pdf(pdf_path)
    if not sop_text.strip():
        print("No text extracted from PDF.")
        return

    # Participants
    print("Identifying main participants with GPT ...")
    resp = call_gpt(get_participants_prompt(sop_text))
    participants = parse_numbered_list(resp)
    if not participants:
        print("Failed to parse any participants.")
        print("Raw GPT response was:\n", resp)
        return

    print("Found participants:", participants)

    # BPMN per participant
    headers = ["Object Type", "Object Name", "Predecessor", "Successor", "Input Data", "Output Data", "Additional Information"]
    bpmn_tables: dict[str, str] = {}
    for p in participants:
        print(f"\nGenerating BPMN data for '{p}' ...")
        md = call_gpt(get_bpmn_prompt(p, sop_text))
        rows = parse_markdown_table(md)
        if not rows:
            print(f"Could not parse table for {p}. Raw response:\n{md}")
            continue
        safe = re.sub(r"[^\w\-]+", "_", p)
        out = os.path.join(output_dir, f"{safe}_bpmn.csv")
        write_csv(rows, headers, out)
        print(f"{len(rows)} rows → {out}")
        bpmn_tables[p] = md

    # Message flows
    if bpmn_tables:
        print("\nGenerating message flows ...")
        mf = call_gpt(get_message_flows_prompt(bpmn_tables))
        mf_rows = parse_markdown_table(mf)
        if mf_rows:
            mf_headers = [
                "Message Sent",
                "Sending Task",
                "Sending Participant",
                "Receiving Task",
                "Receiving Participant"
            ]
            out_msg = os.path.join(output_dir, "message_flows.csv")
            write_csv(mf_rows, mf_headers, out_msg)
            print(f"{len(mf_rows)} flows → {out_msg}")
        else:
            print("Could not parse message flows. Raw response:\n", mf)

    print("\nAll done.")

if __name__ == "__main__":
    main(PDF_FILE_PATH, OUTPUT_DIR)
