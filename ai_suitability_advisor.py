"""AI Suitability Advisor: decide whether business requirements need AI and what type."""
from dataclasses import dataclass
from typing import List, Dict, Any
import csv



# AI is the broad field of building systems that perform tasks associated with intelligence.
# ML learns patterns from examples/data rather than relying only on hand-written rules.
# Deep Learning is a subset of ML that uses multi-layer neural networks; it is useful
# for complex perception tasks such as image defect detection.
# Generative AI creates new content (text, images, etc.) and is useful for summaries,
# conversational answers, and drafting. Not every automation needs AI.


@dataclass
class Requirement:
    id: int
    title: str
    description: str
    fixed_rules_possible: bool
    needs_learning_from_data: bool
    unstructured_input: bool
    needs_generated_content: bool
    data_available: bool
    complex_visual_task: bool = False
    needs_trend_or_aggregation: bool = False



REQUIREMENTS: List[Requirement] = [
    Requirement(1, "Validate PAN number format", "Check PAN against its expected character pattern.", True, False, False, False, True),
    Requirement(2, "Predict customer cancellations", "Predict which customers are likely to cancel next month.", False, True, False, False, True),
    Requirement(3, "Summarise support emails", "Create concise summaries of long support emails.", False, True, True, True, True),
    Requirement(4, "Detect defects in product photos", "Identify visual defects from product images.", False, True, True, False, True, complex_visual_task=True),
    Requirement(5, "Calculate GST on an invoice", "Calculate tax using fixed rates and arithmetic.", True, False, False, False, True),
    Requirement(6, "Forecast weekly product demand", "Forecast future demand from historical sales.", False, True, False, False, True),
    Requirement(7, "Answer questions from HR policy PDF", "Answer employee questions grounded in an HR policy document.", False, True, True, True, True),
    Requirement(8, "Sort invoices by amount", "Sort invoices by numeric amount.", True, False, False, False, True),
    Requirement(9, "Create sales performance dashboard", "Aggregate sales and display trends by region and period.", False, False, False, False, True, needs_trend_or_aggregation=True),
    Requirement(10, "Classify customer feedback sentiment", "Label feedback as positive, neutral, or negative.", False, True, True, False, True),
]


AI_CATEGORIES = {"Analytics/BI", "Classical ML", "Deep Learning", "Generative AI"}


def decide(req: Requirement) -> Dict[str, Any]:
    """Return category and reasons. Rule-based checks take priority over AI."""
    reasons: List[str] = []
    if req.fixed_rules_possible and not req.needs_learning_from_data and not req.needs_generated_content:
        category = "No AI (rules/code)"
        reasons.append("A deterministic rule or calculation can solve this reliably.")
    elif req.needs_generated_content:
        category = "Generative AI"
        reasons.append("The requirement needs new natural-language content or a contextual answer.")
        if req.unstructured_input:
            reasons.append("The input is unstructured text/document content.")
    elif req.complex_visual_task:
        category = "Deep Learning"
        reasons.append("Complex image understanding is well suited to neural-network vision models.")
        reasons.append("Training or evaluation needs labeled examples of normal and defective images.")
    elif req.needs_trend_or_aggregation and not req.needs_learning_from_data:
        category = "Analytics/BI"
        reasons.append("Aggregating historical metrics and showing trends is analytics, not prediction.")
    elif req.needs_learning_from_data:
        category = "Classical ML"
        reasons.append("The task needs patterns learned from historical examples to predict or classify.")
        if req.data_available:
            reasons.append("Relevant historical data is marked as available for an initial model.")
        else:
            reasons.append("Data collection and labeling are prerequisites before model training.")
    else:
        category = "No AI (rules/code)"
        reasons.append("No learning, generation, or complex perception need is specified; start with code.")
    if req.data_available:
        reasons.append("Data availability is marked yes.")
    else:
        reasons.append("Data availability is marked no; gather data before implementation.")
    return {"category": category, "reasons": reasons}


LIFECYCLE: Dict[str, Dict[str, tuple]] = {
    "Analytics/BI": {
        "Problem framing": ("Agree on KPIs and dashboard audience.", "Product owner"),
        "Data": ("Connect sales tables and validate date/region fields.", "Data engineer"),
        "Model": ("Define aggregation logic and trend calculations.", "BI analyst"),
        "Deployment": ("Publish dashboard with access controls.", "BI developer"),
        "Monitoring": ("Track refresh failures and KPI consistency.", "BI analyst"),
    },
    "Classical ML": {
        "Problem framing": ("Define target label, prediction horizon, and success metric.", "Product owner"),
        "Data": ("Extract historical records, clean features, and create train/validation splits.", "Data engineer"),
        "Model": ("Train and compare baseline models; evaluate against business metric.", "Data scientist"),
        "Deployment": ("Package model behind a scheduled job or API.", "ML engineer"),
        "Monitoring": ("Monitor drift, prediction quality, and business outcomes.", "ML engineer"),
    },
    "Deep Learning": {
        "Problem framing": ("Define defect types and acceptable false-negative rate.", "Product owner"),
        "Data": ("Collect, label, and split product photos by defect type.", "Data engineer / domain expert"),
        "Model": ("Fine-tune or train a vision model and evaluate on held-out photos.", "Deep learning engineer"),
        "Deployment": ("Deploy image inference to inspection workflow.", "ML engineer"),
        "Monitoring": ("Review missed defects and image-quality/model drift.", "Quality engineer"),
    },
    "Generative AI": {
        "Problem framing": ("Define answer format, accuracy expectations, and guardrails.", "Product owner"),
        "Data": ("Prepare approved documents and remove sensitive data where needed.", "Data engineer / domain expert"),
        "Model": ("Test prompts and retrieval grounding; evaluate outputs for factuality.", "AI engineer"),
        "Deployment": ("Integrate model with app and source citations/permissions.", "AI / software engineer"),
        "Monitoring": ("Review quality, unsafe outputs, latency, cost, and user feedback.", "AI engineer"),
    },
}


RISKS = {
    "No AI (rules/code)": "Incorrect rules, edge cases, or invalid input handling.",
    "Analytics/BI": "Stale or inconsistent source data can mislead decisions.",
    "Classical ML": "Bias, data leakage, drift, and inaccurate predictions.",
    "Deep Learning": "Label quality, missed defects, bias across image conditions, and compute cost.",
    "Generative AI": "Hallucinations, privacy leakage, prompt injection, and output variability.",
}


FIRST_STEPS = {
    "No AI (rules/code)": "Write the rule, validate edge cases, and implement unit tests.",
    "Analytics/BI": "Confirm KPI definitions and source tables with stakeholders.",
    "Classical ML": "Define the target and inspect historical data quality and label availability.",
    "Deep Learning": "Collect representative photos and define a defect-labeling guide.",
    "Generative AI": "Gather approved source content and create a small evaluation question set.",
}



def lifecycle_for(category: str, title: str) -> List[Dict[str, str]]:
    if category not in AI_CATEGORIES:
        return []
    stages = LIFECYCLE[category]
    result = []
    for stage, (task, role) in stages.items():
        result.append({"requirement": title, "stage": stage, "task": task, "role": role})
    return result



def build_report(requirements: List[Requirement] = REQUIREMENTS) -> List[Dict[str, Any]]:
    rows = []
    for req in requirements:
        decision = decide(req)
        category = decision["category"]
        rows.append({
            "id": req.id,
            "requirement": req.title,
            "decision": category,
            "reasons": " | ".join(decision["reasons"]),
            "risks": RISKS[category],
            "first_next_step": FIRST_STEPS[category],
        })
    return rows



def write_csv_report(path: str = "ai_suitability_report.csv", requirements: List[Requirement] = REQUIREMENTS) -> None:
    rows = build_report(requirements)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_lifecycle_markdown(path: str = "lifecycle_map.md", requirements: List[Requirement] = REQUIREMENTS) -> None:
    lines = ["# AI Project Lifecycle Map", "", "Lifecycle is provided for AI-suitable requirements only.", ""]
    for req in requirements:
        decision = decide(req)["category"]
        if decision not in AI_CATEGORIES:
            continue
        lines.extend([f"## {req.id}. {req.title} — {decision}", "", "| Stage | Concrete task | Responsible role |", "|---|---|---|"])
        for item in lifecycle_for(decision, req.title):
            lines.append(f"| {item['stage']} | {item['task']} | {item['role']} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def run_tests() -> None:
    """Small assert-based checks for obvious routing cases."""
    by_title = {r.title: decide(r)["category"] for r in REQUIREMENTS}
    assert by_title["Calculate GST on an invoice"] == "No AI (rules/code)"
    assert by_title["Validate PAN number format"] == "No AI (rules/code)"
    assert by_title["Detect defects in product photos"] == "Deep Learning"
    assert by_title["Summarise support emails"] == "Generative AI"
    assert by_title["Create sales performance dashboard"] == "Analytics/BI"
    # Input-validation test: unknown requirement type still returns a safe default.
    unknown = Requirement(99, "Unknown task", "", False, False, False, False, False)
    assert decide(unknown)["category"] == "No AI (rules/code)"


def main() -> None:
    run_tests()
    report = build_report()
    write_csv_report()
    write_lifecycle_markdown()
    print("AI Suitability Advisor — tests passed")
    print(f"Analyzed {len(report)} requirements. Reports written: ai_suitability_report.csv, lifecycle_map.md")
    print("\nRequirement decisions:")
    for row in report:
        print(f"- {row['requirement']}: {row['decision']}")


if __name__ == "__main__":
    main()
