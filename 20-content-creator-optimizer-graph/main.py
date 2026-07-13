import os
import dspy
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents import planner, writer, faq_gen, humanizer


from dotenv import load_dotenv
load_dotenv("../.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# TO-DO REPLACE WITH GEPA WRITER
if os.path.exists("writer_optimized.json"):
    writer.load("writer_optimized.json")

# 1. DEFINE THE STATE (Maps to your 12 Inputs)
class SEOState(TypedDict):
    # Inputs
    topic: str
    primary_keywords: str
    secondary_keywords: str
    target_word_count: str
    target_prompts: str
    competitor_analysis: str
    brand_name: str
    negative_keywords: str
    industry_example: str
    client_example: str
    faq_preference: int
    content_type: str
    
    # Internal State
    outline: str
    draft_content: str
    final_output: str

# 2. DEFINE NODES

def plan_node(state: SEOState):
    print(f"--- 1. Strategizing for {state['brand_name']} ---")
    pred = planner(
        topic=state['topic'],
        content_type=state['content_type'],
        primary_keywords=state['primary_keywords'],
        secondary_keywords=state['secondary_keywords'],
        target_prompts=state['target_prompts'],
        competitor_analysis=state['competitor_analysis'],
        target_word_count=state['target_word_count']
    )
    return {"outline": pred.outline}

def write_node(state: SEOState):
    print("--- 2. Writing Content (Injecting Examples) ---")
    pred = writer(
        topic=state['topic'],
        outline=state['outline'],
        brand_name=state['brand_name'],
        industry_example=state['industry_example'],
        client_example=state['client_example'],
        negative_keywords=state['negative_keywords']
    )
    return {"draft_content": pred.content_draft}

def humanize_node(state: SEOState):
    print("--- 2.5. Humanizing Content (Removing AI Markers) ---")
    pred = humanizer(original_content=state['draft_content'])
    return {"draft_content": pred.humanized_content}

def faq_node(state: SEOState):
    if int(state['faq_preference']) <= 0:
        return {"final_output": state['draft_content']}
        
    print(f"--- 3. Generating {state['faq_preference']} FAQs ---")
    pred = faq_gen(
        topic=state['topic'],
        primary_keywords=state['primary_keywords'],
        faq_count=str(state['faq_preference'])
    )
    
    # Combine draft + FAQs
    full_text = f"{state['draft_content']}\n\n## Frequently Asked Questions\n{pred.faq_section}"
    return {"final_output": full_text}

def compliance_check_node(state: SEOState):
    # Simple Python logic to ensure Negative Keywords didn't slip in
    print("--- 4. Checking Negative Keywords ---")
    content = state['final_output']
    negatives = [k.strip() for k in state['negative_keywords'].split(',')]
    
    found_negatives = []
    for word in negatives:
        if word.lower() in content.lower():
            found_negatives.append(word)
            # Basic sanitization
            content = content.replace(word, "[REDACTED]")
            content = content.replace(word.capitalize(), "[REDACTED]")
    
    if found_negatives:
        print(f"WARNING: Removed forbidden words: {found_negatives}")
        
    return {"final_output": content}

# 3. BUILD GRAPH
workflow = StateGraph(SEOState)

workflow.add_node("planner", plan_node)
workflow.add_node("writer", write_node)
workflow.add_node("humanizer", humanize_node)
workflow.add_node("faq_gen", faq_node)
workflow.add_node("compliance", compliance_check_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "writer")
workflow.add_edge("writer", "humanizer")
workflow.add_edge("humanizer", "faq_gen")
workflow.add_edge("faq_gen", "compliance")
workflow.add_edge("compliance", END)

app = workflow.compile()

# 4. EXECUTE WITH YOUR 12 INPUTS
if __name__ == "__main__":
    inputs = {
        "topic": "Automating Indian Payroll & Compliance",
        "primary_keywords": "automation, payroll, compliance, indian payroll, indian compliance",
        "secondary_keywords": "efficiency, ROI, future of work, branding, marketing",
        "target_word_count": "200 words",
        "target_prompts": "How to automate indian payroll, Best AI tools 2024, AI in digital marketing",
        "competitor_analysis": "Keka focuses on automation features. Salesforce focuses on CRM data.",
        "brand_name": "HrONE",
        "negative_keywords": "cheap, free trial, scam, quick fix",
        "industry_example": "The 2023 shift where 60% of Indian mid-caps moved to cloud payroll.",
        "client_example": "Haier Appliances reduced payroll processing time by 4 days.",
        "faq_preference": 5,
        "content_type": "Blog"
    }
    
    result = app.invoke(inputs)
    
    print("\n================ FINAL GENERATED CONTENT ================\n")
    print(result['final_output'])