import os
import dspy
from dotenv import load_dotenv

load_dotenv()
# Setup LLM (Using GPT-4o or similar high-context model recommended)
lm = dspy.LM("openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
#dspy.configure(lm=lm)

# Define Claude (The Creative Writer)
claude = dspy.LM(
    model="anthropic/claude-sonnet-4-20250514", # Latest Sonnet model
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=4000
)

# Define GPT-4o (The Logical Planner - Optional)
# If you don't have OpenAI, just replace 'gpt4' with 'claude' in the modules below.
gpt4 = dspy.LM(
    model="openai/gpt-4o", 
    api_key=os.getenv("OPENAI_API_KEY")
)

# Set Default (We use Claude as default for safety)
dspy.configure(lm=claude)

# --- COMPLEX SIGNATURES ---

class SEOPlan(dspy.Signature):
    """
    Create a content outline based on SEO constraints, competitor analysis, and target intent.
    Allocate estimated word counts to each section to hit the total target.
    """
    topic = dspy.InputField()
    content_type = dspy.InputField(desc="Blog, Product Page, or Homepage")
    primary_keywords = dspy.InputField()
    secondary_keywords = dspy.InputField()
    target_prompts = dspy.InputField(desc="Search intents to rank for")
    competitor_analysis = dspy.InputField(desc="Insights from competitors to beat")
    target_word_count = dspy.InputField()
    
    outline = dspy.OutputField(desc="Structured outline with H2/H3 headers and word count allocation.")

class WriteContent(dspy.Signature):
    """
Write full content based on an outline. 
MUST integrate specific Industry and Client examples naturally.
MUST adhere to the Brand Name and avoid Negative Keywords.

You are an expert human blogger writing content that must pass AI detection tools. Write in a way that is completely indistinguishable from human writing. Follow these rules STRICTLY:

1. FORBIDDEN AI PHRASES - NEVER USE THESE:
    - "In today's world" / "In today's digital age" / "In the modern era"
    - "It is important to note that" / "It's worth noting" / "It should be noted"
    - "On the other hand" / "However, it is" / "That said"
    - "In conclusion" / "To sum up" / "In summary"
    - "Let's dive into" / "Let's explore" / "Let's take a look"
    - "First and foremost" / "Last but not least"
    - "At the end of the day" / "When all is said and done"
    - "It goes without saying" / "Needless to say"
    - "In order to" (use "to" instead)
    - "As a result" / "Therefore" / "Hence" (use sparingly, prefer "so" or "because")
    - "Furthermore" / "Moreover" / "Additionally" (use "also" or "plus" or just continue)
    - "In other words" / "To put it another way"
    - "With that in mind" / "Keeping this in mind"
    - "It's no secret that" / "It's well-known that"
    - "The fact of the matter is" / "The truth is"
    - "Without further ado" / "Without a doubt"
    - "In a nutshell" / "Bottom line"
    - "Make no mistake" / "Rest assured"
    - "By and large" / "For the most part"
    - "All things considered" / "All in all"

2. HUMAN WRITING PATTERNS - USE THESE INSTEAD:
    - Start with a specific observation or story, not a generalization
    - Use contractions naturally (don't, can't, won't, it's)
    - Mix sentence structures: fragments, questions, exclamations, run-ons
    - Use "I" or "we" when appropriate to show personal experience
    - Include rhetorical questions
    - Use casual connectors: "but", "so", "and", "or", "because", "though"
    - Add parenthetical thoughts (like this)
    - Use ellipses... for natural pauses
    - Include specific numbers, dates, names without over-explaining
    - Show uncertainty: "maybe", "perhaps", "I think", "seems like"

3. SENTENCE VARIETY - CRITICAL:
    - Mix 5-word sentences with 25-word sentences
    - Use one-word sentences. Occasionally.
    - Start sentences with "And", "But", "Or", "So" (humans do this)
    - End some sentences with prepositions (natural speech)
    - Use dashes—for emphasis—like this
    - Vary paragraph length: some 1 sentence, some 5 sentences

4. VOICE & PERSONALITY:
    - Show your opinion, not just facts
    - Use specific examples, not generic ones
    - Include small imperfections: typos are bad, but slight informality is good
    - Show emotion: frustration, excitement, skepticism, curiosity
    - Use analogies from everyday life
    - Reference specific moments, not abstract concepts

5. STRUCTURE - AVOID FORMULA:
    - Don't use numbered lists unless absolutely necessary
    - Don't use "Introduction → Body → Conclusion" structure
    - Don't repeat the topic in the first sentence
    - Don't use headings for every section
    - Let ideas flow naturally, not mechanically
    - Jump between related ideas without perfect transitions

6. VOCABULARY MIX:
    - Use simple words 70% of the time
    - Use complex words 30% of the time
    - Avoid repeating the same word in consecutive sentences
    - Use slang or casual terms when appropriate
    - Mix formal and informal language naturally

7. SPECIFIC WRITING TECHNIQUES:
    - Start mid-thought: "The problem? Most people don't realize..."
    - Use direct address: "You know what's interesting?"
    - Include asides: "—and this is the key part—"
    - Use incomplete thoughts: "Or maybe not. Who knows?"
    - Show thinking process: "Wait, let me reconsider that..."

8. ABSOLUTE PROHIBITIONS:
    - NO perfect parallel structure
    - NO balanced pros/cons lists
    - NO three-point explanations
    - NO "Here's why" or "Here's how" transitions
    - NO corporate speak or marketing jargon
    - NO over-explaining obvious points
    - NO summarizing what you just said

9. GOAL:
    Write as if you're a real person who has actually experienced this topic, has opinions about it, and is sharing insights in a natural, conversational way. The writing should feel like it came from a human brain, not an AI model.


    """
    topic = dspy.InputField()
    outline = dspy.InputField()
    brand_name = dspy.InputField()
    industry_example = dspy.InputField(desc="Real world industry case to mention")
    client_example = dspy.InputField(desc="Specific client success story to mention")
    negative_keywords = dspy.InputField(desc="List of words strictly forbidden")
    
    content_draft = dspy.OutputField(desc="The detailed written content.")

class HumanizeContent(dspy.Signature):
    """
    Refine content to remove AI detection markers and make it sound more human.
    Replace any AI phrases with natural alternatives. Add human imperfections, 
    vary sentence structure, and make the writing feel more conversational and personal.
    """
    original_content = dspy.InputField()
    
    humanized_content = dspy.OutputField(desc="The refined content that sounds completely human-written, with no AI detection markers.")

class GenerateFAQs(dspy.Signature):
    """Generate FAQs based on the content topic and SEO keywords."""
    topic = dspy.InputField()
    primary_keywords = dspy.InputField()
    faq_count = dspy.InputField(desc="Number of FAQs to generate")
    
    faq_section = dspy.OutputField(desc="FAQ section formatted with Schema markup concepts.")

# --- MODULES ---
planner = dspy.Predict(SEOPlan)
planner.set_lm(lm=gpt4)
writer = dspy.ChainOfThought(WriteContent) # We will train this one
writer.set_lm(lm=claude)
humanizer = dspy.Predict(HumanizeContent)
humanizer.set_lm(lm=claude)
faq_gen = dspy.Predict(GenerateFAQs)
faq_gen.set_lm(lm=claude)
