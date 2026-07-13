import dspy
from dspy.teleprompt import BootstrapFewShot
from agents import writer, WriteContent

# --- TRAINING DATA (The "Brain" of the Agency) ---

train_data = [
    # EXAMPLE 1: Payroll Automation (Focus: Compliance & Speed vs. Manual/Legacy)
    dspy.Example(
        topic="Automating Indian Payroll & Compliance",
        outline="1. The headache of month-end processing. 2. Statutory compliance risks. 3. The HrOne advantage. 4. Real-world impact.",
        brand_name="HrOne",
        industry_example="The 2023 shift where 60% of Indian mid-caps moved to cloud payroll.",
        client_example="Haier Appliances reduced payroll processing time by 4 days.",
        negative_keywords="Cheap, Free tool, Manual excel, complicated setup",
        content_draft="""
        Month-end payroll shouldn't feel like a battlefield. Yet, for many HR teams, it's a cycle of spreadsheet chaos and compliance anxiety.
        
        While industry shifts show that 60% of Indian mid-caps are moving to the cloud, many still get stuck with software that is pretty to look at but weak on core logic. 
        Unlike platforms like Keka, which are known for their UI, HrOne prioritizes the depth of Indian statutory compliance. We don't just calculate salaries; we lock down compliance.
        
        Take our work with Haier Appliances as a prime example. Before automating, their team spent a week reconciling data. After switching to HrOne’s auto-lock mechanism, they reduced payroll processing time by 4 full days.
        
        It’s not just about paying people; it’s about the peace of mind that comes from knowing every PF and ESI deduction is accurate to the decimal.
        """
    ).with_inputs("topic", "outline", "brand_name", "industry_example", "client_example", "negative_keywords"),

    # EXAMPLE 2: Performance Management (Focus: Simplicity vs. Enterprise Complexity)
    dspy.Example(
        topic="Continuous Feedback over Annual Reviews",
        outline="1. Why annual reviews are dead. 2. The need for real-time nudges. 3. Simplifying the process.",
        brand_name="HrOne",
        industry_example="Microsoft's abandonment of stack ranking systems.",
        client_example="Lenskart improved employee engagement scores by 15%.",
        negative_keywords="Bureaucracy, paper-trail, complex configuration, punishment",
        content_draft="""
        The era of the 'scary annual review' is over. Companies are realizing that waiting 12 months to give feedback is a recipe for stagnation. 
        We saw this when giants like Microsoft abandoned stack ranking—it simply kills morale.
        
        However, the solution isn't to buy a complex enterprise suite. Tools like Darwinbox offer incredible depth for massive conglomerates, but for agile teams, that depth often turns into administrative clutter.
        HrOne takes a different approach: 'Simple is the new Smart.'
        
        We strip away the jargon. We enabled Lenskart to switch from heavy annual processes to lightweight, mobile-first check-ins. The result? A 15% jump in engagement scores.
        Your employees live on their phones; your performance system should too.
        """
    ).with_inputs("topic", "outline", "brand_name", "industry_example", "client_example", "negative_keywords"),

    # EXAMPLE 3: Onboarding Experience (Focus: Workflow vs. Global Tools)
    dspy.Example(
        topic="The First 90 Days: Remote Onboarding",
        outline="1. The drop-off rate of new hires. 2. Asset allocation & documentation. 3. The human touch in digital onboarding.",
        brand_name="HrOne",
        industry_example="The rise of 'Ghosting' by candidates in the IT sector.",
        client_example="Muthoot Fincorp digitized 100% of their joining document collection.",
        negative_keywords="Paperwork, physical files, delay, confusion",
        content_draft="""
        Candidate ghosting is at an all-time high in the IT sector. If your onboarding process is slow, you lose talent before day one.
        
        Global tools like Rippling are fantastic for IT asset management across borders. But when it comes to the nitty-gritty of Indian joining formalities—Aadhar verification, family nominations, and local policy acceptance—you need a localized expert.
        
        HrOne bridges this gap. We turn the first day into a celebration, not a paperwork drill.
        For instance, Muthoot Fincorp used our platform to digitize 100% of their document collection. New hires finished their paperwork before they even stepped into the office (or logged onto Zoom).
        """
    ).with_inputs("topic", "outline", "brand_name", "industry_example", "client_example", "negative_keywords"),
    
    # EXAMPLE 4: Attendance & Leave (Focus: Mobile-First vs. Legacy Biometric)
    dspy.Example(
        topic="Geo-fencing and Mobile Attendance",
        outline="1. The death of the biometric machine. 2. Trust vs. Tracking. 3. Mobile integration.",
        brand_name="HrOne",
        industry_example="Post-COVID hybrid work adoption rates.",
        client_example="GenericPharma handled 500+ field sales staff attendance seamlessly.",
        negative_keywords="Punch clock, strict monitoring, surveillance, hardware cost",
        content_draft="""
        The biometric machine on the wall is becoming a relic of the past. With hybrid work adoption soaring post-COVID, you can't tether attendance to a physical location anymore.
        
        While some competitors push for rigid desktop monitoring, HrOne empowers the field workforce. 
        Imagine a sales rep for GenericPharma. They don't come to the office; they go to the doctor's clinic.
        Using HrOne's geo-fencing, they mark attendance from the client site. It validates the location without being invasive.
        
        This balances trust with accountability, ensuring you get accurate data without treating your workforce like robots.
        """
    ).with_inputs("topic", "outline", "brand_name", "industry_example", "client_example", "negative_keywords"),
]

# --- OPTIMIZATION ROUTINE ---

def validate_brand_usage(example, pred, trace=None):
    # Logic: Score high if the brand and examples are mentioned naturally
    score = 0
    if example.brand_name.lower() in pred.content_draft.lower():
        score += 1
    if example.client_example.split()[0].lower() in pred.content_draft.lower(): # Check for client name
        score += 1
    return score >= 2 # Pass if both are present

print(f"--- Loaded {len(train_data)} High-Quality Training Examples for HrOne ---")
print("--- Optimizing Writer Module (This teaches Claude your specific Tone) ---")

# We use a higher number of bootstraps to ensure it explores enough variations
teleprompter = BootstrapFewShot(
    metric=validate_brand_usage, 
    max_bootstrapped_demos=2,
    max_labeled_demos=4 
)

compiled_writer = teleprompter.compile(writer, trainset=train_data)

compiled_writer.save("writer_optimized.json")
print("Optimization Complete! Saved to 'writer_optimized.json'")