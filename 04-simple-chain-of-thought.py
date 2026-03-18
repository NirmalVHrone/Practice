import dspy
import dotenv
import os

dotenv.load_dotenv(dotenv_path="/Users/nirmal.vatsyayan/Desktop/Nirmal/ENVIRONMENT-CRED/.env", override=True, verbose=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=llm)
#dspy.settings.configure(lm=llm, track_usage=True)

class QA(dspy.Signature):
    '''Return answer to the question'''
    question: str = dspy.InputField(description="Question to get answer")
    reasoning: str = dspy.OutputField(description="Reasoning to get answer")
    answer: str = dspy.OutputField(description="Answer to the question in short summary")

answer_generator = dspy.ChainOfThought(QA)

outputs = answer_generator(question="Oldest cricket stadiums in Asia?")
print(outputs.answer)
print("\n==========\n")
print(outputs.reasoning)
print("\n==========\n")
#print(outputs.get_lm_usage())
print(llm.inspect_history())