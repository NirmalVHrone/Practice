import dspy
import dotenv
import os
from pydantic import BaseModel, Field

dotenv.load_dotenv(dotenv_path="/Users/nirmal.vatsyayan/Desktop/Nirmal/ENVIRONMENT-CRED/.env", override=True, verbose=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


llm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=llm)
#dspy.settings.configure(lm=llm, track_usage=True)

class AnswerConfidence(BaseModel):
    answer: str = Field(description="Answer to the question")
    confidence: float = Field(description="Confidence in the answer")

class QAwithConfidence(dspy.Signature):
    question: str = dspy.InputField(description="Question to get answer")
    answer_confidence: AnswerConfidence = dspy.OutputField(description="Answer to the question with confidence")

predict = dspy.ChainOfThought(QAwithConfidence)

output  = predict(question="Who was the top scorer in the 2011 cricket world cup finals?")
print(output.answer_confidence.answer)
print(output.answer_confidence.confidence)

print(llm.inspect_history())