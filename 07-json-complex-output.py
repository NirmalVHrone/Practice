import dspy
import dotenv
import os
from pydantic import BaseModel, Field
dotenv.load_dotenv(dotenv_path="/Users/nirmal.vatsyayan/Desktop/Nirmal/ENVIRONMENT-CRED/.env", override=True, verbose=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=llm)
#dspy.settings.configure(lm=llm, track_usage=True)

class MetaData(BaseModel):
    population: str = Field(description="Population of the country")
    area: str = Field(description="Area of the country")

class AnswerConfidence(BaseModel):
    country: str = Field(description="Country name")
    capital: str = Field(description="Capital of the country")
    year: int = Field(description="Year of the cricket world cup")
    meta_data: MetaData = Field(description="Metadata of the country")


class QAList(dspy.Signature):
    question: str = dspy.InputField(description="Question to get answer")
    answer_list: list[AnswerConfidence] = dspy.OutputField(description="Answer to the question")

predict = dspy.ChainOfThought(QAList)

output = predict(question="Share details of countries who won cricket world cup?")
print(output.answer_list)
#print(llm.inspect_history())