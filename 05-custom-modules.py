import dspy
import dotenv
import os

dotenv.load_dotenv(dotenv_path="/Users/nirmal.vatsyayan/Desktop/Nirmal/ENVIRONMENT-CRED/.env", override=True, verbose=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=llm)
#dspy.settings.configure(lm=llm, track_usage=True)

class DoubleChainOfThought(dspy.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_generator = dspy.ChainOfThought("question -> step_by_step_thought")
        self.reasoning_generator = dspy.ChainOfThought("question, thought -> one_word_answer")
    
    def forward(self, question: str):
        thought = self.answer_generator(question=question).step_by_step_thought
        answer = self.reasoning_generator(question=question, thought=thought).one_word_answer
        return dspy.Prediction(thought=thought, answer=answer)


doubleCOT = DoubleChainOfThought()

output = doubleCOT(question="What was the birth city of man of the match in 2011 cricket world cup finals?")
print(output)
