import dspy
import dotenv
import os

dotenv.load_dotenv(dotenv_path="/Users/nirmal.vatsyayan/Desktop/Nirmal/ENVIRONMENT-CRED/.env", override=True, verbose=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=llm, track_usage=True)

predict = dspy.Predict("question -> answer")

outputs = predict(question="What is the capital of India?")
print(outputs.answer)
print(outputs.get_lm_usage()) #if same call is made again, it will return empty because of cached usage

print(llm.inspect_history())