import os
import json
import re
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Replace 'your_api_key_here' with your actual Groq API key
api_key = 'USE_YOUR_OWN_API_KEY'

# Initialize Groq client with API key
client = Groq(api_key=api_key)

def submission_prediction(submission_text: str):
    """
    Function to get predictions for a given submission text using Groq API.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert psychologist who helps some developers to collect clean mental disorder
                 patient data. Your mission is filtering and trimming these data.
                                You should respond the question in specified JSON format.""",
            },
            {
                "role": "user",
                "content": f"""
                        For the following subreddit submission text (enclosed in triple backticks), analyze whether it 
                        describes the stories, feelings, or behaviors of a patient with a mental disorder. Respond according to these guidelines:

                        1. If the text describes the stories, feelings, or behaviors of a patient with a mental disorder,
                         trim important part of the text without any lexical change and return following
                        JSON format (you should fill the Text key with trimmed or normal version of the text):
                            {{"Result":"yes", "Text": ""}}

                        2. If the text does not describe such stories, feelings, or behaviors, respond with the following JSON format:
                            {{"Result":"no", "Text": "<write here given_input_text>"}}

                        \n\n
                        Subreddit Submission Text: ```{submission_text}```
                        """
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

def filter_using_llm(df: pd.DataFrame) -> pd.DataFrame:
    """
    """
    wrong_error = 0
    filtered_llm_result_list = []

    for entry in tqdm(df['Entry'].unique(), desc="LLM filtering"):
        try:
            response_text = submission_prediction(submission_text=entry)


            pattern = re.compile(r'\{.*?\}', re.DOTALL)
            match = pattern.search(response_text)

            if match:
                response_json = json.loads(match.group(0))


                if response_json['Result'] == 'yes':
                    response_json['Title'] = df[df['Entry'] == entry]['Title'].values[0]
                    filtered_llm_result_list.append(response_json)
                else:
                    response_json['Title'] = df[df['Entry'] == entry]['Title'].values[0]
                    filtered_llm_result_list.append(response_json)
            else:
                wrong_error += 1

        except Exception as err:
            print(err)
            wrong_error += 1

    print(f"LLM filtering process was finished. The error ratio is: %", round((wrong_error/len(df['Entry'].unique())*100)))

    return pd.DataFrame(filtered_llm_result_list)


def main():

    input_file = 'eksisozluk_entries1.csv'
    df = pd.read_csv(input_file)


    filtered_df = filter_using_llm(df)


    output_file = 'filtered_entries1.csv'
    filtered_df.to_csv(output_file, index=False)
    print(f'Filtering complete. Filtered entries saved to {output_file}')

if __name__ == "__main__":
    main()
