import google.generativeai as genai
from tqdm import tqdm
import re
import pandas as pd
import json
import time


genai.configure(api_key="AIzaSyDzn6ERBoFX0fp-s00L364BCQOnHFV4uO4")

model = genai.GenerativeModel('gemini-1.5-flash')


def submission_prediction(submission_text: str):
    response = model.generate_content("""
    You are an expert psychologist who helps some developers to collect clean mental disorder patient data. Your mission is filtering and trimming these data.
    You should respond the question in specified JSON format. 

    \n\n

    For the following subreddit submission text (enclosed in triple backticks), analyze whether it describes the stories (not recovery story), feelings (not recovered feelings), or behaviors 
    (not recovered feelings) of a patient with a mental disorder. Respond according to these guidelines:

    1. If the text describes the stories, feelings, or behaviors of a patient with a mental disorder, trim important part of the text without any lexical change and return following
    JSON format:
        {"Result":"yes", "Text": <trimmed or normal text>}

    2. If the text does not describe such stories, feelings, or behaviors, analyze it again:
        2.a. If the text has the potential to describe the stories, feelings, or behaviors of another patient, respond with the following JSON format:
        {"Result":"comment", "Text": <given_input_text>}

        2.b. If it does not, respond with the following JSON format:
        {"Result":"no", "Text": <given_input_text>}

    \n\n
    Subreddit Submission Text: """ + submission_text)
    return response.text


def comment_prediction(comment_text: str) -> str:
    """

    :param comment_text:
    :return:
    """
    response = model.generate_content("""
        You are an expert psychologist who helps some developers to collect clean mental disorder patient data. Your mission is filtering and trimming these data.
        You should respond the question in specified JSON format. 

        \n\n

        For the following subreddit submission text (enclosed in triple backticks), analyze whether it describes the stories (not recovery story), feelings (not recovered feelings), or behaviors 
        (not recovered feelings) of a patient with a mental disorder. Respond according to these guidelines:

        1. If the text describes the stories, feelings, or behaviors of a patient with a mental disorder, trim important part of the text without any lexical change and return following
        JSON format:
            {"Result":"yes", "Text": <trimmed or normal text>}

        2. If the text does not describe such stories, feelings, or behaviors, analyze it again:
            2.b. If it does not, respond with the following JSON format:
            {"Result":"no", "Text": <given_input_text>}

        \n\n
        Subreddit Submission Text: """ + comment_text)
    return response.text


def filter_using_llm(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    wrong_error = 0
    filtered_llm_result_list = []

    for post_body in tqdm(df['post_body'].unique(), desc=f"LLM filtering"):
        try:
            response_text = submission_prediction(submission_text=post_body)

            # Extract JSON
            pattern = re.compile(r'\{.*?\}', re.DOTALL)
            match = pattern.search(response_text)

            if match:
                response_json = json.loads(match.group(0))

                # If it requires comment
                if response_json['Result'] == 'comment':
                    # Get specific comments
                    comment_df = df[df['post_body'] == post_body]
                    comment_list = filter_comments_using_llm(comment_df)

                    # Extending due to the fact that filter_comments_using_llm method returns in list format
                    filtered_llm_result_list.extend(comment_list)
                else:
                    filtered_llm_result_list.append(response_json)
            else:
                wrong_error += 1

        except Exception as err:
            print(err)
            wrong_error += 1

    print(f"LLM filtering process was finished. The ratio is: ", round((wrong_error/len(df['post_body'])*100)))

    return pd.DataFrame(filtered_llm_result_list)


def filter_comments_using_llm(comment_df: pd.DataFrame) -> list:
    """

    :param comment_df:
    :return:
    """
    wrong_counter = 0
    result_data_list = []

    try:
        for comment_body in tqdm(comment_df['comment_body']):

            response_comment_str = comment_prediction(comment_text=comment_body)

            # Extract JSON
            pattern = re.compile(r'\{.*?\}', re.DOTALL)
            match = pattern.search(response_comment_str)

            if match:
                response_comment_json = json.loads(match.group(0))
                # Add directly
                result_data_list.append(response_comment_json)
            else:
                wrong_counter += 1

    except Exception as err:
        print(err)
        wrong_counter += 1

    print(f"Correctness ratio:", round((wrong_counter / len(comment_df))*100))
    return result_data_list
