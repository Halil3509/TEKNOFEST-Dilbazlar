import os
import time
import google.generativeai as genai
import ast
import re
import pandas as pd
from tqdm import tqdm
import time
from utils import get_configs
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

configs = get_configs()




genai.configure(api_key=os.getenv('DATABASE_URL'))

model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)
logging.info("Gemini defined")



def submission_prediction(sentence: str):
    global mistake_sentences

    response = model.generate_content("""
    Sen verilen cümleyi genel anlam ve kelime anlamı olarak analiz edip bu cümleden esinlenerek benzer cümleler oluşturan bir sentetik veri üreticisisin.

    \n\n

    Aşağıda verilen cümleyi genel anlam ve kelime anlamı olarak analiz et ve anla. Anlamı anladıktan sonra bu cümleyi söyleyen bir 
    insan bu cümleye benzer nasıl cümleler demiş olabilir bunlardan 5 adet uzunluğu orijinal cümleyle yakın olan cümleleri
    sadece ve sadece aşağıdaki python liste formatındaki gibi yaz. Başka hiçbir şey yazma:
    Çıktı formatı: ["<cumle1>", "<cumle2>", "<cumle3>", "<cumle4>", "<cumle5>"]

    \n\n
    Cümle: """ + sentence)

    try:
        pattern = re.compile(r'\[.*?\]', re.DOTALL)
        # print(response.text)
        match = pattern.search(response.text)
        if match:
            response_json = ast.literal_eval(match.group(0))
            return response_json

        else:
            mistake_sentences.append(sentence)
            print("It could not be augmented.")
            return []

    except Exception as err:
        mistake_sentences.append(sentence)
        print("It could not be augmented. Err:", err)
        return []



