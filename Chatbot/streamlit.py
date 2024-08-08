import time
import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import joblib
import plotly.express as px

from ai import SOTA
import torch
from PIL import Image


im = Image.open("../dilbazlar.ico")
st.set_page_config(
    page_title="Dilbazlar",
    page_icon=im,
    layout="wide",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
MESSAGE_NUMBER_THRESHOLD = 4
DISORDER_RATIO = 0.50

load_dotenv()
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

@st.cache_resource
def build_sota():
    """
    :return: Sota class
    """
    return SOTA()

# Load Models
st.session_state.sota = build_sota()

# Defining safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

genai.configure(api_key=GOOGLE_API_KEY)

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except:
    # data/ folder already exists
    pass

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Geçmiş konuşmalar')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Session seç',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Session seç',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )

    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat
    st.session_state.chat_title = f'Chat-{st.session_state.chat_id}'


    print("Percentages: ", st.session_state.sota.normalized_percentages)
    fig = px.bar(x=list(st.session_state.sota.normalized_percentages.values()),
                 y=list(st.session_state.sota.normalized_percentages.keys()),
                 labels={'x':'Yüzdeler', 'y':'Hastalıklar'},
                 title="Mental Rahatsızlık Dağılım Grafiği")


    st.plotly_chart(fig)

html_content = """
<div style="display: flex; align-items: center; justify-content: center;">
    <img src="https://pbs.twimg.com/media/GUe-Q-OXMAAcbUz?format=jpg&name=small"
     alt="Left Image" style="width:125px;height:auto; margin-right: 20px;">
    <h1 style="margin: 0;">Dilbazlar Chatbot</h1>
    <img src="https://austurkiye.org.tr/uploads/users/267-235.png" alt="Right Image" style="width:100px;height:auto; margin-left: 20px;">
</div>
<div style="display: flex; align-items: center; justify-content: center;">
    <p>"Mental Sağlığına Göz Kırp" </p>
</div>
"""

# Display the HTML in Streamlit
st.markdown(html_content, unsafe_allow_html=True)


# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:

    st.session_state.messages = []
    st.session_state.gemini_history = [
        {
            "role": "user",
            "parts": [{"text": "Sen bir psikologsun. Kişiyle kısa kısa konuş ve ona bir psikologla konuşuyormuş gibi hissettir."}],
        }
    ]
    print('new_cache made')
    st.session_state.sota.empty_cache()  ## Make counter 0


st.session_state.model = genai.GenerativeModel('gemini-1.5-flash', safety_settings=safety_settings)
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Mesaj yazabilirsin'):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )
    ## Send message to AIs
    response = st.session_state.chat.send_message(
        prompt,
        stream=True,
    )
    st.session_state.sota.message_counter += 1  # Increasing message count
    print(f"Counter increased. Counter is: {st.session_state.sota.message_counter}")


    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''
        assistant_response = response
        # Streams in a chunk at a time
        for chunk in response:
            # Simulate stream of chunk
            # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)

            for i, ch in enumerate(chunk.text.split(' ')):
                if i == len(chunk.text.split(' ')) -1:
                    full_response += ch + ''
                    time.sleep(0.05)
                else:
                    full_response += ch + ' '
                    time.sleep(0.05)
                # Rewrites with a cursor at end
                message_placeholder.write(full_response + '')
        # Write full message with placeholder
        message_placeholder.write(full_response)


    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=st.session_state.chat.history[-1].parts[0].text,
            avatar=AI_AVATAR_ICON,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )


    print("AI is being run...")
    specific_result = st.session_state.sota.prediction_flow_standard(sentence=full_response)

    if isinstance(specific_result, list):
        for sp in specific_result:
            # Add results
            st.session_state.sota.results.append({
                "message": prompt,
                "result": sp
            })
    else:
        # Add results
        st.session_state.sota.results.append({
            "message": prompt,
            "result": specific_result
        })


    st.session_state.sota.disorder_ratio = st.session_state.sota.update_disorder_ratio()
    print("Updated disorder ratio: ", st.session_state.sota.disorder_ratio)

    st.session_state.sota.update_percentages()  # Update percentages


    # *** Chatbot Evaluated Response Part ***
    if st.session_state.sota.message_counter > MESSAGE_NUMBER_THRESHOLD and \
            st.session_state.sota.disorder_ratio > DISORDER_RATIO:

        evaluating_response = st.session_state.chat.send_message(
            f"""
                        Aşağıda bir kişinin chatbot ile konuşması ve bunun sonucunda her bir mesajına göre yapay zeka tespit edilmiş 
                        mental rahatsızlık çıktıları yer almaktadır.

                        Sen uzman bir psikologsun. JSON formatında verilen sonuçlar verisine bakarak *result anahtarı* çıktısının 
                        *message anahtarı* göre inceleyip bunları kişiyi etik bir dille açıkla.
                        
                        result anahtarında çıkan çıktılara odaklan ve bunu kişilere uygun bir dille ifade et. 

                        Sonuçlar: ```{st.session_state.sota.results}``` 

                        Uzman psikolog yorumu: 
                    """,
            stream=True
        )
        # Display assistant response in chat message container
        with st.chat_message(
                name=MODEL_ROLE,
                avatar=AI_AVATAR_ICON,
        ):
            message_placeholder = st.empty()
            full_response = ''
            assistant_response = evaluating_response
            # Streams in a chunk at a time
            for chunk in evaluating_response:
                # Simulate stream of chunk
                # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)

                for i, ch in enumerate(chunk.text.split(' ')):
                    if i == len(chunk.text.split(' ')) - 1:
                        full_response += ch + ''
                        time.sleep(0.05)
                    else:
                        full_response += ch + ' '
                        time.sleep(0.05)
                    # Rewrites with a cursor at end
                    message_placeholder.write(full_response + '')
            # Write full message with placeholder
            message_placeholder.write(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            dict(
                role=MODEL_ROLE,
                content=st.session_state.chat.history[-1].parts[0].text,
                avatar=AI_AVATAR_ICON,
            )
        )
        st.session_state.gemini_history = st.session_state.chat.history
        # Save to file
        joblib.dump(
            st.session_state.messages,
            f'data/{st.session_state.chat_id}-st_messages',
        )
        joblib.dump(
            st.session_state.gemini_history,
            f'data/{st.session_state.chat_id}-gemini_messages',
        )

    st.rerun()




