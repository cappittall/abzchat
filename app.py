from io import StringIO
import streamlit as st
from streamlit_chat import message

from tools.tools import read_models, load_env
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes import VectorstoreIndexCreator

# load_env()

path = "data/chats"
os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["openai_api_key"]

loader = DirectoryLoader(path=path, glob='*.txt', show_progress=True, loader_cls=TextLoader)
docs = loader.load()

index = VectorstoreIndexCreator().from_documents(docs)
llm = ChatOpenAI()
chain = load_qa_chain(llm=llm, chain_type="stuff")

if 'messages' not in st.session_state:
    st.session_state.messages=[SystemMessage(content="You are a helpfull chat assistan")]

def main():
    global chain, docs
        
    with st.sidebar:
        files = st.file_uploader('Cevaplarda olmasını istediğiniz txt dosyasını yükleyiniz.', 
                         accept_multiple_files=True, type=["txt"])
        for file in files:           
            filepath = os.path.join(path, file.name)
            with open(f'{path}/{file.name}', 'w') as f:
                f.write(file.getvalue().decode("utf-8"))
            st.write(f'{file.name} dosyanız yüklendi...')
      
        
    query = st.chat_input('Bana istediğin herşeyi sorabilirsin canım')
    if query:
        response = index.query(query)
        print(response)
        if response:
            st.session_state.messages.append(HumanMessage(content=query))
            st.session_state.messages.append(AIMessage(content=response))

    messages = st.session_state.get('messages', [])
    for i, msg in list(enumerate(messages[1:])):
        is_user = True if i % 2 == 0 else False
        message(msg.content, is_user=is_user, key=f"{i}{'_user' if is_user else '_ai'}")
        
if __name__=='__main__':
    main()
    
