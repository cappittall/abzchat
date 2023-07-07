from io import StringIO
import streamlit as st
from streamlit_chat import message

from tools.tools import read_models, load_env
import os
from translations import translation

from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes import VectorstoreIndexCreator

# load_env()

path = "data/chats"

try:
    os.environ['OPENAI_API_KEY'] = st.secrets["openai"]
except:
    load_env()

loader = DirectoryLoader(path=path, glob='*.txt', show_progress=True, loader_cls=TextLoader)
docs = loader.load()

index = VectorstoreIndexCreator().from_documents(docs)
llm = ChatOpenAI()
chain = load_qa_chain(llm=llm, chain_type="stuff")
lang = "TR"

if 'messages' not in st.session_state:
    st.session_state.messages=[SystemMessage(content=translation[lang]['helpfull'])]

def download_file(fn):
    with open(f"data/chats/{fn}", 'rb') as file:
        contents = file.read()
        st.sidebar.download_button(
            label=translation[lang]["clicktoload"],
            data=contents,
            file_name=fn,
            mime='text/plain'
        )
        
def main():
    global chain, docs, lang
        
    with st.sidebar:
        
        lang = st.selectbox(translation["EN"]["language"], options=['EN', "TR"])
        
        files = st.file_uploader(translation[lang]["file_info"], 
                         accept_multiple_files=True, type=["txt"], help=translation[lang]["help"])
        st.header(translation[lang]['header'])
        for file in files:           
            filepath = os.path.join(path, file.name)
            with open(f'{path}/{file.name}', 'w') as f:
                f.write(file.getvalue().decode("utf-8"))
            st.write(f'{file.name} {translation[lang]["loaded"]}')
        
        file_list = os.listdir(path=path)
         # Display chat titles and delete icons
        st.write(translation[lang]['filelist'])
        reversed_keys = reversed(list(file_list))
        for fn in reversed_keys:
            empt = st.empty()
            col1, col2, col3 = empt.columns([6, 2, 2])
            if col1.button(fn, key = f"title{fn}" ):
                print(f'File name : {fn}')
                    
            if col2.button("❌", key = f"del{fn}"):
                st.session_state['delete'] = fn
                                    
            if col3.button('📥', key = f"edit{fn}"):
                download_file(fn)
                
            # Check if we're in editing mode for this chat
            if 'delete' in st.session_state and st.session_state['delete'] == fn:
                if st.button(f'{translation[lang]["areyousure"]} \n\n{fn}?', key="custom_button"):
                    # Store the id of the chat we're deleting
                    if len(file_list) >1 : 
                        print(f'Delete chat is {fn}')
                        os.remove(f"data/chats/{fn}")
                        del file_list[file_list.index(fn)]
                        del st.session_state['delete']  # Exit delete mode after confirmation
                        
                    else:
                        st.write('If only ONE file exists \nUpload another txt document first')
                    
                else:
                    print(f'Waiting for confirmation to delete chat {fn}')
      
        
    query = st.chat_input(translation[lang]["askme"])
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
    
