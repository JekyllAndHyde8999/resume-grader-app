import os
import streamlit as st
import tempfile
import string
import time
import spacy
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatGooglePalm
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.utils.math import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer


# initialize credentials
load_dotenv()

st.set_page_config(page_title="Resume Grader")

# initialize global variables
nlp = spacy.load("en_core_web_sm")
chat_palm = ChatGooglePalm(temperature=0.1)
summarization_prompt_template = """
Write a brief summary of the following content extracted from a resume. Be sure to keep important keywords such as those pertaining to their skills, projects completed, work experience, etc.

{text}

SUMMARY:
"""

prompt = PromptTemplate(template=summarization_prompt_template, input_variables=["text"])
summary_chain = load_summarize_chain(llm=chat_palm, chain_type='stuff', prompt=prompt)
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
eng_stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def perform_ner_spacy(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def replace_entities_with_mask(text, entity_masks=None):
    if entity_masks is None:
        entity_masks = {}

    doc = nlp(text)
    replaced_text = []

    in_entity = {entity: False for entity in entity_masks}

    for token in doc:
        for entity, mask in entity_masks.items():
            if token.ent_type_ == entity:
                if not in_entity[entity]:
                    replaced_text.append(mask)
                    in_entity[entity] = True
                for other_entity in entity_masks:
                    if other_entity != entity:
                        in_entity[other_entity] = False
            else:
                in_entity[entity] = False

        if not any(in_entity.values()):
            replaced_text.append(token.text)

    return ' '.join(replaced_text)


def preprocess(text):
    # remove punctuation
    translation_table = str.maketrans("", "", string.punctuation)
    text = text.replace('\n', '').translate(translation_table)
    
    # mask PERSON entities
    text = replace_entities_with_mask(text, entity_masks={"PERSON": "[PERSON]", "ORG": "[ORG]"}).replace('[', '').replace(']', '')
    st.markdown(f"{text=}")
    
    # remove stop words and lemmatize non-stop words
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words])

    # pos_tags = pos_tag(filtered_words)
    return text


st.title("Resume Grader")
st.markdown("Upload a job description and your resumes to rank each resume's relevance towards the job role.")

resumes = st.file_uploader(label="Upload resumes here", type='pdf', accept_multiple_files=True)
job_description = st.text_area(label="Enter job description here.")

if resumes and job_description:
    with st.spinner("Summarizing Job Description"):
        job_desc_summary = summary_chain.run([Document(page_content=job_description)])
    
    resumes_bar = st.progress(0, text="Summarizing resumes ...")
    resume_collection = dict()
    for i, resume in enumerate(resumes):
        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'uploaded_resume.pdf'), mode='wb') as f:
                f.write(resume.getvalue())
            
            pdf_loader = PyPDFLoader(file_path=os.path.join(tempdir, "uploaded_resume.pdf"))
            pdf = pdf_loader.load()
        
        pdf_summary = preprocess(summary_chain.run(pdf))
        resume_collection[resume.file_id] = {
            "summary": pdf_summary,
            "name": resume.name
        }

        resumes_bar.progress(int((i + 1) / len(resumes) * 100), text="Summarizing resumes ...")
    time.sleep(0.5)
    resumes_bar.empty()

    st.markdown(resume_collection[resumes[0].file_id]["summary"])

    with st.spinner("Fitting vectorizer ..."):
        all_resumes = [resume["summary"] for resume in resume_collection.values()]
        vectorizer.fit([*all_resumes, job_desc_summary])
    
    jd_vector = vectorizer.transform([job_desc_summary]).toarray()
    vectorizing_bar = st.progress(0, text="Calculating similarities ...")
    for key in resume_collection.keys():
        resume_vector = vectorizer.transform([resume_collection[key]["summary"]]).toarray()
        resume_collection[key]["similarity"] = (1 + cosine_similarity(resume_vector, jd_vector).flatten()[0]) / 2
        vectorizing_bar.progress(int((i + 1) / len(resumes) * 100), text="Calculating similarities ...")
    time.sleep(0.5)
    vectorizing_bar.empty()
    
    sorted_collection = sorted(resume_collection.items(), key=lambda x: -x[1]['similarity'])
    columns = st.columns(2)

    for r in sorted_collection:
        columns[0].write(r[1]['name'])
        columns[1].write(r[1]['similarity'])
