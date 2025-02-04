import os
import chromadb

from dotenv import load_dotenv

from openai import OpenAI

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.llms.anyscale import Anyscale
from langchain_community.embeddings import AnyscaleEmbeddings
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


load_dotenv()

client = OpenAI(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key=os.environ['ANYSCALE_API_KEY']
)

llm = Anyscale(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key=os.environ['ANYSCALE_API_KEY'],
    model_name='meta-llama/Meta-Llama-3-70B-Instruct',
    temperature=0
)

embedding_model = AnyscaleEmbeddings(
    client=client,
    model='thenlper/gte-large'
)

chromadb_client = chromadb.EphemeralClient()

collection = chromadb_client.get_or_create_collection(
    name='full_document_chunks',
    metadata={"hnsw:space": "cosine"}
)

documents = [
    Document(
        id=1,
        page_content="We design, develop, manufacture, sell and lease high-performance fully electric vehicles and energy generation and storage systems, and offer services related to our products. We generally sell our products directly to customers, and continue to grow our customer-facing infrastructure through a global network of vehicle showrooms and service centers, Mobile Service, body shops, Supercharger stations and Destination Chargers to accelerate the widespread adoption of our products. We emphasize performance, attractive styling and the safety of our users and workforce in the design and manufacture of our products and are continuing to develop full self-driving technology for improved safety. We also strive to lower the cost of ownership for our customers through continuous efforts to reduce manufacturing costs and by offering financial and other services tailored to our products.",
        metadata={"year": 2023, "section": "business"}
    ),
    Document(
        id=2,
        page_content="We have previously experienced and may in the future experience launch and production ramp delays for new products and features. For example, we encountered unanticipated supplier issues that led to delays during the initial ramp of our first Model X and experienced challenges with a supplier and with ramping full automation for certain of our initial Model 3 manufacturing processes. In addition, we may introduce in the future new or unique manufacturing processes and design features for our products. As we expand our vehicle offerings and global footprint, there is no guarantee that we will be able to successfully and timely introduce and scale such processes or features.",
        metadata={"year": 2023, "section": "risk_factors"}
    ),
    Document(
        id=3,
        page_content="We recognize the importance of assessing, identifying, and managing material risks associated with cybersecurity threats, as such term is defined in Item 106(a) of Regulation S-K. These risks include, among other things: operational risks, intellectual property theft, fraud, extortion, harm to employees or customers and violation of data privacy or security laws. Identifying and assessing cybersecurity risk is integrated into our overall risk management systems and processes. Cybersecurity risks related to our business, technical operations, privacy and compliance issues are identified and addressed through a multi-faceted approach including third party assessments, internal IT Audit, IT security, governance, risk and compliance reviews. To defend, detect and respond to cybersecurity incidents, we, among other things: conduct proactive privacy and cybersecurity reviews of systems and applications, audit applicable data policies, perform penetration testing using external third-party tools and techniques to test security controls, operate a bug bounty program to encourage proactive vulnerability reporting, conduct employee training, monitor emerging laws and regulations related to data protection and information security (including our consumer products) and implement appropriate changes.",
        metadata={"year": 2023, "section": "cyber_security"}
    ),
    Document(
        id=4,
        page_content="The automotive segment includes the design, development, manufacturing, sales and leasing of high-performance fully electric vehicles as well as sales of automotive regulatory credits. Additionally, the automotive segment also includes services and other, which includes non-warranty after- sales vehicle services and parts, sales of used vehicles, retail merchandise, paid Supercharging and vehicle insurance revenue. The energy generation and storage segment includes the design, manufacture, installation, sales and leasing of solar energy generation and energy storage products and related services and sales of solar energy systems incentives.",
        metadata={"year": 2022, "section": "business"}
    ),
    Document(
        id=5,
        page_content="Since the first quarter of 2020, there has been a worldwide impact from the COVID-19 pandemic. Government regulations and shifting social behaviors have, at times, limited or closed non-essential transportation, government functions, business activities and person-to-person interactions. Global trade conditions and consumer trends that originated during the pandemic continue to persist and may also have long-lasting adverse impact on us and our industries independently of the progress of the pandemic.",
        metadata={"year": 2022, "section": "risk_factors"}
    ),
    Document(
        id=6,
        page_content="The German Umweltbundesamt issued our subsidiary in Germany a notice and fine in the amount of 12 million euro alleging its non-compliance under applicable laws relating to market participation notifications and take-back obligations with respect to end-of-life battery products required thereunder. In response to Tesla’s objection, the German Umweltbundesamt issued Tesla a revised fine notice dated April 29, 2021 in which it reduced the original fine amount to 1.45 million euro. This is primarily relating to administrative requirements, but Tesla has continued to take back battery packs, and filed a new objection in June 2021. A hearing took place on November 24, 2022, and the parties reached a settlement which resulted in a further reduction of the fine to 600,000 euro. Both parties have waived their right to appeal.",
        metadata={"year": 2022, "section": "legal_proceedings"}
    )
]

document_embeddings = []

embeddings = client.embeddings.create(
    model="thenlper/gte-large",
    input=[d.page_content for d in documents]
)

document_embeddings.extend(embeddings.data)

document_embeddings_data = [embedding_data.embedding for embedding_data in document_embeddings]

collection.add(
    ids=[d.id for d in documents],
    documents=[d.page_content for d in documents],
    metadatas=[d.metadata for d in documents],
    embeddings=document_embeddings_data
)

vectorstore = Chroma(
    client=chromadb_client,
    collection_name="full_document_chunks",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

user_input = "Any fines in 2022?"

relevant_document_chunks = retriever.invoke(user_input)

context_list = [d.page_content for d in relevant_document_chunks]

# Information compression

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(user_input)

for compressed_doc in compressed_docs:
    print(compressed_doc.page_content)

# Reranking

context_query_pairs_for_scoring = [[user_input, doc_text] for doc_text in context_list]

crossencoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
crossencoder.score(context_query_pairs_for_scoring)

reranker = CrossEncoderReranker(model=crossencoder, top_n=5)

reranker_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=retriever
)

reranked_docs = reranker_retriever.invoke(user_input)

for i in range(len(reranked_docs)):
    if reranked_docs[i].page_content == context_list[i]:
        continue
    else:
        print(f"Document at rank: {i} after reranking differs from the original order")