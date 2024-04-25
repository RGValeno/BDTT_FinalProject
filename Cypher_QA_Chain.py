import yaml
import constants
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase

# from neo4j.debug import watch
# watch("neo4j")

# Load OpenAI API key
CONFIG_FILE = constants.config

# Load the config file
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

# Set the OpenAI API key
OPENAI_API_KEY = config['OpenAi']['key']

# Load the OpenAI API key
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

llm = ChatOpenAI(
    openai_api_key=config['OpenAi']['key']
)

graph = Neo4jGraph(
    url=config['MyNeo4j']['url'],
    username=config['MyNeo4j']['username'],
    password=config['MyNeo4j']['password'],
)

CYPHER_GENERATION_TEMPLATE = """
You are an expert news anchor translating user questions into Cypher to answer questions about articles and provide information.
Convert the user's question based on the schema.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

If no data is returned, do not attempt to answer the question.
Only respond to questions that require you to construct a Cypher statement.
Do not include any explanations or apologies in your responses.

Examples:

Find people and titles:
MATCH (p:Person)-[:authored]->(g)
RETURN a.title, p.author

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)

cypher_chain.invoke({"query": "How many people authored 1 article and who were they?"})