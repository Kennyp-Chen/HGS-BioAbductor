from openai import OpenAI
from pydantic import BaseModel, Field
import requests
from typing import Literal
import os

# client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], 
#                 base_url=os.environ['BASE_URL'])
client = None
class Hypothesis(BaseModel):
    name: str
    reasoning: str
    explanation: str

class Hypothesises(BaseModel):
    Hypothesises: list[Hypothesis]

class HypothesisEvaluation(BaseModel):
    Hypothesis_name: str
    literature_support_score: int 
    literature_support_explanation: str 
    
    biological_plausibility_score: int 
    biological_plausibility_explanation: str 
    
    novelty_score: int 
    novelty_explanation: str 
    
    total_score: int
    decision: str

class HypothesisEvaluations(BaseModel):
    evals: list[HypothesisEvaluation]

def generate_system_prompt(model_name, cancer_type, data_description=None):
    """
    Generate a system prompt for proposing novel scientific hypotheses based on study findings.

    Parameters:
    - model_name (str): Name of the model used in the study.
    - cancer_type (str): Type of cancer studied.
    - key_findings (str): Summary of key findings of the study.
    - data_description (str, optional): Description of how the data was processed or analyzed.

    Returns:
    - str: A system prompt to guide the hypothesis generation.
    """
    data_description_text = f"\n{data_description}" if data_description else ""

    system_prompt = f"""
Propose novel scientific hypotheses based on the study's findings using a {model_name} to evaluate genes and interactions in {cancer_type}. The proposals should be grounded in the key findings of the study, linking specific genes, interactions, and pathways.{data_description_text}

Each hypothesis must:
  - Relate to the identified genes, interactions, or pathway enrichment analysis.
  - Clearly Describe Gene Interactions:
    - Detail how specific genes interact (e.g., direct binding, enzymatic activity, signaling modulation).
    - Explain the biological mechanisms through which these interactions might occur.
  - Focus on Functional Outcomes:
    - Explain how these interactions contribute to specific biological processes.
    - Discuss the downstream effects on cancer progression.
  - Integrate Pathway Enrichment:
    - Link the enriched pathways from the analysis to the proposed mechanisms, explaining how they support or enhance the biological plausibility of the hypothesis.
  - Provide Rationale for Biological Plausibility:
    - Base the hypotheses on known functions of the genes and pathways involved, while proposing novel interactions or mechanisms.

Each hypothesis should clearly link the study's findings to a biologically plausible outcome or a novel interaction/mechanism. Ensure to articulate how specific genes or interactions between genes or pathways may influence cancer pathogenesis or progression.

The final result should be in a list format with hypotheses and rationale:
Do not mention clusters in the title and explanation.
- 'Hypothesis <number>: Hypothesis title':
    - 'Reasoning': 'insert reasoning here',
    - 'Explanation': 'insert hypothesis explanation here'
    """
    return system_prompt
    

def get_hypothesis(system:str, user:str, model:str):
    completion = client.beta.chat.completions.parse(
        messages=[
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": user
        }
        ],
        model=model,
        response_format=Hypothesises
    )
    message = completion.choices[0].message.parsed
    return message


def get_evaluation(user:str, model:str):
    system = f"""
    To evaluate the proposed hypotheses based on existing literature, we can use a scoring system that considers the following criteria:

    Scoring Criteria
    Literature Support (0-5 points): Evidence from peer-reviewed studies supporting the hypothesis.
    Biological Plausibility (0-5 points): The likelihood that the proposed mechanism is feasible based on biological principles.
    Novelty (0-5 points): The uniqueness of the hypothesis and its potential to contribute new knowledge to the field.
    Each hypothesis will be scored out of a total of 15 points, with higher scores indicating stronger support for the hypothesis.
    The decision making progress are based on the overall scores:
      - Recept: Total score ≥ 10 points (Strong evidence and potential)
      - Reject: Total score < 10 points (Insufficient evidence or major flaws)
    """
    # if 'gpt' in model:
    completion = client.beta.chat.completions.parse(
    messages=[
        {"role":"system", "content": system},
        {"role":"user", "content": user}
    ],
    model=model,
    response_format=HypothesisEvaluations
)
    message = completion.choices[0].message.parsed
    return message

def extract_keywords(user:str, model:str):
    system = f"""
Task Description
You are provided with a biology-related description. Your job is to construct a PubMed search query by extracting only the meaningful multi-word biological term(s) from the description. Specifically:

- Identify and preserve multi-word terms/phrases that represent meaningful biological concepts (e.g., “Mitochondrial respiratory chain complex I”).
- Do not break these multi-word terms into separate words.
- Exclude generic or non-essential words (e.g., “assembly” if it does not add new biological meaning).
Output Requirements

- Only output the final PubMed search query string, using quotes if needed, and combining multiple terms with OR if there are multiple.
- Do not provide any explanations or additional text.
- If there is only one final term, simply provide it as "Mitochondrial respiratory chain complex I" (without parentheses or OR).
- If multiple terms exist, enclose each multi-word term in quotes and separate them with OR. For example: ("term1" OR "term2")
"""
    completion = client.chat.completions.create(
        messages=[
            {"role":"system", "content":system},
            {"role":"user", "content":user}
        ],
        model=model,
        temperature=0
    )
    message = completion.choices[0].message.content
    return message


