import asyncio
import aiofiles
import glob
import json
import logging
import os
import re
import numpy as np
import collections
from functools import partial
import multiprocessing as mul
from tqdm import tqdm
import math
from openai import OpenAI
from mcp_agent.core.fastagent import FastAgent

# Initialize FastAgent and OpenAI client
fast = FastAgent("hypothesis candidate tournament", config_path='config.yaml')

client = OpenAI(api_key=fast.config.get('generic').get('api_key'),
                base_url=fast.config.get('generic').get('base_url'))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Agent definitions
@fast.agent(name='relation_prover', 
           instruction="""You are a specialized AI Agent focused on biomedical literature analysis from pubmed. Your task is to evaluate and validate whether a biological relationship between two entities can be semantically supported by scientific literature.

Step-by-step process:
1. Extract the minimal keyword (MeSH Term) from the pair provided to search scientific literature—do not use the relationship itself as the keyword.
3. Begin by reviewing abstracts only:
   - If an abstract contains a direct or semantically implied connection between the two entities, then proceed to review the full text to confirm and strengthen the evidence.
   - If no abstract shows any semantic link, halt the search and report that no credible evidence exists.
4. If the minimal keywords yield no results, refine your query by subtracting elements from the original keyword set—removing non-essential words using close synonyms or semantically equivalent phrases without adding any new terms (e.g., using the gene name or function name without additional descriptors).
5. Apply semantic reasoning to evaluate whether the intended relationship exists, even if the wording does not exactly match.

For any paper that supports the relationship:
- Report the **title and PMID**.
- Extract a **exact sentence** from the abstract or full text that supports the connection.

Conclude with:
Answer: <True/False>

you must obey the following response format strictly:

# Search Strategy and Reasoning
"Insert your keyword query logic and how you reasoned through the abstract/full text here"

# literature evidence
## "Insert title and PMID here"
"Insert the exact sentence from the abstract or full text that semantically supports the connection between the two biological entities here"

Answer: <True/False>
""" ,
           servers=['pubtator3'])

@fast.agent(name='experiment designer', 
            instruction="""You are a specialized AI Agent tasked with designing a testable experimental protocol or simulation model to validate a given biomedical viewpoint. 
Design a Testable Experimental or Simulation Protocol(e.g., in vitro assays, animal models, or computational simulations) brifly in one paragraph.
""")

@fast.agent(name='table_maker',
            instruction="""You are a specialized AI Agent tasked with generating well-structured Markdown tables that represent biomedical hypotheses and their evaluated viewpoints. 

Your input will include a json of the decomposed viewpoints. For each decomposed viewpoint, generate a table row where:
- The row identifier is the decomposed viewpoint text.
- The table columns are as follows:
    • "is proved": Indicates whether the viewpoint has been validated (True/False) from the evidence.
    • "proved PMID": If validated, extract the PubMed ID of the supporting literature from the agent response.
    • "proved literature title": If validated, extract the title of the literature from the agent response.
    • "proved sentence in the literature": If validated, extract the exact sentence from the literature that supports the view from the agent response.

The table itself should be in Markdown format without caption or header. Return only the Markdown row without any additional commentary in one line.
""")

# Utility functions
def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> list[float]:
    """Get embedding for text using OpenAI API"""
    logger.info(f"Embedding for text by {model}. Text: {text}")
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding

def Euclidean_Distance(a, b):
    """Calculate Euclidean distance between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a-b)**2))

def get_json_files(path, include=""):
    """Get JSON files from directory with optional include filter"""
    return [file for file in os.listdir(path) if file.endswith('.json') and include in file]

def count_true_false(md_content):
    """Count True/False occurrences in markdown content"""
    true_count = len(re.findall(r'\bTrue\b', md_content))
    false_count = len(re.findall(r'\bFalse\b', md_content))
    return true_count, false_count

# Novelty scoring functions
def Novelty_Scoring(scores):
    """Calculate ON score from HD, CI, and CD scores"""
    scores['ON'] = (scores['HD']*scores['CI'])/scores['CD']

def PreScoring(text_embedding, k, VD_dic):
    """Calculate novelty scores for a given text embedding"""
    scores = {'HD':0,'CD':0,'CI':0}
    sim_past = collections.defaultdict(dict)
    sim_con = collections.defaultdict(dict)
    
    # Process past papers
    for fn in VD_dic[f'past']:
        with open(os.path.join(VD_dic['path_past'],fn), 'r') as f:
            data = json.load(f)
        for pmid,paper in data.items():
            if 'article_types' in paper and(("Review" in paper['article_types']) or ("Meta-Analysis" in paper['article_types'])):
                continue
            ed = Euclidean_Distance(text_embedding,paper['abstract_embedding'])
            if len(sim_past)<k or (ed<max([sim_past[pmid]['Euclidean_Distance'] for pmid in sim_past]) and paper['title'] not in [sim_past[pmid]['title'] for pmid in sim_past]):
                sim_past[pmid] = {'Euclidean_Distance':ed,'PublishYear':str(paper['publishTime'])[:4]} 
                sim_past[pmid].update(paper)
                if len(sim_past)>k:
                    sim_past.pop(max(sim_past, key=lambda x: sim_past[x]['Euclidean_Distance']))
        for paper in sim_past.values():
            scores['HD']+=paper['Euclidean_Distance']
        scores['HD']/=k
    
    # Process contemporary papers
    for fn in VD_dic[f'con']:
        with open(os.path.join(VD_dic['path_con'],fn), 'r') as f:
            data = json.load(f)
        for pmid,paper in data.items():
            if 'article_types' in paper and(("Review" in paper['article_types']) or ("Meta-Analysis" in paper['article_types'])):
                continue
            ed = Euclidean_Distance(text_embedding,paper['abstract_embedding'])
            if len(sim_con)<k or (ed<max([sim_con[pmid]['Euclidean_Distance'] for pmid in sim_con]) and paper['title'] not in [sim_con[pmid]['title'] for pmid in sim_con]):
                sim_con[pmid] = {'Euclidean_Distance':ed,'PublishYear':str(paper['publishTime'])[:4]} 
                sim_con[pmid].update(paper)
                if len(sim_con)>k:
                    sim_con.pop(max(sim_con, key=lambda x: sim_con[x]['Euclidean_Distance']))
    
    # Calculate CI and CD
    for paper in sim_con.values():
        scores['CD']+=paper['Euclidean_Distance']
        scores['CI']+=paper['citations_count']
    scores['CI']/=k
    scores['CD']/=k
    logger.info(f"Scores: {scores}")
    
    # Get years and top k papers
    same_year_k = {'past':[sim_past[pmid]['PublishYear'] for pmid in sim_past],'con':[sim_con[pmid]['PublishYear'] for pmid in sim_con]}
    top_k_past = [(pmid, paper['title']) for pmid, paper in sorted(sim_past.items(), key=lambda x: x[1]['Euclidean_Distance'])[:k]]
    top_k_con = [(pmid, paper['title']) for pmid, paper in sorted(sim_con.items(), key=lambda x: x[1]['Euclidean_Distance'])[:k]]
    logger.info(f"Similar papers in past: {[sim_past[pmid]['PublishYear'] for pmid in sim_past]}, in contemporary: {[sim_con[pmid]['PublishYear'] for pmid in sim_con]}")
    
    return scores, same_year_k, top_k_past, top_k_con

def get_year_scores(path, year, i):
    """Get average scores for a specific year"""
    # Handle wrong dates in dataset
    if "Past2010-2019" in path and year=='2020':
        year = '2019'
    elif "Contemporary2020-2023"in path and year=='2024':
        year = '2023'
    elif "Past2010-2021" in path and year=='2022':
        year = '2021'
    
    fn_year = os.path.join(path,f'{year}/avg_scores{i}.json')
    with open(fn_year, 'r') as f: 
        return json.load(f)

def ON_norm_minmax(ONs):
    """Normalize ON scores using min-max normalization"""
    ON_min = min(ONs)
    ON_max = max(ONs)
    ONs_norm = [(ON-ON_min)/(ON_max-ON_min) for ON in ONs]
    return ONs_norm

# Scoring functions
def harmonic_mean_score(ON_norm, OP):
    """Calculate harmonic mean score emphasizing balance"""
    if ON_norm + OP == 0:
        return 0.0
    return 2 * (ON_norm * OP) / (ON_norm + OP)

def weighted_linear_score(ON_norm, OP, weight_OP=0.5):
    """Calculate weighted linear combination score"""
    return weight_OP * OP + (1 - weight_OP) * ON_norm

def geometric_mean_score(ON_norm, OP):
    """Calculate geometric mean score reducing extreme values"""
    return math.sqrt(ON_norm * OP)

def harmonic_mean_scores(ON_norm_list, OP_list):
    """Batch calculate harmonic mean scores"""
    scores = []
    for ON_norm, OP in zip(ON_norm_list, OP_list):
        if ON_norm + OP == 0:
            scores.append(0.0)
        else:
            scores.append(2 * (ON_norm * OP) / (ON_norm + OP))
    return scores

def geometric_mean_scores(ON_norm_list, OP_list):
    """Batch calculate geometric mean scores"""
    return [math.sqrt(ON_norm * OP) for ON_norm, OP in zip(ON_norm_list, OP_list)]

def weighted_linear_scores(ON_norm_list, OP_list, weight_OP=0.5):
    """Batch calculate weighted linear scores"""
    return [weight_OP * OP + (1 - weight_OP) * ON_norm 
            for ON_norm, OP in zip(ON_norm_list, OP_list)]

# Agent calling function
async def call_agent(query, agent_name, response_file = None):
    """Call an agent and optionally save/load response from file"""
    if response_file and os.path.exists(response_file):
        async with aiofiles.open(response_file, mode="r", encoding="utf-8") as rf:
            response = await rf.read()
    else:
        async with fast.run() as agent:
            response = await agent[agent_name].send(query)
        if response_file:
            if response != "":
                async with aiofiles.open(response_file, mode="w", encoding="utf-8") as wf:
                    await wf.write(response)
    return response or ""

# Main evaluation functions
async def evaluate_literature_evidence():
    """Evaluate literature evidence for hypotheses viewpoints"""
    # Get configuration from config file
    plausibility_config = fast.config.get('plausibility', {})
    
    viewpoint_pattern = plausibility_config.get('viewpoint_pattern', '')
    viewpoint_files = glob.glob(f"{viewpoint_pattern}/viewpoints.json") if viewpoint_pattern else []
    
    viewpoints = {}
    for viewpoint_file in viewpoint_files:
        async with aiofiles.open(viewpoint_file, mode="r", encoding="utf-8") as f:
            viewpoint_dir = os.path.dirname(viewpoint_file)
            file_content = await f.read()
            try:
                data = json.loads(file_content)['viewpoints']
                viewpoints[viewpoint_dir] = data
            except json.JSONDecodeError as error:
                logging.info(f"Error parsing {viewpoint_file}: {error}")
            
    evidence_results = {}
    evidence_percent = {}
    for dir_path, vp_list in viewpoints.items():
        evidence_results[dir_path] = []

        markdown_table = (
                    f"| viewpoints | is proved | proved PMID | proved literature title | proved sentence in the literature |test method |\n"
                    f"|------------|-----------|-------------|-------------------------|-----------------------------------|------------|\n")
        for vp in vp_list:
            query = (
                f"Prove whether there is literature evidence for the pair: "
                f"'{vp['start_node']}' and '{vp['end_node']}' with relation '{vp['relation']}'. "
                "Return True if evidence exists, otherwise return False. "
            )
            try:
                response_dir = os.path.join(dir_path, "relation_prove_pubtator3")
                os.makedirs(response_dir, exist_ok=True)
                sanitized_start = vp['start_node'].replace("/", "_")
                sanitized_end = vp['end_node'].replace("/", "_")
                sanitized_relation = vp['relation'].replace("/", "_")
                response_file = os.path.join(response_dir, f"{sanitized_start}_{sanitized_relation}_{sanitized_end}.md")
                response = await call_agent(query, "relation_prover", response_file)
                
                result = True if re.search(r'Answer: True', response, re.IGNORECASE) else False
                
                pattern = r'(?s)^(?:.*?)(?=# literature evidence)'
                json_prompt = {
                "start_node": vp["start_node"],
                "relation": vp["relation"],
                "end_node": vp["end_node"],
                "evidence": result,
                "query": query,
                "agent_response": re.sub(pattern, '', response.strip(), count=1, flags=re.MULTILINE),
            }

                markdown_table += (await call_agent(json_prompt, "table_maker", response_file=False) + '\n')
            except Exception as exc:
                logging.error(f"Error checking evidence for {vp}: {exc}")
                result = False
            
            evidence_results[dir_path].append({
                "start_node": vp["start_node"],
                "relation": vp["relation"],
                "end_node": vp["end_node"],
                "evidence": result,
                "query": query,
                "agent_response": response if 'response' in locals() else "No response"
            })
        vs_fn = os.path.join(response_dir, "viewpoint_summary.md")
        async with aiofiles.open(vs_fn, 'w') as f:
            await f.write(markdown_table)
        logging.info(f"Generated Markdown Table for viewpoint {vp}:\n{markdown_table}")
        evidence_percent[dir_path] = sum([i['evidence'] for i 
                                          in evidence_results[dir_path]])/len(evidence_results[dir_path])
    
    # Get output paths from config
    out_dir = plausibility_config.get('out_dir', './output')
    evidence_results_file = plausibility_config.get('evidence_results_file', f"{out_dir}/pubtator3_evidence_results.json")
    evidence_percent_file = plausibility_config.get('evidence_percent_file', f"{out_dir}/pubtator3_evidence_percent.json")
    
    async with aiofiles.open(evidence_results_file, mode="w", encoding="utf-8") as res_file:
        await res_file.write(json.dumps(evidence_results, indent=4))
    async with aiofiles.open(evidence_percent_file, mode="w", encoding="utf-8") as perc_file:
        await perc_file.write(json.dumps(evidence_percent, indent=4))
    
    return evidence_results, evidence_percent

def evaluate_novelty_scores():
    """Evaluate novelty scores for hypotheses"""
    import pandas as pd
    from pathlib import Path
    
    # Get paths from config
    novelty_config = fast.config.get('novelty', {})
    input_config = fast.config.get('input_triplet', {})
    data_root = fast.config.get('paths').get('data_root')
    Set = novelty_config.get('scoring_set', "NoveltyScoring")
    path_con = novelty_config.get('contemporary_path').format(data_root=data_root, scoring_set=Set)
    path_past = novelty_config.get('past_path').format(data_root=data_root, scoring_set=Set)
    
    VD_dic = {
        f"past": get_json_files(path_con),
        f'con':get_json_files(path_con),
        f'path_past':path_past, 
        f'path_con':path_con,
       }
    k = novelty_config.get('k_similar_papers', 5)     
    scores_hyps = {'OP':[],"ON":[],'Hypotheses':[], 'gene name':[], 'XAI score':[], 'pathway name':[], 'biological_process name':[], 'molecular_function name':[]}
    
    # Load evidence percentages
    evidence_percent_file = input_config.get('evidence_percent_file')
    with open(evidence_percent_file, 'r') as f:
        evidence_percent = json.load(f)
    evidence_percent = {os.path.basename(k):v for k,v in evidence_percent.items()}
    
    # Load triplet information
    triplets_info_file = input_config.get('triplets_info_file')
    with open(triplets_info_file, 'r') as f:
        triplets_info = json.load(f)
    
    input_base_path = input_config.get('base_path')
    for triplet in [f.stem for f in Path(input_base_path).iterdir() if f.is_dir()]:
        path_res = f"{input_base_path}/{triplet}"
        os.makedirs(path_res,exist_ok=True)
        fn_res = f"{path_res}/HypothesisScores.json"

        if os.path.exists(fn_res):
            logger.info(f"File {fn_res} already exists")
            with open(fn_res, 'r') as f:
                hyp = json.load(f)
        else:
            path_hyp = f"{input_base_path}/{triplet}"
            fn_hypothesis = f"{path_hyp}/champion.json"
            with open(fn_hypothesis, 'r') as f:
                hyp = json.load(f)[0]
            hyp_abs = hyp['Abstract']
            
            embedding = get_embedding(hyp_abs)
            hyp['abstract_embedding'] = embedding

            test_scores,same_year_k,top_k_past,top_k_con = PreScoring(embedding,k,VD_dic)
            
            # Normalize scores
            HD_avg = 0
            CD_avg = 0
            CI_avg = 0
            for year in same_year_k['past']:
                scores_year = get_year_scores(path_past,year,0)
                HD_avg += scores_year['HD_avg']
            for year in same_year_k['con']:
                scores_year = get_year_scores(path_con,year,0)
                CD_avg += scores_year['CD_avg']
                CI_avg += scores_year['CI_avg']
            HD_avg /= len(same_year_k['past'])
            CD_avg /= len(same_year_k['con'])
            CI_avg /= len(same_year_k['con'])
            test_scores['HD']/= HD_avg
            test_scores['CD']/= CD_avg
            test_scores['CI']/= CI_avg
            Novelty_Scoring(test_scores)
            hyp['scores'] = test_scores
            hyp['top_k_past'] = top_k_past
            hyp['top_k_con'] = top_k_con

        with open(fn_res, "w") as f:
            json.dump(hyp, f, indent=4, ensure_ascii=False)
        hyp['scores']['OP'] = evidence_percent[triplet]
        scores_hyps['OP'].append(hyp['scores']['OP'])
        scores_hyps['ON'].append(hyp['scores']['ON'])
        scores_hyps['Hypotheses'].append(triplet)

        # Parse triplet name to get gene name and other info
        # Find the corresponding triplet info by matching gene name
        gene_name = ''
        triplet_info = None
        
        # Search through all triplets to find matching gene
        for triplet_key, info in triplets_info.items():
            if info.get('gene name', '').lower() in triplet.lower():
                gene_name = info.get('gene name', '')
                triplet_info = info
                break
        
        if triplet_info:
            scores_hyps['gene name'].append(gene_name)
            # Get XAI rank from the specific field name
            xai_field = f"XAI scores of gene '{gene_name}' and"
            xai_score = ''
            for key, value in triplet_info.items():
                if key.startswith('XAI scores of gene') and 'in' in key:
                    xai_score = value
                    break
            scores_hyps['XAI score'].append(xai_score)
            
            # Get pathway/biological_process/molecular_function/cellular_component name
            pathway_name = triplet_info.get('pathway name', '')
            biological_process_name = triplet_info.get('biological_process name', '')
            molecular_function_name = triplet_info.get('molecular_function name', '')
            cellular_component_name = triplet_info.get('cellular_component name', '')
            
            scores_hyps['pathway name'].append(pathway_name)
            scores_hyps['biological_process name'].append(biological_process_name)
            scores_hyps['molecular_function name'].append(molecular_function_name)
        else:
            # If no matching triplet found, try to extract gene name from triplet name
            parts = triplet.split('_')
            if len(parts) >= 2:
                gene_name = parts[1]
            else:
                gene_name = triplet
            
            scores_hyps['gene name'].append(gene_name)
            scores_hyps['XAI score'].append('')
            scores_hyps['pathway name'].append('')
            scores_hyps['biological_process name'].append('')
            scores_hyps['molecular_function name'].append('')
    
    # Calculate normalized scores and combined metrics
    scores_hyps['ON_norm'] = ON_norm_minmax(scores_hyps['ON'])
    scores_hyps['geo_mean'] = geometric_mean_scores(scores_hyps['ON_norm'],scores_hyps['OP']) 
    scores_hyps['har_mean'] = harmonic_mean_scores(scores_hyps['ON_norm'],scores_hyps['OP']) 
    
    print(scores_hyps)
    df_scores = pd.DataFrame(scores_hyps)
    df_scores.set_index('Hypotheses', inplace=True)
    
    column_order = ['OP', 'ON', 'ON_norm', 'geo_mean', 'har_mean', 'gene name', 'XAI score', 'pathway name', 'biological_process name', 'molecular_function name']
    df_scores = df_scores.reindex(columns=column_order)
    
    output_csv = input_config.get('output_csv')
    df_scores.to_csv(output_csv)
    
    return scores_hyps

# Main function
async def main():
    """Main function to run hypothesis evaluation"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Evaluate literature evidence
    logger.info("Starting literature evidence evaluation...")
    evidence_results, evidence_percent = await evaluate_literature_evidence()
    logger.info("Literature evidence evaluation completed.")
    
    # Evaluate novelty scores
    logger.info("Starting novelty score evaluation...")
    scores_hyps = evaluate_novelty_scores()
    logger.info("Novelty score evaluation completed.")
    
    logger.info("Hypothesis evaluation completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 